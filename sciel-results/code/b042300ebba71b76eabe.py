"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def smooth_max(x, tau=0.01):
        """Soft approximation of max using log-sum-exp."""
        x_max = np.max(x)
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    def smooth_max_adaptive(x, tau_base=0.05, tau_min=0.001):
        """Adaptive smooth max: starts loose, tightens as optimization progresses."""
        # Use a schedule based on the spread of x
        x_max = np.max(x)
        x_spread = np.max(x) - np.min(x)
        if x_spread < 1e-6:
            tau = tau_min
        else:
            # Scale tau with the spread to maintain relative smoothness
            tau = max(tau_min, min(tau_base, 0.1 * x_spread))
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    def objective_signed(params, tau=0.05):
        """Objective allowing signed h, with penalty if sum is small."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max_adaptive(conv, tau_base=tau, tau_min=0.001)
        return 2.0 * N * sm / (s * s)

    def dft_based_objective(params, tau=0.05):
        """Objective for DFT-based parameterization.
        params: [dc, r1, p1, r2, p2, ..., r10, p10]
        h[i] = dc + sum_k r_k * cos(2*pi*k*i/N + p_k)
        """
        dc = params[0]
        h = np.full(N, dc)
        n_harm = (len(params) - 1) // 2
        for k in range(1, n_harm + 1):
            r = params[1 + 2*(k-1)]
            p = params[2 + 2*(k-1)]
            h += r * np.cos(2*np.pi*k*np.arange(N)/N + p)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max_adaptive(conv, tau_base=tau, tau_min=0.001)
        return 2.0 * N * sm / (s * s)

    def hadamard_objective(coeffs, tau=0.05):
        """Objective for Hadamard-basis parameterization.
        coeffs: [c0, c1, ..., c_{N-1}] where h = H @ coeffs / sqrt(N)
        c0 is the DC component (sum-related), others are zero-mean.
        """
        h = H @ coeffs / np.sqrt(N)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    # Precompute Hadamard matrix for N=400 (not power of 2, so use DCT-II as proxy
    # or construct via Kronecker products of smaller Hadamards + padding)
    # For N=400 = 16 * 25, we can't use standard Hadamard directly.
    # Instead, use DCT-II basis which is also orthogonal and real.
    # DCT-II basis: phi_k[i] = cos(pi * k * (2*i + 1) / (2*N))
    # This is orthogonal and well-suited for spectral shaping.
    
    # Build DCT-II basis matrix
    i_idx = np.arange(N, dtype=float)
    k_idx = np.arange(N, dtype=float)
    # DCT-II: phi[k, i] = cos(pi * k * (2*i + 1) / (2*N))
    # For efficiency, precompute
    dct_basis = np.cos(np.pi * np.outer(k_idx, (2*i_idx + 1)) / (2*N))
    # Normalize columns
    dct_norm = np.sqrt(np.sum(dct_basis**2, axis=0))
    dct_basis = dct_basis / dct_norm

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Phase 1: DFT-based global search via differential evolution
    # Use more harmonics (10) for richer representation
    if time.time() < deadline - 50.0:
        try:
            from scipy.optimize import differential_evolution
            n_harm = 10
            bounds = [(-2.0, 2.0)]  # DC
            for k in range(1, n_harm + 1):
                bounds.append((-1.5, 1.5))  # amplitude
                bounds.append((0.0, 2.0 * np.pi))  # phase
            de_result = differential_evolution(
                dft_based_objective,
                bounds,
                maxiter=40,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                dc = de_result.x[0]
                h_dft = np.full(N, dc)
                for k in range(1, n_harm + 1):
                    r = de_result.x[1 + 2*(k-1)]
                    p = de_result.x[2 + 2*(k-1)]
                    h_dft += r * np.cos(2*np.pi*k*np.arange(N)/N + p)
                val_dft = compute_value(h_dft)
                if val_dft < best_val and np.all(np.isfinite(h_dft)) and abs(np.sum(h_dft)) > 1e-6:
                    best_h = h_dft.copy()
                    best_val = val_dft
                    print(f"DFT DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT DE error: {e}", file=sys.stderr)

    # Phase 2: DCT-spectral shaping via projected gradient descent
    # Optimize in DCT coefficient space using gradient descent
    if time.time() < deadline - 40.0:
        try:
            from scipy.optimize import minimize
            
            # Multiple restarts in DCT coefficient space
            rng = np.random.RandomState(123)
            for restart in range(6):
                if time.time() > deadline - 30.0:
                    break
                
                # Initialize DCT coefficients
                if restart == 0:
                    # Start from all-ones in DCT space
                    c0 = dct_basis[0] @ best_h
                    c_init = np.zeros(N)
                    c_init[0] = c0
                elif restart == 1:
                    # Start from best_h projected onto DCT
                    c_init = dct_basis @ best_h
                else:
                    # Random low-frequency initialization with varying bandwidth
                    c_init = np.zeros(N)
                    n_low = rng.randint(8, 25)
                    for k in range(1, n_low):
                        c_init[k] = rng.uniform(-0.5, 0.5)
                    c_init[0] = rng.uniform(0.5, 2.0)
                
                def dct_objective(c, tau=0.05):
                    h = dct_basis.T @ c
                    s = np.sum(h)
                    if abs(s) < 1e-4:
                        return 1e10 + (1e-4 - abs(s)) * 1e6
                    conv = np.convolve(h, h)
                    sm = smooth_max_adaptive(conv, tau_base=tau, tau_min=0.001)
                    return 2.0 * N * sm / (s * s)
                
                result = minimize(
                    dct_objective,
                    c_init,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                h_opt = dct_basis.T @ result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"DCT restart {restart}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DCT optimization error: {e}", file=sys.stderr)

    # Phase 3: Local refinement in direct h-space using L-BFGS-B
    if time.time() < deadline - 20.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_signed,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Direct L-BFGS-B: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Direct L-BFGS-B error: {e}", file=sys.stderr)

    # Phase 4: High-precision polish with very tight smooth max
    # Use multiple passes with decreasing tau to converge to the true max
    if time.time() < deadline - 10.0 and best_val < 2.0:
        try:
            from scipy.optimize import minimize
            
            # Progressive tightening of tau
            for tau_pass in [0.01, 0.005, 0.002]:
                if time.time() > deadline - 5.0:
                    break
                
                def objective_tight(params, tau=tau_pass):
                    h = params
                    s = np.sum(h)
                    if abs(s) < 1e-4:
                        return 1e10 + (1e-4 - abs(s)) * 1e6
                    conv = np.convolve(h, h)
                    sm = smooth_max(conv, tau=tau)
                    return 2.0 * N * sm / (s * s)
                
                result = minimize(
                    objective_tight,
                    best_h,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
                )
                h_opt = result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"Polish tau={tau_pass}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Polish error: {e}", file=sys.stderr)

    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
