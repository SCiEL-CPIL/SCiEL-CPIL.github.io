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

    def objective_signed(params, tau=0.05):
        """Objective allowing signed h, with penalty if sum is small."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    def dft_based_objective(params, tau=0.05):
        """Objective for DFT-based parameterization.
        params: [dc, r1, p1, r2, p2, r3, p3, r4, p4, r5, p5, r6, p6]
        h[i] = dc + sum_k r_k * cos(2*pi*k*i/N + p_k)
        """
        dc = params[0]
        i_arr = np.arange(N)
        h = np.full(N, dc)
        for k in range(1, 7):
            r = params[1 + 2*(k-1)]
            p = params[2 + 2*(k-1)]
            h += r * np.cos(2*np.pi*k*i_arr/N + p)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    def zc_sequence(n, u, v):
        """Generate Zadoff-Chu sequence: z[i] = exp(-j*pi*u*v*i*(i+1)/n) for i=0..n-1.
        Returns real-valued sequence by taking real part."""
        i = np.arange(n)
        phase = -np.pi * u * v * i * (i + 1) / n
        return np.cos(phase)

    def zc_objective(params, tau=0.05):
        """Objective for Zadoff-Chu parameters: [u, v, scale, offset, phase_shift]."""
        u, v, scale, offset, phase_shift = params
        u_int = int(np.round(u))
        v_int = int(np.round(v))
        if u_int < 1: u_int = 1
        if u_int >= N: u_int = N - 1
        if v_int < 1: v_int = 1
        if v_int >= N: v_int = N - 1
        
        zc = zc_sequence(N, u_int, v_int)
        shift = int(np.round(phase_shift * N)) % N
        if shift != 0:
            zc = np.roll(zc, shift)
        
        h = scale * zc + offset
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Phase 1: DFT-based global search via differential evolution
    if time.time() < deadline - 60.0:
        try:
            from scipy.optimize import differential_evolution
            bounds = [(-2.0, 2.0)]  # DC
            for k in range(1, 7):
                bounds.append((-1.5, 1.5))  # amplitude
                bounds.append((0.0, 2.0 * np.pi))  # phase
            de_result = differential_evolution(
                dft_based_objective,
                bounds,
                maxiter=30,
                popsize=12,
                seed=42,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                dc = de_result.x[0]
                i_arr = np.arange(N)
                h_dft = np.full(N, dc)
                for k in range(1, 7):
                    r = de_result.x[1 + 2*(k-1)]
                    p = de_result.x[2 + 2*(k-1)]
                    h_dft += r * np.cos(2*np.pi*k*i_arr/N + p)
                val_dft = compute_value(h_dft)
                if val_dft < best_val and np.all(np.isfinite(h_dft)) and abs(np.sum(h_dft)) > 1e-6:
                    best_h = h_dft.copy()
                    best_val = val_dft
                    print(f"DFT DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT DE error: {e}", file=sys.stderr)

    # Phase 2: Zadoff-Chu based search
    if time.time() < deadline - 50.0:
        try:
            from scipy.optimize import differential_evolution
            valid_u = [u for u in range(1, N) if np.gcd(u, N) == 1]
            valid_v = [v for v in range(1, N) if np.gcd(v, N) == 1]
            
            rng_de = np.random.RandomState(42)
            u_samples = rng_de.choice(valid_u, size=20, replace=False)
            v_samples = rng_de.choice(valid_v, size=20, replace=False)
            
            bounds = [
                (min(u_samples), max(u_samples)),
                (min(v_samples), max(v_samples)),
                (0.5, 2.0),
                (-0.5, 0.5),
                (0.0, 1.0)
            ]
            
            de_result = differential_evolution(
                zc_objective,
                bounds,
                maxiter=25,
                popsize=10,
                seed=42,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                u, v, scale, offset, phase_shift = de_result.x
                u_int = int(np.round(u))
                v_int = int(np.round(v))
                if u_int < 1: u_int = 1
                if u_int >= N: u_int = N - 1
                if v_int < 1: v_int = 1
                if v_int >= N: v_int = N - 1
                
                zc = zc_sequence(N, u_int, v_int)
                shift = int(np.round(phase_shift * N)) % N
                if shift != 0:
                    zc = np.roll(zc, shift)
                h_zc = scale * zc + offset
                val_zc = compute_value(h_zc)
                if val_zc < best_val and np.all(np.isfinite(h_zc)) and abs(np.sum(h_zc)) > 1e-6:
                    best_h = h_zc.copy()
                    best_val = val_zc
                    print(f"ZC DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"ZC DE error: {e}", file=sys.stderr)

    # Phase 3: DCT-spectral shaping via L-BFGS-B in coefficient space
    if time.time() < deadline - 30.0:
        try:
            from scipy.optimize import minimize
            
            # Build DCT-II basis
            i_idx = np.arange(N, dtype=float)
            k_idx = np.arange(N, dtype=float)
            dct_basis = np.cos(np.pi * np.outer(k_idx, (2*i_idx + 1)) / (2*N))
            dct_norm = np.sqrt(np.sum(dct_basis**2, axis=0))
            dct_basis = dct_basis / dct_norm
            
            rng = np.random.RandomState(123)
            for restart in range(4):
                if time.time() > deadline - 25.0:
                    break
                
                if restart == 0:
                    c0 = dct_basis[0] @ best_h
                    c_init = np.zeros(N)
                    c_init[0] = c0
                else:
                    c_init = np.zeros(N)
                    n_low = rng.randint(5, 15)
                    for k in range(1, n_low):
                        c_init[k] = rng.uniform(-0.5, 0.5)
                    c_init[0] = rng.uniform(0.5, 2.0)
                
                def dct_objective(c, tau=0.05):
                    h = dct_basis.T @ c
                    s = np.sum(h)
                    if abs(s) < 1e-4:
                        return 1e10 + (1e-4 - abs(s)) * 1e6
                    conv = np.convolve(h, h)
                    sm = smooth_max(conv, tau=tau)
                    return 2.0 * N * sm / (s * s)
                
                result = minimize(
                    dct_objective,
                    c_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = dct_basis.T @ result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"DCT restart {restart}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DCT optimization error: {e}", file=sys.stderr)

    # Phase 4: Local refinement in direct h-space
    if time.time() < deadline - 15.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_signed,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Direct L-BFGS-B: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Direct L-BFGS-B error: {e}", file=sys.stderr)

    # Phase 5: High-precision polish with tighter smooth max
    if time.time() < deadline - 5.0 and best_val < 2.0:
        try:
            from scipy.optimize import minimize
            
            def objective_tight(params, tau=0.01):
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
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Tight polish: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Tight polish error: {e}", file=sys.stderr)

    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)
    
    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
