"""Optimized program combining DFT parameterization and adaptive smooth max."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 145s
    deadline = start_time + 145.0

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
        # Clip to prevent overflow in exp
        x_shifted = (x - x_max) / tau
        x_shifted = np.clip(x_shifted, -50, 50)
        return x_max + tau * np.log(np.sum(np.exp(x_shifted)))

    def dft_based_objective(params, tau=0.05):
        """
        Objective for DFT-based parameterization.
        params: [dc, r1, p1, r2, p2, ..., rK, pK]
        h[i] = dc + sum_k r_k * cos(2*pi*k*i/N + p_k)
        """
        dc = params[0]
        h = np.full(N, dc)
        n_harm = (len(params) - 1) // 2
        i_idx = np.arange(N)
        for k in range(1, n_harm + 1):
            r = params[1 + 2*(k-1)]
            p = params[2 + 2*(k-1)]
            h += r * np.cos(2*np.pi*k*i_idx/N + p)
        
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Phase 1: Differential Evolution on DFT coefficients
    # Use 10 harmonics for a good balance of expressiveness and dimensionality
    if time.time() < deadline - 60.0:
        try:
            from scipy.optimize import differential_evolution
            n_harm = 10
            bounds = [(-2.0, 2.0)]  # DC
            for k in range(1, n_harm + 1):
                bounds.append((-1.5, 1.5))  # amplitude
                bounds.append((0.0, 2.0 * np.pi))  # phase
            
            # Use a custom callback to stop if time is running out
            def callback(xk, convergence=False):
                return time.time() > deadline - 50.0

            de_result = differential_evolution(
                dft_based_objective,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=True,
                callback=callback
            )
            
            if de_result.fun < best_val:
                dc = de_result.x[0]
                h_dft = np.full(N, dc)
                i_idx = np.arange(N)
                for k in range(1, n_harm + 1):
                    r = de_result.x[1 + 2*(k-1)]
                    p = de_result.x[2 + 2*(k-1)]
                    h_dft += r * np.cos(2*np.pi*k*i_idx/N + p)
                
                val_dft = compute_value(h_dft)
                if val_dft < best_val and np.all(np.isfinite(h_dft)) and abs(np.sum(h_dft)) > 1e-6:
                    best_h = h_dft.copy()
                    best_val = val_dft
                    print(f"DFT DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT DE error: {e}", file=sys.stderr)

    # Phase 2: Local Refinement using L-BFGS-B on DFT coefficients
    if time.time() < deadline - 30.0:
        try:
            from scipy.optimize import minimize
            
            # Extract current best params if possible, or use random restarts
            # Since we don't store the best params from DE easily without extra work, 
            # we'll use the best_h to initialize a few random DFT fits or just random restarts.
            
            # A better approach: Use the result from DE if available, otherwise random.
            # For simplicity and robustness, we do a few random restarts in DFT space.
            rng = np.random.RandomState(42)
            
            for restart in range(5):
                if time.time() > deadline - 20.0:
                    break
                
                # Initialize params
                n_harm = 10
                params_init = np.zeros(1 + 2*n_harm)
                params_init[0] = rng.uniform(0.5, 1.5) # DC
                for k in range(n_harm):
                    params_init[1 + 2*k] = rng.uniform(-0.5, 0.5)
                    params_init[2 + 2*k] = rng.uniform(0, 2*np.pi)
                
                result = minimize(
                    dft_based_objective,
                    params_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.fun < best_val:
                    dc = result.x[0]
                    h_opt = np.full(N, dc)
                    i_idx = np.arange(N)
                    for k in range(1, n_harm + 1):
                        r = result.x[1 + 2*(k-1)]
                        p = result.x[2 + 2*(k-1)]
                        h_opt += r * np.cos(2*np.pi*k*i_idx/N + p)
                    
                    val_opt = compute_value(h_opt)
                    if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                        best_h = h_opt.copy()
                        best_val = val_opt
                        print(f"DFT L-BFGS restart {restart}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT L-BFGS error: {e}", file=sys.stderr)

    # Phase 3: Direct local refinement in h-space (if time permits)
    # This can escape local minima in the DFT representation
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            
            def direct_objective(h_flat):
                h = h_flat
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10
                conv = np.convolve(h, h)
                sm = smooth_max(conv, tau=0.01)
                return 2.0 * N * sm / (s * s)
            
            # Start from best_h
            res = minimize(
                direct_objective,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_val:
                h_direct = res.x
                val_direct = compute_value(h_direct)
                if val_direct < best_val and np.all(np.isfinite(h_direct)) and abs(np.sum(h_direct)) > 1e-6:
                    best_h = h_direct.copy()
                    best_val = val_direct
                    print(f"Direct L-BFGS: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Direct L-BFGS error: {e}", file=sys.stderr)

    # Final validation
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
    if abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
