"""Optimized program for the third autocorrelation inequality task using Zadoff-Chu complementary sequences."""
import numpy as np
import time
import sys

N = 400

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

    def generate_zadoff_chu(root, length):
        """Generate a Zadoff-Chu sequence."""
        n = np.arange(length)
        # Standard ZC sequence definition
        if root % 4 == 0 or root % 4 == 3:
            exp_term = -1j * np.pi * (n * (n + 1)) / length
        else:
            exp_term = -1j * np.pi * (n * (n + 1)) / length
        return np.exp(exp_term)

    def smooth_max(x, tau=0.01):
        """Soft approximation of max using log-sum-exp."""
        x_max = np.max(x)
        if tau < 1e-8:
            return x_max
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    def objective_from_params(params, tau=0.05):
        """
        Objective function based on Zadoff-Chu parameters.
        params: [a1, a2, p1, p2]
        a1, a2: amplitudes (real)
        p1, p2: phases (radians)
        
        h = a1 * real(ZC * exp(1j*p1)) + a2 * real(ZC_shifted * exp(1j*p2))
        """
        a1, a2, p1, p2 = params
        # Use a root that gives good properties. For N=400, we need a root coprime to N.
        # N=400 = 16 * 25. Roots like 3, 7, etc.
        # Let's use two different roots or shifts.
        # ZC sequences are defined for prime powers. 400 is not prime power.
        # We can generate a ZC of length L (prime power near 400) and truncate/pad, 
        # or use a generalized chirp.
        # Let's use a generalized chirp: exp(-1j * pi * k * n^2 / N)
        # This is a ZC-like sequence for any N.
        
        n = np.arange(N)
        # Chirp 1
        zc1 = np.exp(-1j * np.pi * 3 * n * n / N)
        # Chirp 2 (different root/shift)
        zc2 = np.exp(-1j * np.pi * 7 * (n + N//4) * (n + N//4) / N)
        
        h = a1 * np.real(zc1 * np.exp(1j * p1)) + a2 * np.real(zc2 * np.exp(1j * p2))
        
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    def objective_direct(params, tau=0.05):
        """Objective for direct h-space refinement."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Phase 1: Global search over ZC parameters
    if time.time() < deadline - 60.0:
        try:
            from scipy.optimize import differential_evolution
            # Bounds: amplitudes can be negative, phases 0 to 2pi
            bounds = [
                (-2.0, 2.0),  # a1
                (-2.0, 2.0),  # a2
                (0.0, 2.0 * np.pi),  # p1
                (0.0, 2.0 * np.pi)   # p2
            ]
            
            de_result = differential_evolution(
                objective_from_params,
                bounds,
                maxiter=60,
                popsize=20,
                seed=42,
                tol=1e-6,
                polish=True
            )
            
            if de_result.fun < best_val:
                a1, a2, p1, p2 = de_result.x
                n = np.arange(N)
                zc1 = np.exp(-1j * np.pi * 3 * n * n / N)
                zc2 = np.exp(-1j * np.pi * 7 * (n + N//4) * (n + N//4) / N)
                h_zc = a1 * np.real(zc1 * np.exp(1j * p1)) + a2 * np.real(zc2 * np.exp(1j * p2))
                
                val_zc = compute_value(h_zc)
                if val_zc < best_val and np.all(np.isfinite(h_zc)) and abs(np.sum(h_zc)) > 1e-6:
                    best_h = h_zc.copy()
                    best_val = val_zc
                    print(f"ZC DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"ZC DE error: {e}", file=sys.stderr)

    # Phase 2: Coordinate-wise descent on direct h-space
    # This is a structural change: instead of just gradient descent, we do cyclic coordinate descent
    # which can navigate non-smooth landscapes better when combined with smooth surrogate.
    if time.time() < deadline - 30.0:
        try:
            h_curr = best_h.copy()
            val_curr = best_val
            
            # Adaptive step size
            step_size = 0.01
            no_improve_count = 0
            
            # Perform a limited number of passes
            max_passes = 10
            for pass_idx in range(max_passes):
                if time.time() > deadline - 25.0:
                    break
                
                improved_in_pass = False
                # Randomize start index to avoid bias
                start_idx = np.random.randint(N)
                
                for i in range(N):
                    if time.time() > deadline - 25.0:
                        break
                    
                    idx = (start_idx + i) % N
                    orig_val = h_curr[idx]
                    
                    # Try a few perturbations
                    best_local_val = val_curr
                    best_local_h = orig_val
                    
                    # Grid search around current value
                    # Range of search depends on magnitude
                    search_range = max(0.1, abs(orig_val) * 0.5)
                    candidates = np.linspace(orig_val - search_range, orig_val + search_range, 5)
                    
                    for cand in candidates:
                        h_test = h_curr.copy()
                        h_test[idx] = cand
                        val_test = compute_value(h_test)
                        if val_test < best_local_val:
                            best_local_val = val_test
                            best_local_h = cand
                    
                    if best_local_val < val_curr - 1e-8:
                        h_curr[idx] = best_local_h
                        val_curr = best_local_val
                        improved_in_pass = True
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                    # Early stop if stuck
                    if no_improve_count > N:
                        break
                
                if not improved_in_pass:
                    break
                else:
                    # Decay step size if we are making progress but maybe too large? 
                    # Or keep it if small steps are working.
                    pass
                
                if val_curr < best_val:
                    best_h = h_curr.copy()
                    best_val = val_curr
                    print(f"Coord Descent Pass {pass_idx}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Coord Descent error: {e}", file=sys.stderr)

    # Phase 3: Local refinement with L-BFGS-B on the best found so far
    if time.time() < deadline - 15.0:
        try:
            from scipy.optimize import minimize
            
            # Use a tight smooth max for the final polish
            def objective_tight(params, tau=0.005):
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
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Final L-BFGS-B: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Final L-BFGS-B error: {e}", file=sys.stderr)

    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)
