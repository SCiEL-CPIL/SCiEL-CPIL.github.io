import numpy as np
import time
import sys

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    # Continuous autoconvolution of step function f(t) = sum h_i 1_{[x_i, x_{i+1})}(t)
    # The peak is proportional to the max of the discrete convolution of h with itself.
    # Specifically, max(f*f) = DELTA * max(conv(h,h))
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Objective function for minimization (negative of score)
    # We work with unnormalized h, normalize inside to handle the simplex constraint
    # via the DE bounds [0, 1].
    def objective(h):
        h = np.asarray(h, dtype=np.float64)
        s = np.sum(h)
        if s < 1e-12:
            return 1e10 # Large penalty
        h_norm = h / s
        score = _score_raw(h_norm)
        return -score

    # Define bounds for Differential Evolution
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    best_score = 0.0
    best_h = np.ones(N) / N
    
    # 1. Run Differential Evolution
    # DE is a global optimizer, good for non-convex landscapes.
    from scipy.optimize import differential_evolution
    
    try:
        # Strategy: Run DE with a reasonable number of iterations.
        # Then refine the best result.
        # popsize=25, maxiter=1500 is a good balance for N=50 within time limits.
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=1500, 
            popsize=25, 
            tol=1e-8,
            init='latinhypercube',
            polish=False,
            workers=1
        )
        
        if res.fun < 0:
            h_de = res.x
            s_de = -res.fun
            if s_de > best_score:
                best_score = s_de
                best_h = h_de.copy()
                
    except Exception as e:
        print(f"DE failed: {e}")

    # 2. Local Refinement
    # Use COBYLA on the best candidate found.
    from scipy.optimize import minimize
    
    if time.time() < deadline - 10:
        
        def neg_score_constrained(h):
            h = np.asarray(h, dtype=np.float64)
            # COBYLA might violate bounds slightly, clip
            h = np.clip(h, 0, 1)
            s = np.sum(h)
            if s < 1e-12:
                return 1e10
            h_norm = h / s
            return -_score_raw(h_norm)

        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
            {'type': 'ineq', 'fun': lambda h: h}
        ]
        
        try:
            res_local = minimize(
                neg_score_constrained, 
                best_h, 
                method='COBYLA', 
                constraints=constraints, 
                options={'maxiter': 5000, 'rhobeg': 1e-4, 'catol': 1e-8}
            )
            
            if res_local.fun < -best_score:
                h_cand = res_local.x
                s_cand = -res_local.fun
                if s_cand > best_score:
                    best_score = s_cand
                    best_h = h_cand
        
        except Exception as e:
            print(f"Local refinement failed: {e}")
            
    # 3. Final check and return
    # Ensure the result is a list of floats
    # Normalize one last time just in case
    s_final = np.sum(best_h)
    if s_final > 0:
        best_h = best_h / s_final
    
    return list(best_h)
