import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.special import roots_laguerre
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    n = 1

    # Precomputation: Transformation matrix from standard basis to Laguerre basis
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    weights = np.arange(13) + 1

    def compute_cost_from_roots(roots):
        """Compute cost given positive roots."""
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e20
        
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e20
        
        return 1.0 / norm_sq

    def compute_cost(X_flat):
        """
        Compute cost given X (flattened 6x12 matrix).
        D = D0 + X.T @ X ensures positive definiteness.
        Eigenvalues of D are the roots.
        """
        if time.time() > deadline:
            return 1e20
        
        try:
            X = X_flat.reshape(6, 12)
            D = D0 + X.T @ X
            # D is guaranteed SPD
            eigvals = np.linalg.eigvalsh(D)
            roots = np.sort(eigvals)
            
            if np.any(roots <= 0) or np.any(np.isnan(roots)):
                return 1e20
            
            return compute_cost_from_roots(roots)
        except Exception:
            return 1e20

    # Fixed positive definite baseline
    D0 = np.eye(12) * 1.0

    best_val = 1e20
    best_X_flat = None

    def update_best(val, x):
        nonlocal best_val, best_X_flat
        if val < best_val:
            best_val = val
            best_X_flat = x.copy()

    # --- Phase 1: Smart Initial Guesses ---
    initial_points = []
    rng = np.random.RandomState(42)

    # 1. Start from known good roots (Laguerre roots)
    try:
        l_roots, _ = roots_laguerre(12, 1)
        l_roots = np.sort(l_roots)
        
        # Find X such that D0 + X.T @ X has eigenvalues close to l_roots
        # This is hard, so we'll just use random X and let optimization work
        for _ in range(5):
            X_init = rng.normal(0, 1.0, size=(6, 12))
            initial_points.append(X_init.flatten())
    except Exception:
        pass

    # 2. Random initializations with varying scales
    for scale in [0.5, 1.0, 2.0, 5.0]:
        for _ in range(3):
            X_init = rng.normal(0, scale, size=(6, 12))
            initial_points.append(X_init.flatten())

    # 3. Structured initializations
    for i in range(5):
        X_init = np.zeros((6, 12))
        for j in range(6):
            X_init[j, :] = rng.normal(0, 1.0 + i*0.5, size=12)
        initial_points.append(X_init.flatten())

    # --- Phase 2: Basin Hopping Search ---
    local_opts = {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
    
    def step(x):
        step_size = 0.3
        return x + rng.normal(0, step_size, size=x.shape)

    num_bh_runs = 80
    maxiter_bh = 25

    for i in range(num_bh_runs):
        if time.time() > deadline - 15:
            break
            
        if i < len(initial_points):
            x0 = initial_points[i]
        else:
            x0 = rng.normal(0, 1.0, size=72)
        
        try:
            res = basinhopping(
                compute_cost,
                x0,
                stepfunc=step,
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': local_opts},
                niter=maxiter_bh,
                seed=i
            )
            
            if res.fun < best_val:
                update_best(res.fun, res.x)
                
            if best_val < 1e-8 and time.time() < deadline - 10:
                res_polish = minimize(
                    compute_cost, 
                    res.x, 
                    method='L-BFGS-B', 
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if res_polish.fun < best_val:
                    update_best(res_polish.fun, res_polish.x)
                    
        except Exception:
            try:
                res_loc = minimize(
                    compute_cost, 
                    x0, 
                    method='L-BFGS-B', 
                    options=local_opts
                )
                if res_loc.fun < best_val:
                    update_best(res_loc.fun, res_loc.x)
            except Exception:
                pass

    # --- Phase 3: Final Polish ---
    if best_X_flat is not None:
        try:
            res_final = minimize(
                compute_cost, 
                best_X_flat, 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_X_flat = res_final.x
        except Exception:
            pass

    # Convert back to roots
    if best_X_flat is not None:
        X = best_X_flat.reshape(6, 12)
        D = D0 + X.T @ X
        best_roots = np.sort(np.linalg.eigvalsh(D))
        best_roots = np.maximum(best_roots, 1e-10)
    else:
        # Fallback
        l_roots, _ = roots_laguerre(12, 1)
        best_roots = np.sort(l_roots)
        best_val = compute_cost_from_roots(best_roots)
    
    return list(best_roots)

if __name__ == "__main__":
    roots = construct()
    print(roots)
