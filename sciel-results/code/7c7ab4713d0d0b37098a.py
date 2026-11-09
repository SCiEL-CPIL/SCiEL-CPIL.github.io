import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 570

    m = 12

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    weight_vec = np.arange(m + 1) + 1

    def compute_norm_sq(roots):
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = A_T_inv @ p
        norm_sq = np.dot(c**2, weight_vec)
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    def make_geo_init(a, q, n=12):
        return np.log(a) + np.arange(n) * np.log(q)
    
    # Generate diverse initializations
    inits = []
    # Geometric sequences
    for a in [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
        for q in [1.3, 1.5, 1.7, 1.8, 2.0, 2.2, 2.5, 2.8]:
            init = make_geo_init(a, q)
            if np.all(np.isfinite(init)) and np.all(np.diff(init) > 0):
                inits.append(init)
    
    # Arithmetically-spaced logs
    for start_log in np.logspace(-1, 1, 15):
        for span in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            init = np.linspace(start_log, start_log + span, 12)
            if np.all(np.diff(init) > 0):
                inits.append(init)
    
    # Add seed
    inits.append(make_geo_init(4.0, 2.0))
    
    # Random perturbations
    rng = np.random.RandomState(42)
    for _ in range(30):
        base_idx = rng.randint(len(inits))
        base = inits[base_idx].copy()
        perturbation = rng.randn(12) * rng.uniform(0.1, 0.5)
        perturbed = base + perturbation
        if np.all(np.diff(perturbed) > 0):
            inits.append(perturbed)
    
    # Random uniform in log-space
    for _ in range(20):
        lo = rng.uniform(0.1, 3.0)
        hi = rng.uniform(5.0, 12.0)
        init = np.sort(rng.uniform(lo, hi, 12))
        inits.append(init)
    
    bounds = [(-1.0, 12.0)] * 12

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Phase 1: Local optimization from multiple starting points
    for i, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 150:
            break
        
        try:
            result = minimize(
                cost_func, init, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Phase 2: Differential evolution for global exploration
    seeds = [7, 17, 23, 42, 99, 123, 256, 512, 777]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 80:
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=100, 
                popsize=15, 
                tol=1e-9, 
                polish=True,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback: seed initialization
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_log_roots = np.log(seed_roots)
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots.copy()
        best_log_roots = seed_log_roots.copy()

    # Phase 3: Final high-precision polish
    if time.time() - start_time < deadline - 10 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Second polish round if time permits
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish_result2 = minimize(
                cost_func, best_log_roots, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if polish_result2.fun < best_val:
                best_val = polish_result2.fun
                best_log_roots = polish_result2.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive and sorted
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-8)
        best_roots = np.sort(best_roots)
        return list(best_roots)
    
    return [4.0 * (2.0 ** i) for i in range(12)]
