import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

    m = 12
    max_deg = 2 * m

    # Precompute the transformation matrix from standard basis to Laguerre basis
    A = np.zeros((max_deg + 1, max_deg + 1))
    for k in range(max_deg + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    weights = np.arange(max_deg + 1) + 1

    def compute_norm_sq_from_roots(roots):
        """
        Compute the squared norm of P(x) = prod (x - r_i)^2 in the Laguerre L_k^(1) basis.
        """
        coeffs_Q_desc = np.poly(roots)
        coeffs_P_desc = np.convolve(coeffs_Q_desc, coeffs_Q_desc)
        p = coeffs_P_desc[::-1]
        
        if len(p) < max_deg + 1:
            p = np.pad(p, (0, max_deg + 1 - len(p)))
        else:
            p = p[:max_deg + 1]
            
        c = A_T_inv @ p
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func_roots(params):
        """
        Minimize 1/||P||^2. params are the 12 positive roots directly.
        """
        if time.time() > deadline:
            return 1e10
            
        roots = params
        if np.any(roots <= 1e-8):
            return 1e10 - np.min(roots)
            
        norm_sq = compute_norm_sq_from_roots(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    # Generate diverse initial seeds
    seeds = []
    
    # Seed 1: Geometric sequence
    seeds.append(4.0 * (2.0 ** np.arange(m)))
    
    # Seed 2: Roots of Laguerre Polynomial L_12^(1)(x)
    from scipy.special import roots_genlaguerre
    try:
        x_lag, w_lag = roots_genlaguerre(m, 1)
        seeds.append(x_lag.copy())
    except:
        pass
    
    # Seed 3: Linearly spaced
    seeds.append(np.linspace(0.5, 50.0, m))
    
    # Seed 4: Exponential spacing
    seeds.append(np.exp(np.linspace(0, 3.5, m)))
    
    # Seed 5: Clustered near small values
    seeds.append(np.abs(np.random.default_rng(42).normal(2.0, 1.5, m)) + 0.1)
    
    # Seed 6: Wide spread
    seeds.append(np.abs(np.random.default_rng(123).uniform(0.5, 100.0, m)))
    
    # Evaluate all seeds
    best_val = 1e10
    best_params = seeds[0]
    
    for seed in seeds:
        val = cost_func_roots(seed)
        if val < best_val:
            best_val = val
            best_params = seed.copy()
    
    # Phase 1: Local polish from best seed
    bounds = [(1e-8, 1000.0)] * m
    
    if time.time() < deadline - 60:
        try:
            res = minimize(cost_func_roots, best_params, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Differential evolution for global search
    if time.time() < deadline - 30:
        try:
            de_res = differential_evolution(
                cost_func_roots, 
                bounds=bounds,
                seed=42,
                maxiter=300,
                tol=1e-10,
                popsize=20,
                mutation=(0.3, 1.0),
                recombination=0.9,
                polish=True,
                init='sobol'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res2 = minimize(cost_func_roots, best_params, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Ensure roots are positive and sorted
    roots = np.maximum(best_params, 1e-8)
    roots = np.sort(roots)
    
    return list(roots)
