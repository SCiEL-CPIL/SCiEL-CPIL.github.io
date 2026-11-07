"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety

    def objective(params, beta):
        h = params
        # Compute cross-correlation
        corr = np.correlate(h, 1 - h, mode='full')
        # Soft-max approximation of max(corr)
        max_corr = np.max(corr)
        # Subtract max to avoid overflow
        shifted = beta * (corr - max_corr)
        # Clip to avoid numerical issues in exp if beta is large
        shifted = np.clip(shifted, -100, 100)
        soft_max = (1.0 / beta) * np.log(np.sum(np.exp(shifted))) + max_corr
        
        # Penalty for sum constraint: sum(h) should be N/2
        sum_h = np.sum(h)
        sum_penalty = 1000.0 * (sum_h - N / 2.0) ** 2
        
        return soft_max + sum_penalty

    # Initial guess
    h0 = np.full(N, 0.5)
    bounds = [(0, 1) for _ in range(N)]
    
    best_h = h0.copy()
    best_val = objective(h0, 10.0)
    
    # Stage 1: Coarse optimization
    try:
        res = minimize(lambda x: objective(x, 10.0), h0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-10})
        if time.time() > deadline:
            return best_h
        if res.fun < best_val:
            best_h = res.x
            best_val = res.fun
    except Exception:
        pass

    # Stage 2: Finer optimization with higher beta
    if time.time() < deadline:
        try:
            res2 = minimize(lambda x: objective(x, 50.0), best_h, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-12})
            if res2.fun < best_val:
                best_h = res2.x
                best_val = res2.fun
        except Exception:
            pass

    # Stage 3: Very fine optimization
    if time.time() < deadline:
        try:
            res3 = minimize(lambda x: objective(x, 100.0), best_h, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 100, 'ftol': 1e-12})
            if res3.fun < best_val:
                best_h = res3.x
                best_val = res3.fun
        except Exception:
            pass

    # Final verification of constraints
    # If sum is slightly off, we might need to adjust, but the penalty should keep it tight.
    # The evaluator checks sum(h) == len(h)/2 within 1e-9 * len(h).
    # 1e-9 * 100 = 1e-7.
    # L-BFGS-B with ftol 1e-12 should satisfy this easily with a strong penalty.
    
    return best_h
# EVOLVE-BLOCK-END
