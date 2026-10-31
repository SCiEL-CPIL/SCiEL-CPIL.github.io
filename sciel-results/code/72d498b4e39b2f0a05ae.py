"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50

# EVOLVE-BLOCK-START
def construct():
    # We want to maximize R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # Fix ||h||_1 = 1. Maximize ||h||_2^2 / ||h*h||_inf.
    # Equivalent to minimizing ||h*h||_inf / ||h||_2^2.
    # The ratio is scale-invariant, so we can search over [0,1]^N and normalize.
    
    def objective(h):
        # h is a vector of length N
        # Minimize max_conv / l2_sq
        if np.sum(h) < 1e-10:
            return 1e10
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-10:
            return 1e10
        if max_conv < 1e-10:
            return 1e10
        return max_conv / l2_sq

    # Step 1: Global search using Differential Evolution
    # DE handles bounds but not equality constraints. 
    # Since the objective is scale-invariant, we don't need sum(h)=1 during search.
    bounds = [(0, 1) for _ in range(N)]
    
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            popsize=20, 
            tol=1e-8, 
            maxiter=500, 
            seed=42, 
            polish=False
        )
        h_de = res_de.x
    except Exception:
        # Fallback to uniform if DE fails
        h_de = np.ones(N) / N

    # Step 2: Local refinement with SLSQP enforcing sum(h)=1
    def constraint_sum(h):
        return np.sum(h) - 1.0

    cons = [{'type': 'eq', 'fun': constraint_sum}]
    bounds_slsqp = [(0, 1) for _ in range(N)]
    
    try:
        res_slsqp = minimize(
            objective, 
            h_de, 
            method='SLSQP', 
            bounds=bounds_slsqp, 
            constraints=cons, 
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        h_opt = res_slsqp.x
    except Exception:
        h_opt = h_de

    # Normalize to ensure L1 = 1
    l1 = np.sum(h_opt)
    if l1 > 0:
        h_opt = h_opt / l1
    else:
        h_opt = np.ones(N) / N

    return h_opt
# EVOLVE-BLOCK-END
