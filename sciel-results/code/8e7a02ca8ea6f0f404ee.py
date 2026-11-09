"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def heuristics(prizes, weights):
    # Time guard: 330s limit, keep margin
    start_time = time.time()
    if time.time() - start_time > 290:
        # Fallback to simple heuristic if already close to deadline
        p = prizes.sum(axis=0) if prizes.ndim == 2 else prizes
        return p / (weights.sum(axis=0) + 1e-9)

    C = 25.0
    n_items = weights.shape[1]
    
    # Determine prize vector
    if prizes.ndim == 2:
        p = prizes.sum(axis=0)
    else:
        p = prizes

    # LP Relaxation
    # Maximize sum(p_j * x_j)
    # Subject to sum(w_cj * x_j) <= C for all c
    # 0 <= x_j <= 1
    
    # Convert to Minimize -sum(p_j * x_j)
    c_lp = -p
    
    # A_ub: (n_constraints, n_items)
    # weights is (5, 100). This is already in the right shape for A_ub.
    A_ub = weights
    b_ub = np.full(weights.shape[0], C)
    
    # Bounds
    bounds = [(0, 1)] * n_items
    
    try:
        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success and res.ineqlin is not None and res.ineqlin.marginals is not None:
            lambdas = res.ineqlin.marginals
            
            # Check for non-finite duals
            if not np.all(np.isfinite(lambdas)):
                raise ValueError("Non-finite duals")
            
            # Reduced Costs: p_j - sum_c lambda_c * w_cj
            # lambdas shape (5,), weights.T shape (100, 5)
            # Result shape (100,)
            weighted_shadow_cost = lambdas @ weights.T
            reduced_costs = p - weighted_shadow_cost
            
            # Heuristic: Items with positive reduced cost are "profitable" relative to constraints
            # Clip to 1e-9 to ensure positivity for ACO
            heur = np.maximum(reduced_costs, 1e-9)
            
            # Ensure finite
            heur = np.where(np.isfinite(heur), heur, 1e-9)
            
            return heur
        else:
            # Fallback
            return p / (weights.sum(axis=0) + 1e-9)
            
    except Exception:
        # Fallback on any error
        return p / (weights.sum(axis=0) + 1e-9)
# EVOLVE-BLOCK-END
