# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _evaluate_measure_fast(support, weights):
    """
    Fast evaluation for fixed support.
    """
    n = len(support)
    w = weights / np.sum(weights)
    
    # Precompute pairwise sums and their probabilities
    S = support[:, None] + support[None, :]
    W_pair = w[:, None] * w[None, :]
    
    S_flat = S.flatten()
    W_pair_flat = W_pair.flatten()
    
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_pair_flat[sort_idx]
    
    CDF = np.cumsum(W_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            target = 2.0 * support[l] - support[k]
            idx = np.searchsorted(S_sorted, target, side='left')
            prob_y = CDF[idx-1] if idx > 0 else 0.0
            P += w[k] * w[l] * prob_y
            
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590
    
    best_score = -np.inf
    best_support = None
    best_weights = None
    
    # Define a set of candidate support grids
    # We will optimize weights for each grid.
    
    grids = []
    
    # 1. Uniform grids of varying sizes
    for n in [4, 5, 6, 8, 10, 12]:
        grids.append(np.linspace(0, 1, n))
        grids.append(np.linspace(0, 2, n))
        grids.append(np.linspace(0, 5, n))
        
    # 2. Geometric grids
    for base in [1.5, 2.0, 3.0]:
        for n in [4, 5, 6]:
            grids.append(np.array([base**i for i in range(n)]))
            
    # 3. Specific structured grids
    # Points related to the inequality structure
    grids.append(np.array([0, 0.5, 1, 2, 4]))
    grids.append(np.array([0, 1, 2, 4, 8]))
    grids.append(np.array([0, 0.1, 0.5, 1, 2, 5]))
    
    # 4. Random grids (sorted)
    np.random.seed(42)
    for _ in range(20):
        n = np.random.randint(4, 9)
        pts = np.sort(np.random.uniform(0, 5, n))
        grids.append(pts)
        
    # Remove duplicate grids (approximate)
    unique_grids = []
    for g in grids:
        is_dup = False
        for ug in unique_grids:
            if len(g) == len(ug) and np.allclose(g, ug):
                is_dup = True
                break
        if not is_dup:
            unique_grids.append(g)
            
    grids = unique_grids
    
    for support in grids:
        if time.time() > deadline:
            break
            
        n = len(support)
        
        # Initial weights: Uniform
        w_init = np.ones(n) / n
        
        # We optimize log-weights to ensure positivity
        # Let log_w be the variable.
        # We want to maximize P(w) where w = softmax(log_w).
        # Note: softmax is invariant to adding a constant, so we can fix one log_w to 0 or just let it vary.
        # Nelder-Mead works on unconstrained variables.
        
        def objective(log_w):
            # Convert to weights
            # Subtract max for numerical stability
            max_lw = np.max(log_w)
            exp_lw = np.exp(log_w - max_lw)
            w = exp_lw / np.sum(exp_lw)
            
            # Evaluate P
            p = _evaluate_measure_fast(support, w)
            return -p # Minimize negative P
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, np.zeros(n), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < 0: # res.fun is -P, so lower is better
                # Extract best weights
                max_lw = np.max(res.x)
                exp_lw = np.exp(res.x - max_lw)
                w_opt = exp_lw / np.sum(exp_lw)
                
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_support = support.copy()
                    best_weights = w_opt.copy()
                    
        except Exception:
            continue
            
    # If no grid worked, fallback
    if best_support is None:
        best_support = np.arange(4, dtype=np.float64)
        best_weights = np.ones(4, dtype=np.float64) / 4
    else:
        # Ensure validity
        if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
            best_support = np.arange(4, dtype=np.float64)
            best_weights = np.ones(4, dtype=np.float64) / 4
            
    return best_support, best_weights
# EVOLVE-BLOCK-END
