import numpy as np
import time
import scipy.optimize as so

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: 
    1. Discretize the support into a fixed grid (e.g., 128 points in [0, 3]).
    2. Optimize the weights on this grid using Projected Gradient Descent.
       The objective is P[X1+X2+X3 < 2*X4].
       This avoids the discrete "jumps" of support points, allowing smoother convergence.
    3. Use a fast vectorized score computation.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s buffer for safety
    
    def compute_score_fast(support, weights):
        """
        Fast computation of P[X1 + X2 + X3 < 2*X4].
        Assumes support is sorted and weights are normalized (sum=1).
        Returns the probability P.
        """
        n = len(support)
        if n == 0:
            return 0.0
        
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pre-compute pairwise sums of support and probabilities for Z = X1 + X2
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        # Sort the pairwise sums to allow efficient CDF lookup via searchsorted
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        
        # Prefix sum of probabilities for Z
        p_prefix = np.cumsum(p_sorted)
        
        # Iterate over possible values of X3 (k) and X4 (l)
        # Threshold T_{k,l} = 2*s_l - s_k
        # We need sum_{k,l} p_k p_l P(Z < T_{k,l})
        
        s_k_grid = support[:, None]
        s_l_grid = support[None, :]
        
        thresholds = 2.0 * s_l_grid - s_k_grid
        
        # Find index for strict inequality Z < T
        # searchsorted(a, v, side='left') gives index of first element >= v
        # So elements < v are [0, idx)
        idxs = np.searchsorted(s_sorted, thresholds, side='left')
        
        # CDF lookup
        p_prefix_ext = np.empty(len(p_prefix) + 1, dtype=np.float64)
        p_prefix_ext[0] = 0.0
        p_prefix_ext[1:] = p_prefix
        
        prob_z_less = p_prefix_ext[idxs]
        
        p_k_grid = p[:, None]
        p_l_grid = p[None, :]
        
        total_prob = np.sum(p_k_grid * p_l_grid * prob_z_less)
        
        return float(total_prob)

    def project_simplex(v):
        """
        Project vector v onto the probability simplex (sum=1, v_i >= 0).
        """
        n = len(v)
        if n == 0:
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        w = np.maximum(v - theta, 0)
        return w

    def optimize_weights_on_grid(support, init_weights, max_iters=500, step_size=0.1):
        """
        Optimize weights on a fixed support grid using Projected Gradient Descent.
        Objective: Maximize P[X1+X2+X3 < 2*X4].
        Since the objective is not linear in weights (due to the strict inequality and sorting),
        we use a finite-difference gradient approximation or a heuristic update.
        However, P is a multilinear function of the *squared* weights if we view it carefully?
        No, P = sum p_k p_l P(Z < T). Z depends on p.
        Actually, P = sum_{k,l,i,j} p_k p_l p_i p_j * I(s_i+s_j < 2s_l - s_k).
        This is a 4th-order polynomial in p.
        
        Gradient ascent with projection is the way.
        To avoid finite differences (slow), we can use a "momentum" or "mirror descent" style.
        But let's stick to simple projected gradient with a few FD steps if time permits, 
        or just use a heuristic "mass shifting" approach.
        
        Alternative: Since we are on a grid, we can use scipy.optimize.minimize with 
        SLSQP or trust-constr on the weights, fixing the support.
        This is robust.
        """
        n = len(support)
        
        def objective_neg(w):
            # w is normalized in compute_score_fast
            # But for optimization, we want to keep w on simplex or handle normalization
            # Let's work with unnormalized weights and normalize inside?
            # No, SLSQP handles bounds. Let's fix sum(w)=1 via constraint or just project.
            # Let's use a function that takes w, normalizes it, and returns -P.
            w_norm = w / np.sum(w)
            return -compute_score_fast(support, w_norm)
        
        # Initial point
        w0 = init_weights.copy()
        w0 = w0 / np.sum(w0)
        
        bounds = [(0, 1)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        try:
            res = so.minimize(
                objective_neg,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
            )
            return res.x
        except Exception:
            return w0

    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Strategy:
    # 1. Define a fine grid.
    # 2. Initialize weights with a few different distributions (uniform, peaked, etc.).
    # 3. Optimize weights on the grid using SLSQP.
    # 4. Keep the best result.
    
    # Grid setup
    # Support in [0, 3] is usually sufficient. Let's use 128 points.
    n_grid = 128
    support_grid = np.linspace(0, 3.0, n_grid)
    
    # Initializations
    rng = np.random.default_rng(42)
    
    inits = []
    # 1. Uniform
    inits.append(np.ones(n_grid))
    # 2. Peak at 0
    inits.append(np.exp(-support_grid / 0.5))
    # 3. Peak at 1
    inits.append(np.exp(-np.abs(support_grid - 1.0) / 0.5))
    # 4. Peak at 2
    inits.append(np.exp(-np.abs(support_grid - 2.0) / 0.5))
    # 5. Two peaks
    inits.append(np.exp(-np.abs(support_grid - 0.5) / 0.2) + np.exp(-np.abs(support_grid - 2.5) / 0.2))
    # 6. Random
    inits.append(rng.dirichlet(np.ones(n_grid)))
    
    for i, w_init in enumerate(inits):
        if time.time() > deadline - 60:
            break
            
        w_opt = optimize_weights_on_grid(support_grid, w_init)
        
        # Clean up weights
        w_opt = np.maximum(w_opt, 0)
        w_sum = np.sum(w_opt)
        if w_sum <= 1e-9:
            continue
        w_opt = w_opt / w_sum
        
        # Prune zero weights to reduce n for final output?
        # The task allows any n. Smaller n is faster for the evaluator?
        # "The evaluation builds all n^2 pairwise sums, so very large n is slow."
        # So we SHOULD prune.
        
        # Keep only weights > 1e-6
        mask = w_opt > 1e-6
        support_pruned = support_grid[mask]
        weights_pruned = w_opt[mask]
        
        if len(support_pruned) == 0:
            continue
            
        # Normalize again after pruning
        weights_pruned = weights_pruned / np.sum(weights_pruned)
        
        score = compute_score_fast(support_pruned, weights_pruned)
        
        if score > best_score:
            best_score = score
            best_support = support_pruned.copy()
            best_weights = weights_pruned.copy()
            
        # If we have time, try to refine the support points themselves?
        # Moving support points is harder. But we can try a local search on the pruned set.
        
        if time.time() < deadline - 30:
             # Local refinement on the pruned support
             n_pruned = len(support_pruned)
             if n_pruned > 1 and n_pruned < 50: # Limit size for speed
                 # Use L-BFGS-B on support and weights
                 x0 = np.concatenate([support_pruned, weights_pruned])
                 bounds_refine = [(0.0, 5.0)] * n_pruned + [(0.0, 1.0)] * n_pruned
                 
                 def obj_refine(x):
                     s = np.sort(x[:n_pruned])
                     w = np.maximum(x[n_pruned:], 0)
                     ws = np.sum(w)
                     if ws < 1e-9: return 1.0
                     w = w/ws
                     return -compute_score_fast(s, w)
                 
                 try:
                     res_refine = so.minimize(
                         obj_refine,
                         x0,
                         method='L-BFGS-B',
                         bounds=bounds_refine,
                         options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
                     )
                     x_res = res_refine.x
                     s_res = np.sort(x_res[:n_pruned])
                     w_res = np.maximum(x_res[n_pruned:], 0)
                     w_res = w_res / np.sum(w_res)
                     
                     # Prune again
                     mask2 = w_res > 1e-6
                     s_final = s_res[mask2]
                     w_final = w_res[mask2]
                     if len(s_final) > 0:
                         w_final = w_final / np.sum(w_final)
                         score_final = compute_score_fast(s_final, w_final)
                         if score_final > best_score:
                             best_score = score_final
                             best_support = s_final.copy()
                             best_weights = w_final.copy()
                 except Exception:
                     pass

    # Fallback
    if best_support is None:
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n

    # Final cleanup
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    w_sum = np.sum(best_weights)
    if w_sum <= 1e-9:
         n = 32
         return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n
    best_weights = best_weights / w_sum

    return best_support, best_weights
