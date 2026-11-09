# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return -1e9
    return D / M

def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    # recompute M to ensure validity
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def _optimize_weights_for_support(mask, idx_a, idx_b, idx_c, idx_d, max_iter=1000):
    """
    Given a binary mask (support), optimize the weights on the support to maximize D/M.
    We parameterize weights as softmax(z) restricted to the support.
    """
    support_indices = np.where(mask.ravel() > 0)[0]
    k = len(support_indices)
    if k == 0:
        return 0.0, None
    
    # Initial weights: uniform on support
    z0 = np.zeros(k)
    
    def objective(z):
        # z are log-weights for the support
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        w = exp_z / exp_z.sum()
        
        # Construct full P vector
        pf = np.zeros(100)
        pf[support_indices] = w
        
        # Calculate entropies
        la = idx_a.max() + 1
        lb = idx_b.max() + 1
        lc = idx_c.max() + 1
        ld = idx_d.max() + 1
        
        qa = np.bincount(idx_a, weights=pf, minlength=la)
        qb = np.bincount(idx_b, weights=pf, minlength=lb)
        qc = np.bincount(idx_c, weights=pf, minlength=lc)
        qd = np.bincount(idx_d, weights=pf, minlength=ld)
        
        D = _entropy(qc)
        M = max(_entropy(qa), _entropy(qb), _entropy(qd))
        
        if M < 1e-12:
            return 1e9  # Minimize negative value, so return large positive for bad
        return -D / M

    try:
        res = minimize(objective, z0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-10})
        if res.success:
            z_opt = res.x
            z_shifted = z_opt - np.max(z_opt)
            exp_z = np.exp(z_shifted)
            w = exp_z / exp_z.sum()
            
            pf = np.zeros(100)
            pf[support_indices] = w
            P = pf.reshape(10, 10)
            val = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            return val, P
        else:
            # If optimization fails, return uniform on support
            w = np.ones(k) / k
            pf = np.zeros(100)
            pf[support_indices] = w
            P = pf.reshape(10, 10)
            val = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            return val, P
    except Exception:
        return 0.0, None

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin
    
    best_val = -1e9
    best_P = np.full((10, 10), 1.0 / 100.0)
    
    # Initial best: Uniform
    val_u = _value_from_P(best_P, idx_a, idx_b, idx_c, idx_d)
    if val_u > best_val:
        best_val = val_u
        best_P = best_P.copy()
        
    # Strategy: Simulated Annealing on Support Masks
    # We represent the state as a binary mask of which cells are active.
    # We start with a random mask and try to flip bits to improve the score.
    # To make the search effective, we periodically re-optimize the weights on the current support.
    
    # We'll use a "greedy" local search on the mask, guided by the optimized weights.
    # Since L-BFGS is expensive, we do it sparingly or on smaller problems.
    # Actually, 100 variables is small. L-BFGS is fast.
    
    # Let's try multiple random initial masks and refine them.
    rng = np.random.default_rng(123)
    
    num_restarts = 50
    max_flips_per_restart = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        # Initialize mask: Random subset of cells
        # Heuristic: Start with a structure that might be good.
        # Let's try starting with a "diagonal" or "row" or random.
        if r % 3 == 0:
            # Row
            mask = np.zeros((10, 10), dtype=bool)
            row = rng.integers(0, 10)
            mask[row, :] = True
        elif r % 3 == 1:
            # Column
            mask = np.zeros((10, 10), dtype=bool)
            col = rng.integers(0, 10)
            mask[:, col] = True
        elif r % 3 == 2:
            # Random sparse
            mask = rng.random((10, 10)) > 0.8 # ~20% active
        else:
            # Random dense
            mask = rng.random((10, 10)) > 0.5
            
        # Ensure at least 2 cells
        if mask.sum() < 2:
            mask.flat[rng.integers(0, 100)] = True
            mask.flat[rng.integers(0, 100)] = True
            
        current_mask = mask.copy()
        
        # Optimize weights on initial mask
        val, P = _optimize_weights_for_support(current_mask, idx_a, idx_b, idx_c, idx_d)
        if P is not None and val > best_val:
            best_val = val
            best_P = P.copy()
            
        # Local search on mask: Flip bits
        # We accept moves that improve the score (after re-optimizing weights? That's too slow).
        # Instead, we evaluate the score with UNIFORM weights on the mask to guide the mask search.
        # This is a proxy for the true score.
        
        def _value_uniform_mask(mask):
            pf = np.zeros(100)
            m_flat = mask.ravel()
            if m_flat.sum() == 0:
                return -1e9
            pf[m_flat] = 1.0 / m_flat.sum()
            P = pf.reshape(10, 10)
            return _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            
        current_val_uniform = _value_uniform_mask(current_mask)
        
        for flip_iter in range(max_flips_per_restart):
            if time.time() > deadline - 10.0:
                break
                
            # Pick a random cell to flip
            idx_y = rng.integers(0, 10)
            idx_x = rng.integers(0, 10)
            
            # Flip
            current_mask[idx_y, idx_x] = not current_mask[idx_y, idx_x]
            
            # Check if valid (at least 2 cells)
            if current_mask.sum() < 2:
                current_mask[idx_y, idx_x] = not current_mask[idx_y, idx_x]
                continue
                
            new_val_uniform = _value_uniform_mask(current_mask)
            
            # Accept if better, or with some probability if worse (Simulated Annealing style)
            # Since we don't have a temperature, let's just do greedy hill climbing on the uniform proxy.
            # To escape local minima, we occasionally accept worse moves.
            if new_val_uniform > current_val_uniform:
                current_val_uniform = new_val_uniform
            else:
                # Reject, revert
                current_mask[idx_y, idx_x] = not current_mask[idx_y, idx_x]
                
            # Periodically re-optimize weights to check if this mask is actually good
            if flip_iter % 10 == 0:
                val_opt, P_opt = _optimize_weights_for_support(current_mask, idx_a, idx_b, idx_c, idx_d)
                if P_opt is not None and val_opt > best_val:
                    best_val = val_opt
                    best_P = P_opt.copy()
                    
    # Final Polish: Take the best P found and try to refine it with a global optimization
    # or just return the best P found.
    # The best P found is already optimized for its support.
    
    # One last check: try optimizing the best_P's support again with more iterations?
    # Or just return best_P.
    
    P_final = _finalize(best_P, idx_a, idx_b, idx_c, idx_d)
    return P_final

# EVOLVE-BLOCK-END
