import numpy as np
import time
from scipy.optimize import minimize

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values
U_VALS = X_FLAT - Y_GRID.flatten()       # x - y
V_VALS = X_FLAT.flatten()                # x
W_VALS = X_FLAT.flatten() + Y_FLAT.flatten()       # x + y
Z_VALS = X_FLAT.flatten() + 2 * Y_FLAT.flatten()   # x + 2y

# Unique values and inverse indices for bincount
U_UNIQ, U_INV = np.unique(U_VALS, return_inverse=True)
V_UNIQ, V_INV = np.unique(V_VALS, return_inverse=True)
W_UNIQ, W_INV = np.unique(W_VALS, return_inverse=True)
Z_UNIQ, Z_INV = np.unique(Z_VALS, return_inverse=True)

def _entropy_from_probs(probs):
    """Calculate Shannon entropy in bits, ignoring zeros."""
    mask = probs > 1e-12
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    p_sum = np.sum(p)
    if p_sum <= 1e-12:
        return 0.0
    p = p / p_sum
    return -np.sum(p * np.log2(p))

def _compute_ratio(P_flat):
    """
    Compute the ratio D / M for a given flat probability vector.
    P_flat: 1D array of length 100, non-negative, summing to 1.
    """
    P_flat = np.clip(P_flat, 0, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_flat = P_flat / total
    
    p_u = np.bincount(U_INV, weights=P_flat, minlength=len(U_UNIQ))
    p_v = np.bincount(V_INV, weights=P_flat, minlength=len(V_UNIQ))
    p_w = np.bincount(W_INV, weights=P_flat, minlength=len(W_UNIQ))
    p_z = np.bincount(Z_INV, weights=P_flat, minlength=len(Z_UNIQ))
    
    h_u = _entropy_from_probs(p_u)
    h_v = _entropy_from_probs(p_v)
    h_w = _entropy_from_probs(p_w)
    h_z = _entropy_from_probs(p_z)
    
    D = h_u
    M = max(h_v, h_w, h_z)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _generate_line_mixture(u, v, alpha):
    """
    Generate a probability distribution that is a mixture of:
    1. Uniform distribution on the line x - y = u
    2. Uniform distribution on the line x = v
    
    alpha is the weight of the first line.
    Returns a flat array of size 100.
    """
    P = np.zeros(100)
    
    # Line 1: x - y = u => x = y + u
    # y ranges such that 0 <= y < 10 and 0 <= y+u < 10
    y_min = max(0, -u)
    y_max = min(9, 9 - u)
    if y_max >= y_min:
        ys = np.arange(y_min, y_max + 1)
        xs = ys + u
        idxs = xs * 10 + ys  # Assuming row-major y, x? No, meshgrid indexing='xy' means P[y, x]
        # Wait, X_GRID, Y_GRID = np.meshgrid(..., indexing='xy')
        # X_GRID[i, j] = j, Y_GRID[i, j] = i
        # So P[i, j] corresponds to x=j, y=i.
        # Flatten order is C-order: index = i * 10 + j = y * 10 + x.
        # My idxs above: xs * 10 + ys is x * 10 + y. That is WRONG.
        # Correct index for x, y is y * 10 + x.
        idxs = ys * 10 + xs
        count = len(ys)
        P[idxs] += (1 - alpha) / count  # Wait, alpha is usually weight of first? Let's say alpha is weight of line 1.
        # Let's stick to: P = alpha * P_line1 + (1-alpha) * P_line2
        # I'll re-define alpha as weight of line 1.
        P[idxs] = np.zeros(100) # Reset to avoid confusion in logic below
        
    # Let's do it cleanly
    P = np.zeros(100)
    
    # Line 1: x - y = u
    y_min = max(0, -u)
    y_max = min(9, 9 - u)
    if y_max >= y_min:
        ys = np.arange(y_min, y_max + 1)
        xs = ys + u
        idxs = ys * 10 + xs
        count = len(ys)
        P[idxs] = alpha / count
        
    # Line 2: x = v
    # x is fixed to v, y ranges 0..9
    if 0 <= v < 10:
        ys = np.arange(10)
        xs = np.full(10, v)
        idxs = ys * 10 + xs
        count = 10
        P[idxs] += (1 - alpha) / count
        
    return P

def _optimize_alpha(u, v, start_time, deadline):
    """
    For fixed u and v, optimize alpha in [0, 1] to maximize ratio.
    """
    if time.time() > deadline - 5:
        return None, 0.0
        
    def neg_ratio(alpha):
        P = _generate_line_mixture(u, v, alpha)
        return -_compute_ratio(P)
    
    # Use bounded minimization
    try:
        res = minimize(neg_ratio, x0=0.5, method='L-BFGS-B', 
                       bounds=[(0, 1)], 
                       options={'maxiter': 100, 'ftol': 1e-12})
        alpha_opt = res.x[0]
        ratio_opt = -res.fun
        return alpha_opt, ratio_opt
    except Exception:
        return None, 0.0

def _coordinate_descent_refine(P_flat, start_time, deadline):
    """
    Refine the probability vector using coordinate descent.
    """
    P = P_flat.copy()
    current_ratio = _compute_ratio(P)
    
    # Pre-compute bincounts for speed? 
    # Coordinate descent is 100D. 
    # We can speed up _compute_ratio by noting that changing one cell only affects 
    # one bin in each of the 4 projections.
    
    # Let's implement a faster local evaluation for single cell change
    # But for simplicity and robustness, we'll use the standard _compute_ratio 
    # but limit iterations.
    
    max_iters = 20
    for _ in range(max_iters):
        if time.time() > deadline - 10:
            break
            
        improved = False
        # Shuffle indices for randomness in convergence
        indices = np.arange(100)
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10:
                break
                
            old_val = P[idx]
            # Try a few perturbations
            # We can solve for the optimal value in that cell analytically?
            # No, entropy is non-linear.
            # Use a small local search: 0, old*0.5, old*2, and a few fixed values
            candidates = [0.0, old_val * 0.5, old_val * 2.0, 0.01, 0.05, 0.1]
            # Also try to normalize after change
            
            best_candidate_ratio = current_ratio
            best_candidate_val = old_val
            
            for new_val in candidates:
                P_test = P.copy()
                P_test[idx] = new_val
                P_test = np.clip(P_test, 0, None)
                total = np.sum(P_test)
                if total <= 1e-9:
                    continue
                P_test /= total
                
                # Quick check: if P_test is identical to P, skip
                if np.allclose(P_test, P, atol=1e-12):
                    continue
                    
                ratio_test = _compute_ratio(P_test)
                if ratio_test > best_candidate_ratio + 1e-10:
                    best_candidate_ratio = ratio_test
                    best_candidate_val = new_val
            
            if best_candidate_val != old_val:
                P[idx] = best_candidate_val
                # Renormalize P globally to maintain sum 1
                P = np.clip(P, 0, None)
                total = np.sum(P)
                if total <= 1e-9:
                    P = np.ones(100)/100
                else:
                    P /= total
                current_ratio = best_candidate_ratio
                improved = True
        
        if not improved:
            break
            
    return P, current_ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # Full 600s limit
    
    # Initial best: Uniform
    best_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    # Phase 1: Structured Line Mixture Search
    # u in -9..9, v in 0..9
    # For each (u, v), find optimal alpha
    
    # We can be smarter. 
    # The number of (u, v) pairs is 19 * 10 = 190.
    # Each optimization is fast (1D).
    
    u_range = range(-9, 10)
    v_range = range(0, 10)
    
    # Sort by potential? No, just iterate.
    # To save time, we can subsample if needed, but 190 * 100ms = 19s is fine.
    
    for u in u_range:
        if time.time() > deadline - 100:
            break
        for v in v_range:
            if time.time() > deadline - 100:
                break
            
            # Skip if the lines are identical or disjoint in a bad way?
            # They are always defined.
            
            alpha_opt, ratio_opt = _optimize_alpha(u, v, start_time, deadline)
            
            if alpha_opt is not None and ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_P = _generate_line_mixture(u, v, alpha_opt)
    
    # Phase 2: Coordinate Descent Refinement
    # Start from the best line mixture found
    if time.time() < deadline - 20:
        P_refined, ratio_refined = _coordinate_descent_refine(best_P, start_time, deadline)
        if ratio_refined > best_ratio:
            best_P = P_refined
            best_ratio = ratio_refined
            
    # Phase 3: Additional Randomized Line Mixtures
    # Sometimes the optimal structure might be a mixture of two non-parallel lines 
    # or a line and a point, which the simple (u,v) cover might miss if the 
    # "line" is not axis-aligned in the (u,v) parameter space.
    # But x-y=u and x=v are a good basis.
    # Let's try mixtures of x-y=u and y=v (horizontal lines)
    
    if time.time() < deadline - 30:
        # Define generator for x-y=u and y=v
        def _generate_line_mixture_h(u, v, alpha):
            P = np.zeros(100)
            
            # Line 1: x - y = u
            y_min = max(0, -u)
            y_max = min(9, 9 - u)
            if y_max >= y_min:
                ys = np.arange(y_min, y_max + 1)
                xs = ys + u
                idxs = ys * 10 + xs
                count = len(ys)
                P[idxs] = alpha / count
                
            # Line 2: y = v
            if 0 <= v < 10:
                xs = np.arange(10)
                ys = np.full(10, v)
                idxs = ys * 10 + xs
                count = 10
                P[idxs] += (1 - alpha) / count
                
            return P

        def _optimize_alpha_h(u, v, start_time, deadline):
            if time.time() > deadline - 5:
                return None, 0.0
                
            def neg_ratio(alpha):
                P = _generate_line_mixture_h(u, v, alpha)
                return -_compute_ratio(P)
            
            try:
                res = minimize(neg_ratio, x0=0.5, method='L-BFGS-B', 
                               bounds=[(0, 1)], 
                               options={'maxiter': 100, 'ftol': 1e-12})
                alpha_opt = res.x[0]
                ratio_opt = -res.fun
                return alpha_opt, ratio_opt
            except Exception:
                return None, 0.0

        for u in u_range:
            if time.time() > deadline - 30:
                break
            for v in v_range:
                if time.time() > deadline - 30:
                    break
                
                alpha_opt, ratio_opt = _optimize_alpha_h(u, v, start_time, deadline)
                
                if alpha_opt is not None and ratio_opt > best_ratio:
                    best_ratio = ratio_opt
                    best_P = _generate_line_mixture_h(u, v, alpha_opt)
                    
                    # Refine this new best
                    if time.time() < deadline - 10:
                        P_refined, ratio_refined = _coordinate_descent_refine(best_P, start_time, deadline)
                        if ratio_refined > best_ratio:
                            best_P = P_refined
                            best_ratio = ratio_refined

    # Final safety check and normalization
    P_out = best_P.reshape((10, 10))
    P_out = np.clip(P_out, 0, None)
    total = np.sum(P_out)
    if total <= 1e-9:
        P_out = np.ones((10, 10)) / 100.0
    else:
        P_out = P_out / total
        
    return P_out
