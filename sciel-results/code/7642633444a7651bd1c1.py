# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)
_XY = _XS + _YS
_X2Y = _XS + 2 * _YS
_XM1Y = _XS - _YS  # range -9..9
_IDX_XM1Y = _XM1Y + 9 # 0..18

# Global state for annealing
_current_alpha = 1.0

def _entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    s = p.sum()
    if s <= 1e-30:
        return 0.0
    q = p / s
    q = q[q > 0.0]
    if len(q) == 0:
        return 0.0
    return float(-(q * np.log2(q)).sum())

def _compute_value(P):
    """Compute the ratio D/M for a given distribution P using hard max."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0

    # D: distribution of x - y (slope -1), 19 bins
    d_bins = np.bincount(_IDX_XM1Y, weights=p, minlength=19)
    hd = _entropy(d_bins)

    # H_0: distribution of x, 10 bins
    h0_bins = np.bincount(_XS, weights=p, minlength=10)
    h0 = _entropy(h0_bins)

    # H_1: distribution of x + y, 19 bins
    h1_bins = np.bincount(_XY, weights=p, minlength=19)
    h1 = _entropy(h1_bins)

    # H_2: distribution of x + 2y, 28 bins
    h2_bins = np.bincount(_X2Y, weights=p, minlength=28)
    h2 = _entropy(h2_bins)

    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
    return hd / m

def _compute_value_smooth(P, alpha):
    """Compute the ratio D/M_smooth for a given distribution P."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0

    # D: distribution of x - y (slope -1), 19 bins
    d_bins = np.bincount(_IDX_XM1Y, weights=p, minlength=19)
    hd = _entropy(d_bins)

    # H_0: distribution of x, 10 bins
    h0_bins = np.bincount(_XS, weights=p, minlength=10)
    h0 = _entropy(h0_bins)

    # H_1: distribution of x + y, 19 bins
    h1_bins = np.bincount(_XY, weights=p, minlength=19)
    h1 = _entropy(h1_bins)

    # H_2: distribution of x + 2y, 28 bins
    h2_bins = np.bincount(_X2Y, weights=p, minlength=28)
    h2 = _entropy(h2_bins)

    if alpha == np.inf:
        m = max(h0, h1, h2)
    else:
        # Soft max: (1/alpha) * log(sum(exp(alpha * Hi)))
        h_vals = np.array([h0, h1, h2])
        max_h = np.max(h_vals)
        if max_h < 1e-15:
            m = 0.0
        else:
            # log-sum-exp trick
            exp_terms = np.exp(alpha * (h_vals - max_h))
            log_sum = np.log(np.sum(exp_terms))
            m = (max_h + log_sum / alpha)
            
    if m <= 1e-9:
        return 0.0
    return hd / m

def _build_from_theta(theta):
    """Convert log-probability parameters to a probability distribution."""
    theta_shifted = theta - np.max(theta)
    p = np.exp(theta_shifted)
    s = p.sum()
    if s <= 1e-30:
        return np.ones(_N * _N) / (_N * _N)
    p = p / s
    return p

def _objective_smooth(theta):
    """Negative of the value we want to maximize, using current alpha."""
    p = _build_from_theta(theta)
    P = p.reshape(_N, _N)
    val = _compute_value_smooth(P, _current_alpha)
    return -val

def _generate_initial_points():
    """Generate diverse initial points for the optimizer."""
    points = []
    n_cells = _N * _N
    
    # 1. Uniform
    points.append(np.zeros(n_cells))
    
    # 2. Random starts
    rng = np.random.RandomState(42)
    for _ in range(20):
        points.append(rng.randn(n_cells) * 1.5)
        
    # 3. Structured: Anti-diagonals (Maximize D = H(x-y))
    # x - y = k  =>  x = y + k
    # y ranges 0..9, x must be 0..9
    for k in range(-9, 10):
        z = np.full(n_cells, -10.0)
        for y in range(_N):
            x = y + k
            if 0 <= x < _N:
                idx = y * _N + x
                z[idx] = 0.0
        # Ensure at least one point is active (it is, since k is valid)
        points.append(z)
        
    # 4. Structured: Diagonals (Minimize H(x+y))
    # x + y = d  =>  x = d - y
    for d in range(0, 19, 2):
        z = np.full(n_cells, -10.0)
        for y in range(_N):
            x = d - y
            if 0 <= x < _N:
                idx = y * _N + x
                z[idx] = 0.0
        points.append(z)
        
    # 5. Structured: Columns (Minimize H(x))
    for x0 in range(0, _N, 2):
        z = np.full(n_cells, -10.0)
        for y in range(_N):
            idx = y * _N + x0
            z[idx] = 0.0
        points.append(z)
        
    # 6. Structured: Rows (Minimize H(y) - not directly in M but good structure)
    for y0 in range(0, _N, 2):
        z = np.full(n_cells, -10.0)
        for x in range(_N):
            idx = y0 * _N + x
            z[idx] = 0.0
        points.append(z)
            
    return points

def _greedy_sparse_search(P, max_iters=200, deadline=None):
    """
    Discrete support refinement via greedy zeroing of low-mass cells.
    This helps escape local optima by finding sparser distributions.
    """
    p = P.ravel().copy()
    n_cells = len(p)
    current_val = _compute_value(P)
    
    # Sort indices by mass (ascending)
    sorted_idx = np.argsort(p)
    
    # We will consider zeroing out cells in order of lowest mass
    # But we need to be careful not to zero out cells that are essential for entropy
    # Strategy: Try zeroing out one cell at a time, check if value improves or stays similar
    # while mass decreases (sparsity increases).
    
    # Actually, a simpler greedy approach:
    # Iterate through cells sorted by mass.
    # Try to set p[idx] = 0.
    # If the new value is >= current_val * (1 - epsilon) and the sum is still > 0, accept.
    # This might be too aggressive.
    
    # Better: Local search on the support.
    # Start with current P.
    # For each cell, try to move its mass to a neighbor or zero it.
    # Given the time constraints, let's try a simple "zero out small probabilities" approach
    # which was measured to improve score in the past.
    
    # Threshold for zeroing
    # We want to keep the top K cells? Or just zero out anything below a small threshold?
    # Let's try zeroing out cells with p < 1e-3 and see.
    
    # However, the measured improvement 2.094 -> 2.125 suggests this is a valid strategy.
    # Let's implement a more careful version.
    
    # 1. Normalize P
    p = p / p.sum()
    
    # 2. Identify "small" cells
    # Let's try to zero out cells with mass < 1e-4
    # And then re-normalize.
    
    # But simply zeroing might reduce D too much.
    # The insight is that concentrating mass on a specific structure (like a diagonal) can improve the ratio.
    
    # Let's do a local search:
    # For each cell i, consider moving all its mass to cell j (neighbor).
    # This is O(N^2) which is 100*100 = 10k operations per step.
    # With 200 iterations, that's 2M ops, which is fast.
    
    best_val = current_val
    best_p = p.copy()
    
    # Pre-compute neighbors? No, just iterate.
    # To speed up, we can just try zeroing out the smallest elements first.
    
    # Let's try a different approach:
    # Project to a structured support.
    # Identify the dominant diagonal (for D) and try to concentrate there.
    
    # Actually, the simplest effective heuristic from the observation:
    # "Discrete support refinement via greedy zeroing of low-mass cells"
    
    # Let's implement a greedy zeroing that checks the value.
    # We will iterate through the cells sorted by mass (smallest first).
    # For each cell, if p[i] < 1e-3, try setting it to 0.
    # If the new value is not significantly worse, keep it zeroed.
    
    p_work = p.copy()
    val_work = current_val
    
    # Sort indices by mass
    idxs = np.argsort(p_work)
    
    for idx in idxs:
        if deadline is not None and time.time() > deadline - 5:
            break
        if p_work[idx] > 1e-4:
            # If we've reached cells with significant mass, stop zeroing
            # We only want to prune the tails
            break
            
        # Try zeroing out this cell
        old_val = p_work[idx]
        p_work[idx] = 0.0
        
        # Check if sum is still valid
        s = p_work.sum()
        if s <= 1e-9:
            p_work[idx] = old_val
            continue
            
        # Evaluate
        P_new = p_work.reshape(_N, _N)
        v_new = _compute_value(P_new)
        
        # Accept if value is close (within 1%) or better
        if v_new >= val_work * 0.99:
            val_work = v_new
        else:
            # Revert
            p_work[idx] = old_val
            
    # Re-normalize
    s = p_work.sum()
    if s > 1e-9:
        p_work /= s
        P_final = p_work.reshape(_N, _N)
        v_final = _compute_value(P_final)
        if v_final > best_val:
            return P_final, v_final
            
    return P, current_val

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Combines diverse structured initialization with annealed soft-max optimization
    and discrete support refinement.
    """
    global _current_alpha
    deadline = time.time() + 550
    n_cells = _N * _N
    
    best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    best_val = _compute_value(best_P)
    best_theta = np.zeros(n_cells)
    
    initial_points = _generate_initial_points()
    
    # Annealing schedule: Start smooth, end sharp
    # High alpha makes the soft max closer to hard max
    alphas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    # Strategy:
    # 1. For alpha in [10, 50]: Run all initial points. Keep top K thetas.
    # 2. For alpha in [100, 200, 500, 1000]: Run only the top K thetas from previous stage.
    
    top_k = 10
    candidates = [] # list of (theta, val)
    
    for alpha in alphas:
        if time.time() > deadline - 30:
            break
            
        _current_alpha = alpha
        
        if alpha <= 50.0:
            # Broad search
            search_points = initial_points
        else:
            # Refine best candidates
            if not candidates:
                break
            # Sort candidates by value (desc)
            candidates.sort(key=lambda x: x[1], reverse=True)
            search_points = [c[0] for c in candidates[:top_k]]
            # Also add a few random perturbations of the best one
            best_theta = candidates[0][0]
            rng = np.random.RandomState(42 + int(alpha))
            for _ in range(5):
                search_points.append(best_theta + rng.randn(n_cells) * 0.5)
            
        candidates = []
        
        for theta0 in search_points:
            if time.time() > deadline - 10:
                break
                
            try:
                result = minimize(
                    _objective_smooth, theta0, method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                p_opt = _build_from_theta(result.x)
                P_opt = p_opt.reshape(_N, _N)
                
                # Evaluate with HARD max to see true score
                val_opt = _compute_value(P_opt)
                
                if val_opt > best_val:
                    best_val = val_opt
                    best_P = P_opt.copy()
                    best_theta = result.x.copy()
                
                candidates.append((result.x.copy(), val_opt))
                
            except Exception:
                continue
        
        # Keep only top K for next iteration if we are moving to higher alpha
        if alpha < 500.0:
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                candidates = candidates[:top_k]
            else:
                # Fallback: keep best_theta found so far
                candidates = [(best_theta.copy(), best_val)]
    
    # Final polish with very high alpha (effectively hard max)
    if time.time() < deadline - 10:
        _current_alpha = 2000.0
        try:
            result = minimize(
                _objective_smooth, best_theta, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            p_opt = _build_from_theta(result.x)
            P_opt = p_opt.reshape(_N, _N)
            val_opt = _compute_value(P_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_P = P_opt.copy()
        except Exception:
            pass

    # Discrete support refinement (Greedy Zeroing)
    # This is the key improvement from the observed successful child.
    if time.time() < deadline - 5:
        P_sparse, val_sparse = _greedy_sparse_search(best_P, deadline=deadline)
        if val_sparse > best_val:
            best_P = P_sparse
            best_val = val_sparse

    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    else:
        best_P /= s
    
    return best_P

# EVOLVE-BLOCK-END
