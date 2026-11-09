# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
# Pre-compute flattened indices for binning
# P is 10x10, P[y, x]. Ravel order is C-order: index = y * 10 + x.
# We need indices for x, y, x+y, x-y, x+2y.
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
    return _compute_value_smooth(P, np.inf)

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

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Combines diverse structured initialization with annealed soft-max optimization.
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
    
    # Phase 1: Broad search with moderate alpha
    # We iterate through alphas, and for each alpha, we try a subset of initial points
    # to find good basins.
    
    # To save time, we don't run all 100+ initial points for all alphas.
    # We will run all initial points for the first few alphas, then only refine the best.
    
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
        # Try a few more iterations on the global best
        for _ in range(3):
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
                    best_theta = result.x.copy()
                else:
                    break # If no improvement, stop
            except Exception:
                break

    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    else:
        best_P /= s
    
    return best_P

# EVOLVE-BLOCK-END
