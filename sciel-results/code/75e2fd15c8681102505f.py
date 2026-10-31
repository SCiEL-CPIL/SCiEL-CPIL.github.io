"""Circle packing in unit square - 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26
_DEADLINE = 140.0  # seconds, safety margin before 150s kill
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _soft_max(x, beta):
    """Smooth maximum of x via log-sum-exp."""
    m = np.max(x)
    if m > 1e12:
        return m
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _soft_min(x, beta):
    return -_soft_max(-x, beta)

def _pairwise_penalty(x, y, r, beta=20.0):
    """
    Smooth penalty for circle-circle overlaps.
    For each pair, violation = (r_i + r_j) - dist_ij (positive = overlap)
    We penalize the soft-max of violations squared.
    """
    n = len(r)
    # Compute all pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12)
    # Required separation
    req = r[:, None] + r[None, :]
    # Violation (positive = overlap)
    viol = req - dist
    # Zero out diagonal
    np.fill_diagonal(viol, -np.inf)
    # Soft-max of violations, then square
    sm = _soft_max(viol, beta)
    # Also penalize the soft-max of squared violations for stronger effect
    viol_sq = np.where(viol > 0, viol**2, 0.0)
    sm_sq = _soft_max(viol_sq, beta)
    return sm_sq

def _boundary_penalty(x, y, r):
    """
    Penalty for circles extending outside [0,1]^2.
    For each circle: violation = max(0, r - x, r - (1-x), r - y, r - (1-y))
    """
    viol_left = r - x
    viol_right = r - (1.0 - x)
    viol_bottom = r - y
    viol_top = r - (1.0 - y)
    # For each circle, take max of the 4 boundary violations
    # Use soft-max per circle
    beta = 30.0
    # Stack all 4 boundary violations per circle: shape (n, 4)
    all_viol = np.stack([viol_left, viol_right, viol_bottom, viol_top], axis=1)
    # Soft-max across the 4 boundaries for each circle
    m = np.max(all_viol, axis=1)
    sm = m + np.log(np.sum(np.exp(beta * (all_viol - m[:, None])), axis=1)) / beta
    # Square and sum
    return np.sum(sm**2)

def _objective(x, y, r, beta=20.0, penalty_weight=100.0):
    """
    Returns (value, grad_x, grad_y, grad_r) for gradient ascent.
    value = sum(r) - penalty_weight * (pairwise_penalty + boundary_penalty)
    """
    n = len(r)
    obj = np.sum(r)
    
    # Pairwise penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12)
    req = r[:, None] + r[None, :]
    viol = req - dist
    np.fill_diagonal(viol, -np.inf)
    
    # Gradient of soft-max of viol^2
    viol_sq = np.where(viol > 0, viol**2, 0.0)
    np.fill_diagonal(viol_sq, 0.0)
    
    # d(sm_sq)/d(viol_sq) = softmax weights
    m = np.max(viol_sq, axis=1)
    exp_v = np.exp(beta * (viol_sq - m[:, None]))
    exp_v = np.where(np.isfinite(exp_v), exp_v, 0.0)
    weights = exp_v / np.sum(exp_v, axis=1, keepdims=True)
    # Average over both directions (i,j) and (j,i)
    weights = (weights + weights.T) / 2.0
    np.fill_diagonal(weights, 0.0)
    
    # d(viol_sq)/d(viol) = 2*viol where viol > 0
    dviol_sq_dviol = 2.0 * np.where(viol > 0, viol, 0.0)
    np.fill_diagonal(dviol_sq_dviol, 0.0)
    
    # d(viol)/d(x_i) = -dx_ij / dist_ij
    # d(viol)/d(y_i) = -dy_ij / dist_ij
    # d(viol)/d(r_i) = 1
    
    # Accumulate gradients
    dsm = weights * dviol_sq_dviol  # shape (n, n)
    
    # grad_x: for circle i, sum over j of dsm[i,j] * (-dx_ij / dist_ij)
    inv_dist = 1.0 / dist
    inv_dist = np.where(np.isfinite(inv_dist), inv_dist, 0.0)
    np.fill_diagonal(inv_dist, 0.0)
    
    grad_x = -np.sum(dsm * dx * inv_dist, axis=1)
    grad_y = -np.sum(dsm * dy * inv_dist, axis=1)
    grad_r = np.sum(dsm, axis=1)  # d(viol)/d(r_i) = 1 for each j
    
    # Add 1 from objective sum(r)
    grad_r = grad_r + 1.0
    
    # Boundary penalty gradients
    bp = _boundary_penalty(x, y, r)
    beta_b = 30.0
    viol_left = r - x
    viol_right = r - (1.0 - x)
    viol_bottom = r - y
    viol_top = r - (1.0 - y)
    all_viol = np.stack([viol_left, viol_right, viol_bottom, viol_top], axis=1)
    m_b = np.max(all_viol, axis=1)
    exp_b = np.exp(beta_b * (all_viol - m_b[:, None]))
    exp_b = np.where(np.isfinite(exp_b), exp_b, 0.0)
    w_b = exp_b / np.sum(exp_b, axis=1, keepdims=True)
    
    # d(boundary_penalty)/d(x_i): from viol_left and viol_right
    # d(viol_left)/dx = -1, d(viol_right)/dx = +1
    d_bp_dx = -w_b[:, 0] + w_b[:, 1]
    d_bp_dy = -w_b[:, 2] + w_b[:, 3]
    d_bp_dr = w_b[:, 0] + w_b[:, 1] + w_b[:, 2] + w_b[:, 3]
    
    # 2x because of square
    grad_x = grad_x - penalty_weight * 2.0 * _soft_max(all_viol, beta_b) * d_bp_dx
    grad_y = grad_y - penalty_weight * 2.0 * _soft_max(all_viol, beta_b) * d_bp_dy
    grad_r = grad_r - penalty_weight * 2.0 * _soft_max(all_viol, beta_b) * d_bp_dr
    
    # Pairwise contribution to objective value
    sm_sq_val = _soft_max(viol_sq, beta)
    obj -= penalty_weight * sm_sq_val
    obj -= penalty_weight * bp
    
    return obj, grad_x, penalty_weight * grad_x, penalty_weight * grad_y, penalty_weight * grad_r

def _is_valid(x, y, r, tol=1e-6):
    """Check if configuration is valid (no overlaps, inside square)."""
    n = len(r)
    for i in range(n):
        if x[i] < -tol or x[i] > 1 + tol or y[i] < -tol or y[i] > 1 + tol:
            return False
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > 1 + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _score(x, y, r):
    """Sum of radii if valid, else 0."""
    if _is_valid(x, y, r, tol=1e-7):
        return np.sum(r)
    return 0.0

def _project_inside(x, y, r):
    """Project radii and positions to be inside [0,1]^2."""
    r = np.maximum(r, 0.0)
    # For each circle, ensure it fits in [0,1]^2
    for i in range(len(r)):
        r[i] = min(r[i], min(x[i], 1 - x[i], y[i], 1 - y[i]))
        r[i] = max(r[i], 0.0)
    return x, y, r

def _optimize(x0, y0, r0, max_iters=5000, lr_init=0.005, beta_init=10.0,
              penalty_weight_init=50.0, verbose=False):
    """
    Gradient ascent to maximize sum(r) with smooth penalties.
    Returns (x, y, r, best_score).
    """
    x = x0.copy()
    y = y0.copy()
    r = r0.copy()
    n = len(r)
    
    best_score = _score(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    lr = lr_init
    beta = beta_init
    pw = penalty_weight_init
    
    for it in range(max_iters):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Compute objective and gradients
        obj, _, gx, gy, gr = _objective(x, y, r, beta=beta, penalty_weight=pw)
        
        # Adaptive learning rate: decrease if oscillating
        # Clip gradients
        gmax = max(np.max(np.abs(gx)), np.max(np.abs(gy)), np.max(np.abs(gr)), 1e-12)
        gx = gx / gmax
        gy = gy / gmax
        gr = gr / gmax
        
        # Update
        x_new = x + lr * gx
        y_new = y + lr * gy
        r_new = r + lr * gr
        
        # Keep inside [0,1]
        x_new = np.clip(x_new, 0.0, 1.0)
        y_new = np.clip(y_new, 0.0, 1.0)
        r_new = np.clip(r_new, 0.0, 0.5)
        
        # Check improvement
        new_score = _score(x_new, y_new, r_new)
        if new_score > best_score:
            best_score = new_score
            best_x, best_y, best_r = x_new.copy(), y_new.copy(), r_new.copy()
        
        # Accept if objective improves (including penalty)
        if obj > -1e10:
            x, y, r = x_new, y_new, r_new
        
        # Decay learning rate and increase penalty weight
        lr *= 0.995
        pw *= 1.005
        beta = min(beta * 1.01, 100.0)
        
        if verbose and it % 500 == 0:
            print(f"  iter {it}, obj={obj:.6f}, score={best_score:.6f}, lr={lr:.6f}")
    
    return best_x, best_y, best_r, best_score

def _final_polish(x, y, r, max_iters=2000):
    """
    High-precision polish: use very small steps to resolve near-overlaps.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = len(r)
    
    best_score = _score(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    lr = 0.001
    beta = 50.0
    pw = 500.0
    
    for it in range(max_iters):
        if _elapsed() > _DEADLINE - 5:
            break
        
        obj, _, gx, gy, gr = _objective(x, y, r, beta=beta, penalty_weight=pw)
        
        gmax = max(np.max(np.abs(gx)), np.max(np.abs(gy)), np.max(np.abs(gr)), 1e-12)
        gx = gx / gmax
        gy = gy / gmax
        gr = gr / gmax
        
        x_new = np.clip(x + lr * gx, 0.0, 1.0)
        y_new = np.clip(y + lr * gy, 0.0, 1.0)
        r_new = np.clip(r + lr * gr, 0.0, 0.5)
        
        new_score = _score(x_new, y_new, r_new)
        if new_score > best_score:
            best_score = new_score
            best_x, best_y, best_r = x_new.copy(), y_new.copy(), r_new.copy()
        
        x, y, r = x_new, y_new, r_new
        
        lr *= 0.998
        pw *= 1.01
        beta = min(beta * 1.02, 200.0)
    
    return best_x, best_y, best_r, best_score

def _initial_config(seed=0):
    """
    Generate initial configuration: grid with perturbation.
    """
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    # Create grid positions
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                cx = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
                cy = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
                cx = np.clip(cx, 0.05, 0.95)
                cy = np.clip(cy, 0.05, 0.95)
                positions.append((cx, cy))
    
    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])
    
    # Initial radius: based on local spacing
    r = np.zeros(N)
    for i in range(N):
        min_dist = 1.0
        for j in range(N):
            if i != j:
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dist = min(min_dist, d)
        # Also consider boundary
        min_dist = min(min_dist, x[i], 1 - x[i], y[i], 1 - y[i])
        r[i] = min_dist * 0.45
    
    return x, y, r

def construct():
    global _start_time
    _start_time = time.time()
    
    best_x, best_y, best_r = None, None, None
    best_score = 0.0
    
    # Try multiple initial configurations with different seeds
    num_restarts = 8
    max_iters_per_restart = 3000
    
    for restart in range(num_restarts):
        if _elapsed() > _DEADLINE - 20:
            break
        
        seed = restart * 12345 + 42
        x0, y0, r0 = _initial_config(seed=seed)
        
        # Run optimization
        x, y, r, score = _optimize(x0, y0, r0, max_iters=max_iters_per_restart,
                                    lr_init=0.004, beta_init=15.0,
                                    penalty_weight_init=30.0)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x, y, r
        
        # Final polish
        if _elapsed() < _DEADLINE - 15:
            x, y, r, score2 = _final_polish(x, y, r, max_iters=1000)
            if score2 > best_score:
                best_score = score2
                best_x, best_y, best_r = x, y, r
    
    # If we didn't get a valid solution, fall back to grid
    if best_score == 0.0 or best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_x, best_y, best_r])
    return result
