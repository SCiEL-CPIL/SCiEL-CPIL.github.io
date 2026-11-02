"""
Combined program: Smooth penalty optimization with L-BFGS-B and restarts.
Strategy: 
1. Use smooth softplus penalties for wall and overlap constraints.
2. Optimize -sum(r) + penalty * (violations) using L-BFGS-B.
3. Perform multiple restarts with different initializations and cooling penalty weights.
4. Strict validation and repair to ensure final output is valid.
"""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _elapsed()

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    # Using logaddexp for numerical stability
    return np.logaddexp(0, z / eps) * eps

def _validate(circles, tol=1e-7):
    """
    Check if circles are inside [0,1]^2 and non-overlapping.
    circles: (N, 3) array of [x, y, r]
    """
    if circles is None or len(circles) != N:
        return False
    
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Check bounds
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
        
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _repair(circles):
    """
    Attempt to repair an invalid configuration by shrinking radii and clipping positions.
    Returns a valid configuration or the original if it can't be easily fixed.
    """
    c = circles.copy()
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    
    # Clip positions to [0, 1] and ensure r is small enough to fit in box locally
    # Note: This might still leave overlaps, but ensures wall constraints are met.
    for i in range(N):
        # Max radius allowed by walls given current position
        max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        if r[i] > max_r:
            r[i] = max(0.0, max_r)
            # Re-center if radius shrank significantly? No, just clip pos to be safe
            c[i, 0] = np.clip(x[i], r[i], 1.0 - r[i])
            c[i, 1] = np.clip(y[i], r[i], 1.0 - r[i])
            
    # Iteratively resolve overlaps by shrinking
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = c[i, 2] + c[j, 2] - dist
                if overlap > 1e-9:
                    # Shrink both radii by half the overlap + epsilon
                    shrink = overlap * 0.5 + 1e-6
                    # Ensure radii don't go negative
                    if c[i, 2] > shrink:
                        c[i, 2] -= shrink
                    else:
                        c[i, 2] = 0.0
                    if c[j, 2] > shrink:
                        c[j, 2] -= shrink
                    else:
                        c[j, 2] = 0.0
                    changed = True
        if not changed:
            break
            
    # Final check and clip
    for i in range(N):
        max_r = min(c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1])
        if c[i, 2] > max_r:
             c[i, 2] = max(0.0, max_r)
             c[i, 0] = np.clip(c[i, 0], c[i, 2], 1.0 - c[i, 2])
             c[i, 1] = np.clip(c[i, 1], c[i, 2], 1.0 - c[i, 2])

    return c

def _objective(params, penalty_weight, eps=1e-4):
    """
    Objective function: -sum(r) + penalty * (sum of squared softplus violations).
    params: flat array [x0, y0, r0, x1, y1, r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Wall violations
    # x < r -> violation r - x > 0
    # x > 1 - r -> violation x + r - 1 > 0
    v_left = r - x
    v_right = x + r - 1.0
    v_bottom = r - y
    v_top = y + r - 1.0
    
    penalty = np.sum(_softplus(v_left, eps)**2) + \
              np.sum(_softplus(v_right, eps)**2) + \
              np.sum(_softplus(v_bottom, eps)**2) + \
              np.sum(_softplus(v_top, eps)**2)
              
    # Negative radius penalty
    penalty += np.sum(_softplus(-r, eps)**2)
    
    # Pairwise overlaps
    # dist < r_i + r_j -> violation r_i + r_j - dist > 0
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    overlaps = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlaps, eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _gradient(params, penalty_weight, eps=1e-4):
    """
    Analytic gradient of the objective function.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Derivatives of softplus(z, eps) = 1 / (1 + exp(-z/eps))
    def sigmoid(z, eps):
        # Stable sigmoid
        return 1.0 / (1.0 + np.exp(-z / eps))
        
    g = np.zeros_like(params)
    
    # Wall constraints
    # d/dx (softplus(r-x)^2) = 2 * softplus(r-x) * sigmoid(r-x) * (-1)
    # d/dr (softplus(r-x)^2) = 2 * softplus(r-x) * sigmoid(r-x) * (+1)
    
    v_left = r - x
    v_right = x + r - 1.0
    v_bottom = r - y
    v_top = y + r - 1.0
    
    # Calculate derivatives
    # Note: softplus(z)^2 derivative is 2*softplus(z)*sigmoid(z)
    
    # Left wall: r - x
    sp_l = _softplus(v_left, eps)
    sig_l = sigmoid(v_left, eps)
    g[0::3] += penalty_weight * 2 * sp_l * sig_l * (-1) # dx
    g[2::3] += penalty_weight * 2 * sp_l * sig_l * (1)  # dr
    
    # Right wall: x + r - 1
    sp_r = _softplus(v_right, eps)
    sig_r = sigmoid(v_right, eps)
    g[0::3] += penalty_weight * 2 * sp_r * sig_r * (1)  # dx
    g[2::3] += penalty_weight * 2 * sp_r * sig_r * (1)  # dr
    
    # Bottom wall: r - y
    sp_b = _softplus(v_bottom, eps)
    sig_b = sigmoid(v_bottom, eps)
    g[1::3] += penalty_weight * 2 * sp_b * sig_b * (-1) # dy
    g[2::3] += penalty_weight * 2 * sp_b * sig_b * (1)  # dr
    
    # Top wall: y + r - 1
    sp_t = _softplus(v_top, eps)
    sig_t = sigmoid(v_top, eps)
    g[1::3] += penalty_weight * 2 * sp_t * sig_t * (1)  # dy
    g[2::3] += penalty_weight * 2 * sp_t * sig_t * (1)  # dr
    
    # Negative radius: -r
    sp_nr = _softplus(-r, eps)
    sig_nr = sigmoid(-r, eps)
    g[2::3] += penalty_weight * 2 * sp_nr * sig_nr * (-1) # dr
    
    # Pairwise overlaps
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(dist_sq + 1e-12)
    
    overlaps = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlaps, eps)
    sig_ov = sigmoid(overlaps, eps)
    
    # d/du (softplus(u)^2) where u = r_i + r_j - dist
    # du/dx_i = -dx/dist, du/dx_j = dx/dist
    # du/dr_i = 1, du/dr_j = 1
    
    coeff = penalty_weight * 2 * sp_ov * sig_ov
    
    # Accumulate gradients for x and y
    # g_x_i += coeff * (-dx/dist)
    # g_x_j += coeff * (dx/dist)
    
    # Using np.add.at for accumulation
    np.add.at(g[0::3], idx_i, -coeff * dx / dist)
    np.add.at(g[0::3], idx_j, coeff * dx / dist)
    np.add.at(g[1::3], idx_i, -coeff * dy / dist)
    np.add.at(g[1::3], idx_j, coeff * dy / dist)
    
    # Radii gradients
    np.add.at(g[2::3], idx_i, coeff * 1.0)
    np.add.at(g[2::3], idx_j, coeff * 1.0)
    
    # Objective term: -sum(r) -> gradient -1 for all r
    g[2::3] -= 1.0
    
    return g

def _generate_initial(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    # Hexagonal-like grid with jitter
    side = 6
    pts = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            if len(pts) < N:
                pts.append([x, y, 0.1])
    # If we need more, add random
    while len(pts) < N:
        pts.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.1])
    
    pts = np.array(pts[:N])
    # Add jitter
    pts[:, :2] += rng.uniform(-0.05, 0.05, (N, 2))
    pts[:, 2] = np.clip(pts[:, 2] + rng.uniform(-0.02, 0.02, N), 0.01, 0.3)
    
    # Flatten to params
    params = np.empty(3 * N)
    params[0::3] = pts[:, 0]
    params[1::3] = pts[:, 1]
    params[2::3] = pts[:, 2]
    return params

def _hex_initial(seed):
    """Hexagonal initial configuration."""
    rng = np.random.RandomState(seed)
    # Approximate hex packing for 26 circles
    # 5 rows: 5, 5, 5, 5, 6 = 26
    row_counts = [5, 5, 5, 5, 6]
    pts = []
    d = 0.2
    vert_spacing = d * np.sqrt(3) / 2
    for row_idx, count in enumerate(row_counts):
        offset = 0 if row_idx % 2 == 0 else d / 2
        total_width = (count - 1) * d
        start_x = 0.5 - total_width / 2 + offset
        for col in range(count):
            x = start_x + col * d
            y = 0.5 + (row_idx - 2) * vert_spacing
            pts.append([x, y, 0.09])
            
    pts = np.array(pts[:N])
    # Jitter
    pts[:, :2] += rng.uniform(-0.01, 0.01, (N, 2))
    
    params = np.empty(3 * N)
    params[0::3] = pts[:, 0]
    params[1::3] = pts[:, 1]
    params[2::3] = pts[:, 2]
    return params

def construct():
    best_score = -np.inf
    best_params = None
    
    # Bounds for L-BFGS-B
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    bounds = list(zip(lo, hi))
    
    # List of initializers
    initializers = []
    # Hexagonal
    initializers.append(_hex_initial(0))
    initializers.append(_hex_initial(42))
    # Random
    for i in range(5):
        initializers.append(_generate_initial(100 + i))
        
    # Annealing schedule for penalty weight
    # Start high to satisfy constraints, end low to maximize sum(r)
    penalty_schedule = [1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]
    
    for init_params in initializers:
        if _remaining() < 15.0:
            break
            
        current_params = init_params.copy()
        
        # Run annealing
        for pen in penalty_schedule:
            if _remaining() < 5.0:
                break
                
            try:
                result = minimize(
                    _objective,
                    current_params,
                    jac=_gradient,
                    method='L-BFGS-B',
                    bounds=bounds,
                    args=(pen,),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < _objective(current_params, pen):
                    current_params = result.x
            except Exception:
                break
        
        # Convert to circles
        circles = np.empty((N, 3))
        circles[:, 0] = current_params[0::3]
        circles[:, 1] = current_params[1::3]
        circles[:, 2] = current_params[2::3]
        
        # Validate and repair
        if not _validate(circles):
            circles = _repair(circles)
            
        if _validate(circles):
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_params = circles.copy()
        else:
            # If repair failed, try a simple grid fallback for this iteration?
            # Or just skip.
            pass
            
    # Final fallback if no valid solution found
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        best_params = np.array(cells[:N])
        
    return best_params

# EVOLVE-BLOCK-END
