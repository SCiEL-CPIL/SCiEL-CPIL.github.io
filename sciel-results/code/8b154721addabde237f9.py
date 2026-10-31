"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _pairwise_distances(pos):
    """Compute pairwise distances between all circle centers."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    return dist

def _boundary_violations(pos, radii, eps=1e-10):
    """Compute boundary violations: circles must be inside [0,1]^2."""
    # For each circle, the distance to each boundary
    # Circle i is inside if x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    # Violation is max(0, r_i - x_i, r_i - y_i, r_i - (1-x_i), r_i - (1-y_i))
    n = len(pos)
    x = pos[:, 0]
    y = pos[:, 1]
    r = radii
    v_left = np.maximum(0, r - x)
    v_right = np.maximum(0, r - (1 - x))
    v_bottom = np.maximum(0, r - y)
    v_top = np.maximum(0, r - (1 - y))
    return v_left + v_right + v_bottom + v_top

def _overlap_violations(pos, radii, eps=1e-10):
    """Compute overlap violations: sum of (r_i + r_j - d_ij) for overlapping pairs."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    # Overlap amount for pair (i,j) is max(0, r_i + r_j - d_ij)
    overlap = np.maximum(0, min_dist - dist)
    # Set diagonal to 0
    np.fill_diagonal(overlap, 0)
    # Return sum of overlaps (or use soft-max for smoothness)
    return np.sum(overlap)

def _objective(x, soft_beta=50.0):
    """
    Objective: minimize negative sum of radii + penalty for violations.
    x = [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_N, y_N, r_N]
    """
    n = N
    pos = x[0:3*n:3].copy()  # x-coords
    pos_y = x[1:3*n:3].copy()  # y-coords
    radii = x[2:3*n:3].copy()
    
    pos_full = np.column_stack([pos, pos_y])
    
    # Boundary violations
    bv = _boundary_violations(pos_full, radii)
    # Overlap violations
    ov = _overlap_violations(pos_full, radii)
    
    # Total violation
    total_violation = np.sum(bv) + ov
    
    # Soft penalty
    if total_violation < 1e-10:
        penalty = 0.0
    else:
        # Soft-max style penalty: use exponential for smoothness
        penalty = soft_beta * np.log(1 + np.exp(soft_beta * total_violation)) / soft_beta
        # Simpler: just use quadratic penalty
        penalty = soft_beta * total_violation ** 2
    
    # We want to maximize sum of radii, so minimize -sum(radii)
    neg_sum_r = -np.sum(radii)
    
    return neg_sum_r + penalty

def _objective_smooth(x, soft_beta=100.0):
    """
    Smooth objective using log-sum-exp for soft-min/soft-max.
    """
    n = N
    pos_x = x[0:3*n:3]
    pos_y = x[1:3*n:3]
    radii = x[2:3*n:3]
    
    pos_full = np.column_stack([pos_x, pos_y])
    
    # Boundary violations (smooth)
    bv = _boundary_violations(pos_full, radii)
    # Overlap violations (smooth)
    ov = _overlap_violations(pos_full, radii)
    
    # Use soft penalty: sum of soft-max of violations
    # For each violation term, use soft-max: beta * log(sum(exp(v_i/beta)))
    all_violations = np.concatenate([bv, [ov]])
    # Soft-max
    if len(all_violations) > 0:
        max_v = np.max(all_violations)
        if max_v > 1e-10:
            soft_penalty = soft_beta * np.log(np.sum(np.exp(all_violations / soft_beta)))
        else:
            soft_penalty = 0.0
    else:
        soft_penalty = 0.0
    
    neg_sum_r = -np.sum(radii)
    
    return neg_sum_r + soft_penalty

def _is_valid(pos, radii, tol=1e-7):
    """Check if configuration is valid."""
    n = len(pos)
    # Check boundaries
    if np.any(pos[:, 0] < -tol) or np.any(pos[:, 0] > 1 + tol):
        return False
    if np.any(pos[:, 1] < -tol) or np.any(pos[:, 1] > 1 + tol):
        return False
    if np.any(radii < -tol):
        return False
    
    # Check overlaps
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    if np.any(min_dist - dist > tol):
        return False
    
    return True

def _fix_positions_and_grow_radii(pos, radii, max_iter=500, lr=0.001):
    """
    Fix positions and try to grow radii by squeezing into gaps.
    """
    n = N
    pos = pos.copy()
    radii = radii.copy()
    
    for it in range(max_iter):
        # For each circle, compute the max radius it can have given current positions
        # and other circles' radii
        improved = False
        for i in range(n):
            # Max radius limited by boundaries
            r_max_boundary = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            
            # Max radius limited by other circles
            r_max_overlap = r_max_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx**2 + dy**2)
                # r_i <= d - r_j
                r_i_max = d - radii[j]
                if r_i_max < r_max_overlap:
                    r_max_overlap = r_i_max
            
            # Try to grow radius
            target_r = max(0, r_max_overlap - 1e-10)
            if target_r > radii[i] + 1e-10:
                radii[i] = target_r
                improved = True
        
        if not improved:
            break
    
    return pos, radii

def _polish_slsqp(x0, max_iter=100):
    """
    Final polish using SLSQP with tight tolerances.
    """
    n = N
    
    # Convert to pos, radii
    pos_x = x0[0:3*n:3].copy()
    pos_y = x0[1:3*n:3].copy()
    radii = x0[2:3*n:3].copy()
    
    # Define constraints for SLSQP
    constraints = []
    
    # Boundary constraints: x_i >= r_i, x_i <= 1 - r_i, etc.
    for i in range(n):
        # x_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        # x_i <= 1 - r_i
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
        # y_i >= r_i
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        # y_i <= 1 - r_i
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
    
    # Overlap constraints: d_ij >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2) - x[3*i+2] - x[3*j+2]
            })
    
    # Objective: maximize sum of radii
    def obj(x):
        return -x[2:3*n:3].sum()
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    result = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': 1e-12})
    
    return result.x

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = N
    best_x = None
    best_value = -np.inf
    
    # Initial guess: grid
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([cx, cy, r])
    
    # Take first n circles
    init_cells = cells[:n]
    x0 = np.array([])
    for c in init_cells:
        x0 = np.concatenate([x0, [c[0], c[1], c[2]]])
    
    # Phase 1: L-BFGS-B optimization with soft penalty
    def obj_lbgfgs(x):
        return _objective(x, soft_beta=50.0)
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    result1 = minimize(obj_lbgfgs, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12})
    
    if time.time() > deadline:
        return np.array([[result1.x[3*i], result1.x[3*i+1], result1.x[3*i+2]] for i in range(n)])
    
    x_best = result1.x
    val_best = -result1.x[2:3*n:3].sum()
    
    # Phase 2: Grow radii with fixed positions
    pos = np.column_stack([x_best[0:3*n:3], x_best[1:3*n:3]])
    radii = x_best[2:3*n:3].copy()
    pos, radii = _fix_positions_and_grow_radii(pos, radii)
    
    # Build new x
    x2 = np.array([])
    for i in range(n):
        x2 = np.concatenate([x2, [pos[i, 0], pos[i, 1], radii[i]]])
    
    val2 = -radii.sum()
    if val2 > val_best:
        x_best = x2
        val_best = val2
    
    if time.time() > deadline:
        return np.array([[x_best[3*i], x_best[3*i+1], x_best[3*i+2]] for i in range(n)])
    
    # Phase 3: SLSQP polish
    try:
        x3 = _polish_slsqp(x_best, max_iter=50)
        val3 = -x3[2:3*n:3].sum()
        if val3 > val_best:
            x_best = x3
            val_best = val3
    except Exception:
        pass
    
    # Build final result
    result = np.array([[x_best[3*i], x_best[3*i+1], x_best[3*i+2]] for i in range(n)])
    
    return result

# EVOLVE-BLOCK-END
