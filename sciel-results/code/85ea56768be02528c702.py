"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np


def _is_valid(x, y, r, tol=1e-9):
    """Check validity: circles inside [0,1]^2 and disjoint interiors."""
    n = len(x)
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
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - 1e-10:
                return False
    return True


def _score(x, y, r):
    """Sum of radii."""
    return float(np.sum(r))


def _make_grid_initial(n):
    """Create a grid initial placement."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                positions.append([x, y, r])
    return np.array(positions[:n])


def _make_hex_initial(n):
    """Create a hexagonal packing initial placement."""
    positions = []
    row_height = 1.0 / (n + 2)
    col_width = row_height * 2 / np.sqrt(3)
    y = row_height * 0.5
    row = 0
    while len(positions) < n:
        x_offset = 0.0
        if row % 2 == 1:
            x_offset = col_width * 0.5
        x = x_offset + col_width * 0.5
        col = 0
        while len(positions) < n and x <= 1.0:
            r = row_height * 0.5 - 1e-4
            positions.append([x, y, r])
            x += col_width
            col += 1
        y += row_height
        row += 1
    return np.array(positions[:n])


def _make_random_initial(n, seed):
    """Create a random initial placement."""
    rng = np.random.RandomState(seed)
    positions = []
    for _ in range(n):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        r = 0.01 + rng.random() * 0.05
        positions.append([x, y, r])
    return np.array(positions[:n])


def _smooth_objective(vars_, beta=50.0):
    """
    Smooth objective using log-sum-exp for penalty terms.
    vars_ layout: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    Returns (objective, gradient).
    """
    n = N
    x = vars_[0::3]
    y = vars_[1::3]
    r = vars_[2::3]
    
    # Smooth penalty for boundary violations
    # For each constraint g(x) <= 0, use smooth max(0, g) via log-sum-exp
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    # We want penalty when these are negative.
    
    # Collect all constraint violations (should be <= 0 for feasibility)
    # h_i = violation amount (positive when infeasible)
    
    # Boundary violations
    h_left = r - x       # > 0 when x < r
    h_right = x + r - 1  # > 0 when x > 1 - r
    h_bottom = r - y     # > 0 when y < r
    h_top = y + r - 1    # > 0 when y > 1 - r
    
    # Overlap violations: (r_i + r_j) - dist_ij > 0
    # Compute all pairwise distances
    dx = np.zeros((n, n))
    dy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx[i, j] = x[i] - x[j]
                dy[i, j] = y[i] - y[j]
    
    dist = np.sqrt(dx * dx + dy * dy + 1e-30)
    dist = np.where(dist < 1e-15, 1e-15, dist)
    
    overlap_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                overlap_mat[i, j] = r[i] + r[j] - dist[i, j]
    
    # Smooth max(0, h) using log-sum-exp: for a vector of h values,
    # smooth_max(h) = (1/beta) * log(sum(exp(beta * h_k)))
    # But we only want positive parts. Use: soft_plus(h) = (1/beta)*log(1 + exp(beta*h))
    # This is smooth and approximates max(0, h).
    
    def soft_plus(h):
        """Smooth approximation of max(0, h)."""
        # For numerical stability
        h_clipped = np.clip(h, -50, 50)
        return np.log1p(np.exp(beta * h_clipped)) / beta
    
    def soft_plus_grad(h):
        """Gradient of soft_plus w.r.t. h."""
        h_clipped = np.clip(h, -50, 50)
        return np.exp(beta * h_clipped) / (1 + np.exp(beta * h_clipped))
    
    # Boundary penalty
    h_boundary = np.concatenate([h_left, h_right, h_bottom, h_top])
    boundary_penalty = np.sum(soft_plus(h_boundary))
    
    # Overlap penalty (use upper triangle to avoid double counting)
    h_overlap = []
    for i in range(n):
        for j in range(i + 1, n):
            h_overlap.append(overlap_mat[i, j])
    h_overlap = np.array(h_overlap)
    overlap_penalty = np.sum(soft_plus(h_overlap))
    
    obj = -np.sum(r) + 10.0 * (boundary_penalty + overlap_penalty)
    
    # Gradient
    grad = np.zeros(3 * n)
    
    # Gradient of -sum(r)
    for i in range(n):
        grad[i * 3 + 2] = -1.0
    
    # Gradient of boundary penalty
    bp_grad = 10.0 * soft_plus_grad(h_boundary)
    for i in range(n):
        # h_left[i] = r[i] - x[i]
        grad[i * 3 + 0] += bp_grad[i] * (-1.0)
        grad[i * 3 + 2] += bp_grad[i] * (1.0)
        # h_right[i] = x[i] + r[i] - 1
        grad[i * 3 + 0] += bp_grad[n + i] * (1.0)
        grad[i * 3 + 2] += bp_grad[n + i] * (1.0)
        # h_bottom[i] = r[i] - y[i]
        grad[i * 3 + 1] += bp_grad[2 * n + i] * (-1.0)
        grad[i * 3 + 2] += bp_grad[2 * n + i] * (1.0)
        # h_top[i] = y[i] + r[i] - 1
        grad[i * 3 + 1] += bp_grad[3 * n + i] * (1.0)
        grad[i * 3 + 2] += bp_grad[3 * n + i] * (1.0)
    
    # Gradient of overlap penalty
    op_grad = 10.0 * soft_plus_grad(h_overlap)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            g = op_grad[idx]
            # overlap = r[i] + r[j] - dist
            # d(overlap)/d(x_i) = -dx[i,j]/dist[i,j]
            # d(overlap)/d(y_i) = -dy[i,j]/dist[i,j]
            # d(overlap)/d(r_i) = 1
            # d(overlap)/d(x_j) = dx[i,j]/dist[i,j]
            # d(overlap)/d(y_j) = dy[i,j]/dist[i,j]
            # d(overlap)/d(r_j) = 1
            d_dist_dx = dx[i, j] / dist[i, j]
            d_dist_dy = dy[i, j] / dist[i, j]
            
            grad[i * 3 + 0] += g * (-d_dist_dx)
            grad[i * 3 + 1] += g * (-d_dist_dy)
            grad[i * 3 + 2] += g * 1.0
            grad[j * 3 + 0] += g * (d_dist_dx)
            grad[j * 3 + 1] += g * (d_dist_dy)
            grad[j * 3 + 2] += g * 1.0
            idx += 1
    
    return obj, grad


def _scipy_optimize_smooth(positions, beta=50.0, maxiter=500):
    """
    Use scipy.optimize.minimize with L-BFGS-B and smooth objective.
    """
    n = len(positions)
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[i * 3 + 0] = positions[i][0]
        x0[i * 3 + 1] = positions[i][1]
        x0[i * 3 + 2] = positions[i][2]
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    result = minimize(
        lambda v: _smooth_objective(v, beta),
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'gtol': 1e-10}
    )
    
    opt_vars = result.x
    opt_positions = np.zeros((n, 3))
    for i in range(n):
        opt_positions[i][0] = opt_vars[i * 3 + 0]
        opt_positions[i][1] = opt_vars[i * 3 + 1]
        opt_positions[i][2] = opt_vars[i * 3 + 2]
    
    return opt_positions


def _relax_and_grow(positions, max_iters=200, tol=1e-10):
    """
    Iteratively relax overlaps and grow radii using vectorized operations.
    """
    n = len(positions)
    pos = positions.copy()
    
    for it in range(max_iters):
        moved = False
        # Vectorized overlap check and resolution
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = pos[i][2] + pos[j][2]
                
                if dist < min_dist - 1e-12:
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    overlap = min_dist - dist
                    pos[i][0] += dx * overlap * 0.5
                    pos[i][1] += dy * overlap * 0.5
                    pos[j][0] -= dx * overlap * 0.5
                    pos[j][1] -= dy * overlap * 0.5
                    moved = True
        
        # Clamp to unit square
        for i in range(n):
            pos[i][0] = np.clip(pos[i][0], pos[i][2], 1 - pos[i][2])
            pos[i][1] = np.clip(pos[i][1], pos[i][2], 1 - pos[i][2])
        
        # Try to grow radii
        max_growth = 0
        for i in range(n):
            max_r = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                available = (dist - pos[j][2])
                max_r = min(max_r, available)
            max_r = max(max_r, 0)
            if max_r > pos[i][2] + 1e-10:
                growth = (max_r - pos[i][2]) * 0.15
                pos[i][2] += growth
                max_growth = max(max_growth, growth)
        
        # Clamp radii to be valid
        for i in range(n):
            pos[i][2] = min(pos[i][2], pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        
        if max_growth < tol and not moved:
            break
    
    return pos


def _high_precision_polish(positions, deadline):
    """
    High-precision polish: alternate between smooth optimization with increasing
    beta and relaxation to squeeze out maximum radii.
    """
    pos = positions.copy()
    n = len(pos)
    
    betas = [20, 50, 100, 200, 500]
    for beta in betas:
        if time.time() > deadline - 3:
            break
        
        # Smooth optimize
        opt = _scipy_optimize_smooth(pos, beta=beta, maxiter=200)
        
        # Relax to ensure validity
        relaxed = _relax_and_grow(opt, max_iters=50, tol=1e-12)
        
        if _is_valid(relaxed[:, 0], relaxed[:, 1], relaxed[:, 2]):
            score = _score(relaxed[:, 0], relaxed[:, 1], relaxed[:, 2])
            old_score = _score(pos[:, 0], pos[:, 1], pos[:, 2])
            if score > old_score - 1e-12:
                pos = relaxed
    
    # Final aggressive growth
    pos = _relax_and_grow(pos, max_iters=200, tol=1e-12)
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_make_grid_initial(N))
    inits.append(_make_hex_initial(N))
    
    # Add random inits
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        inits.append(_make_random_initial(N, seed))
    
    best_result = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() > deadline - 25:
            break
        
        # First: quick relax to get a valid starting point
        relaxed = _relax_and_grow(init, max_iters=100)
        
        # Check validity
        if _is_valid(relaxed[:, 0], relaxed[:, 1], relaxed[:, 2]):
            score = _score(relaxed[:, 0], relaxed[:, 1], relaxed[:, 2])
            if score > best_score:
                best_score = score
                best_result = relaxed
        
        # Then: use smooth scipy optimization
        if time.time() > deadline - 30:
            break
        
        opt_result = _scipy_optimize_smooth(relaxed, beta=50.0, maxiter=300)
        
        # Verify validity
        if _is_valid(opt_result[:, 0], opt_result[:, 1], opt_result[:, 2]):
            opt_score = _score(opt_result[:, 0], opt_result[:, 1], opt_result[:, 2])
            if opt_score > best_score:
                best_score = opt_score
                best_result = opt_result
        else:
            # If scipy result is invalid, try to repair it
            repaired = _relax_and_grow(opt_result, max_iters=100)
            if _is_valid(repaired[:, 0], repaired[:, 1], repaired[:, 2]):
                rep_score = _score(repaired[:, 0], repaired[:, 1], repaired[:, 2])
                if rep_score > best_score:
                    best_score = rep_score
                    best_result = repaired
    
    # Final high-precision polish
    if best_result is not None and time.time() < deadline - 10:
        polished = _high_precision_polish(best_result, deadline)
        if _is_valid(polished[:, 0], polished[:, 1], polished[:, 2]):
            polished_score = _score(polished[:, 0], polished[:, 1], polished[:, 2])
            if polished_score > best_score:
                best_result = polished
                best_score = polished_score
    
    if best_result is None:
        # Fallback to grid
        best_result = _make_grid_initial(N)
    
    return best_result
# EVOLVE-BLOCK-END
