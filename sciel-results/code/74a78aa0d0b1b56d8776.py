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
    Smooth objective using log-sum-exp for penalty terms (vectorized).
    vars_ layout: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    Returns (objective, gradient).
    """
    n = N
    x = vars_[0::3].copy()
    y = vars_[1::3].copy()
    r = vars_[2::3].copy()
    
    # Boundary violations (vectorized)
    h_left = r - x
    h_right = x + r - 1.0
    h_bottom = r - y
    h_top = y + r - 1.0
    
    # Pairwise distances (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-30)
    np.fill_diagonal(dist, 1.0)
    
    # Overlap violations: (r_i + r_j) - dist_ij
    r_sum = r[:, None] + r[None, :]
    overlap_mat = r_sum - dist
    
    # Upper triangle indices
    iu = np.triu_indices(n, 1)
    h_overlap = overlap_mat[iu]
    h_boundary = np.concatenate([h_left, h_right, h_bottom, h_top])
    
    def soft_plus(h):
        h_c = np.clip(h, -50, 50)
        return np.log1p(np.exp(beta * h_c)) / beta
    
    def soft_plus_grad(h):
        h_c = np.clip(h, -50, 50)
        e = np.exp(beta * h_c)
        return e / (1.0 + e)
    
    boundary_penalty = np.sum(soft_plus(h_boundary))
    overlap_penalty = np.sum(soft_plus(h_overlap))
    
    obj = -np.sum(r) + 10.0 * (boundary_penalty + overlap_penalty)
    
    # Gradient (vectorized)
    grad = np.zeros(3 * n)
    grad[2::3] = -1.0  # d(-sum(r))/dr_i = -1
    
    bp_grad = 10.0 * soft_plus_grad(h_boundary)
    # h_left[i] = r[i] - x[i]: d/dx_i=-1, d/dr_i=+1
    grad[0::3] += -bp_grad[0:n]
    grad[2::3] += bp_grad[0:n]
    # h_right[i] = x[i]+r[i]-1: d/dx_i=+1, d/dr_i=+1
    grad[0::3] += bp_grad[n:2*n]
    grad[2::3] += bp_grad[n:2*n]
    # h_bottom[i] = r[i]-y[i]: d/dy_i=-1, d/dr_i=+1
    grad[1::3] += -bp_grad[2*n:3*n]
    grad[2::3] += bp_grad[2*n:3*n]
    # h_top[i] = y[i]+r[i]-1: d/dy_i=+1, d/dr_i=+1
    grad[1::3] += bp_grad[3*n:4*n]
    grad[2::3] += bp_grad[3*n:4*n]
    
    # Overlap gradient (vectorized)
    op_grad = 10.0 * soft_plus_grad(h_overlap)
    # For pair (i,j): d(overlap)/dx_i = -dx[i,j]/dist[i,j], etc.
    # Build full gradient contributions
    gdx = dx / dist
    gdy = dy / dist
    # op_grad is indexed by upper triangle pairs
    # Scatter back to full matrix
    gmat = np.zeros((n, n))
    gmat[iu] = op_grad
    # For pair (i,j): contribution to grad_x[i] = -gmat[i,j]*gdx[i,j], grad_x[j] = gmat[i,j]*gdx[i,j]
    # Sum over j for each i
    grad[0::3] += -np.sum(gmat * gdx, axis=1)
    grad[1::3] += -np.sum(gmat * gdy, axis=1)
    # For radii: each pair contributes +g to both r_i and r_j
    # gmat is symmetric in contribution sense: sum over j of gmat[i,j] for each i
    grad[2::3] += np.sum(gmat, axis=1)
    
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
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        dx_mat = xs[:, None] - xs[None, :]
        dy_mat = ys[:, None] - ys[None, :]
        dist_mat = np.sqrt(dx_mat * dx_mat + dy_mat * dy_mat + 1e-30)
        np.fill_diagonal(dist_mat, 1.0)
        min_dist_mat = rs[:, None] + rs[None, :]
        
        # Upper triangle
        iu = np.triu_indices(n, 1)
        overlaps = min_dist_mat[iu] - dist_mat[iu]
        
        for idx in range(len(iu[0])):
            i = iu[0][idx]
            j = iu[1][idx]
            ov = overlaps[idx]
            if ov > 1e-12:
                d = dist_mat[i, j]
                if d < 1e-12:
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                else:
                    dx = dx_mat[i, j] / d
                    dy = dy_mat[i, j] / d
                
                pos[i, 0] += dx * ov * 0.5
                pos[i, 1] += dy * ov * 0.5
                pos[j, 0] -= dx * ov * 0.5
                pos[j, 1] -= dy * ov * 0.5
                moved = True
        
        # Clamp to unit square
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], pos[i, 2], 1 - pos[i, 2])
            pos[i, 1] = np.clip(pos[i, 1], pos[i, 2], 1 - pos[i, 2])
        
        # Try to grow radii (vectorized)
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        # Boundary limits
        boundary_max = np.minimum(np.minimum(xs, 1 - xs), np.minimum(ys, 1 - ys))
        
        # Pairwise limits: for circle i, max_r_i = min_j (dist_ij - r_j)
        dx_mat = xs[:, None] - xs[None, :]
        dy_mat = ys[:, None] - ys[None, :]
        dist_mat = np.sqrt(dx_mat * dx_mat + dy_mat * dy_mat + 1e-30)
        np.fill_diagonal(dist_mat, 1e10)  # ignore self
        pairwise_max = dist_mat - rs[None, :]  # shape (n, n): max r_i given j
        pairwise_min = np.min(pairwise_max, axis=1)
        
        max_r = np.minimum(boundary_max, pairwise_min)
        max_r = np.maximum(max_r, 0)
        
        growth = np.maximum(max_r - rs, 0) * 0.15
        pos[:, 2] += growth
        max_growth = np.max(growth)
        
        # Clamp radii to be valid
        pos[:, 2] = np.minimum(pos[:, 2], np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]), np.minimum(pos[:, 1], 1 - pos[:, 1])))
        
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
    
    betas = [20, 50, 100, 200, 500, 1000]
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        # Smooth optimize
        opt = _scipy_optimize_smooth(pos, beta=beta, maxiter=300)
        
        # Relax to ensure validity
        relaxed = _relax_and_grow(opt, max_iters=80, tol=1e-12)
        
        if _is_valid(relaxed[:, 0], relaxed[:, 1], relaxed[:, 2]):
            score = _score(relaxed[:, 0], relaxed[:, 1], relaxed[:, 2])
            old_score = _score(pos[:, 0], pos[:, 1], pos[:, 2])
            if score > old_score - 1e-12:
                pos = relaxed
    
    # Final aggressive growth
    pos = _relax_and_grow(pos, max_iters=300, tol=1e-12)
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_make_grid_initial(N))
    inits.append(_make_hex_initial(N))
    
    # Add random inits
    for seed in range(8):
        if time.time() > deadline - 15:
            break
        inits.append(_make_random_initial(N, seed))
    
    best_result = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() > deadline - 40:
            break
        
        # First: quick relax to get a valid starting point
        relaxed = _relax_and_grow(init, max_iters=150)
        
        # Check validity
        if _is_valid(relaxed[:, 0], relaxed[:, 1], relaxed[:, 2]):
            score = _score(relaxed[:, 0], relaxed[:, 1], relaxed[:, 2])
            if score > best_score:
                best_score = score
                best_result = relaxed
        
        # Then: use smooth scipy optimization
        if time.time() > deadline - 35:
            break
        
        opt_result = _scipy_optimize_smooth(relaxed, beta=50.0, maxiter=400)
        
        # Verify validity
        if _is_valid(opt_result[:, 0], opt_result[:, 1], opt_result[:, 2]):
            opt_score = _score(opt_result[:, 0], opt_result[:, 1], opt_result[:, 2])
            if opt_score > best_score:
                best_score = opt_score
                best_result = opt_result
        else:
            # If scipy result is invalid, try to repair it
            repaired = _relax_and_grow(opt_result, max_iters=150)
            if _is_valid(repaired[:, 0], repaired[:, 1], repaired[:, 2]):
                rep_score = _score(repaired[:, 0], repaired[:, 1], repaired[:, 2])
                if rep_score > best_score:
                    best_score = rep_score
                    best_result = repaired
    
    # Final high-precision polish
    if best_result is not None and time.time() < deadline - 15:
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
