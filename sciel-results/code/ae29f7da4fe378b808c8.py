"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with a dense hexagonal lattice.
2. Use a robust energy function that combines:
   - Log-determinant barrier for uniformity (promotes spread)
   - Soft-min of pairwise distances and boundary distances
3. Multi-stage optimization:
   - Global L-BFGS-B with increasing beta
   - Local SLSQP with active-set constraints
   - Final micro-polish with direct radius maximization
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    if n < 2:
        return max(min_boundary, 0.0)
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    return max(min(min_pair_dist / 2.0, min_boundary), 0.0)

def _energy_logdet(params, beta=100.0, gamma=1.0):
    """
    Combined energy: soft-min of distances + log-det barrier for uniformity.
    
    The log-det term encourages the points to be well-spread by maximizing
    the volume of the Gram matrix of centered positions.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    
    # Boundary distances
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    d_bounds = np.concatenate([d_left, d_right, d_down, d_up])
    
    # Soft-min of all distances
    all_d = np.concatenate([d_pairs, d_bounds])
    d_min = np.min(all_d)
    shifted = all_d - d_min
    weights = np.exp(-beta * shifted)
    sum_w = np.sum(weights)
    E_softmin = d_min - np.log(sum_w) / beta
    
    # Log-determinant barrier for uniformity
    # Center the points
    c_centered = c - np.mean(c, axis=0, keepdims=True)
    # Gram matrix
    G = c_centered @ c_centered.T
    # Add small regularization for numerical stability
    G += 1e-8 * np.eye(N)
    # Log-determinant
    sign, logdet = np.linalg.slogdet(G)
    if sign <= 0:
        logdet = -1000.0
    E_logdet = -gamma * logdet / N
    
    # Total energy (we minimize this)
    E = E_softmin + E_logdet
    
    # Gradient
    p = weights / sum_w
    
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    # Pair part gradient
    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
    
    p_pairs = p[:len(d_pairs)]
    
    grad_x[i_idx] -= p_pairs * vec_x * inv_d
    grad_x[j_idx] += p_pairs * vec_x * inv_d
    grad_y[i_idx] -= p_pairs * vec_y * inv_d
    grad_y[j_idx] += p_pairs * vec_y * inv_d
    
    # Boundary part gradient
    p_bounds = p[len(d_pairs):]
    p_left = p_bounds[:N]
    p_right = p_bounds[N:2*N]
    p_down = p_bounds[2*N:3*N]
    p_up = p_bounds[3*N:]
    
    grad_x += -p_left + p_right
    grad_y += -p_down + p_up
    
    # Log-det gradient: d(-logdet(G))/dc = -2 * G^{-1} * c_centered / N
    # G^{-1} * c_centered is a vector
    try:
        G_inv_c = np.linalg.solve(G, c_centered)
        grad_logdet = -2.0 * G_inv_c / N
        grad_x += gamma * grad_logdet[:, 0]
        grad_y += gamma * grad_logdet[:, 1]
    except np.linalg.LinAlgError:
        pass
    
    total_grad = np.concatenate([grad_x, grad_y])
    return E, total_grad

def _initialize_hexagonal():
    """Initialize with hexagonal pattern."""
    best_c = None
    best_r = 0.0
    
    for d in np.linspace(0.052, 0.075, 50):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.4999, 0.4999)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_global(c0, beta=100.0, gamma=0.5, maxiter=500):
    """Run global L-BFGS-B optimization."""
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    try:
        res = minimize(
            lambda p: _energy_logdet(p, beta=beta, gamma=gamma),
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c0

def _slsqp_polish(c, r_target=None):
    """
    SLSQP polish with active-set constraints.
    Maximizes r subject to pairwise and boundary constraints.
    """
    if r_target is None:
        r_target = _compute_radius(c) * 0.999
    
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    # Objective: minimize -r (maximize r)
    # We use a smooth approximation: minimize the negative of the soft-min
    def objective(params):
        c_p = params.reshape(N, 2)
        x = c_p[:, 0]
        y = c_p[:, 1]
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1.0)
        
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        d_bounds = np.concatenate([d_left, d_right, d_down, d_up])
        
        all_d = np.concatenate([d_pairs, d_bounds])
        # Minimize negative soft-max (equivalent to maximizing soft-min)
        beta = 500.0
        d_max = np.max(all_d)
        shifted = d_max - all_d
        weights = np.exp(-beta * shifted)
        sum_w = np.sum(weights)
        soft_max = d_max + np.log(sum_w) / beta
        return -soft_max
    
    def objective_grad(params):
        c_p = params.reshape(N, 2)
        x = c_p[:, 0]
        y = c_p[:, 1]
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1.0)
        
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        d_bounds = np.concatenate([d_left, d_right, d_down, d_up])
        
        all_d = np.concatenate([d_pairs, d_bounds])
        beta = 500.0
        d_max = np.max(all_d)
        shifted = d_max - all_d
        weights = np.exp(-beta * shifted)
        sum_w = np.sum(weights)
        p = weights / sum_w
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        i_idx, j_idx = iu
        vec_x = x[i_idx] - x[j_idx]
        vec_y = y[i_idx] - y[j_idx]
        inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
        
        p_pairs = p[:len(d_pairs)]
        
        # For soft-max, gradient sign is flipped
        grad_x[i_idx] += p_pairs * vec_x * inv_d
        grad_x[j_idx] -= p_pairs * vec_x * inv_d
        grad_y[i_idx] += p_pairs * vec_y * inv_d
        grad_y[j_idx] -= p_pairs * vec_y * inv_d
        
        p_bounds = p[len(d_pairs):]
        p_left = p_bounds[:N]
        p_right = p_bounds[N:2*N]
        p_down = p_bounds[2*N:3*N]
        p_up = p_bounds[3*N:]
        
        grad_x += p_left - p_right
        grad_y += p_down - p_up
        
        total_grad = np.concatenate([grad_x, grad_y])
        return -total_grad  # Negate because we're minimizing -soft_max
    
    # Constraints: all pairwise distances >= 2*r_target, boundary distances >= r_target
    constraints = []
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    
    # Only add constraints for pairs that are close to the target
    # This keeps the constraint set manageable
    c_flat = c.flatten()
    x = c[:, 0]
    y = c[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    d_pairs = d[iu]
    
    # Add constraints for pairs within 1.2 * 2*r_target
    close_pairs = np.where(d_pairs < 1.2 * 2 * r_target)[0]
    for idx in close_pairs[:200]:  # Limit number of constraints
        ii = i_idx[idx]
        jj = j_idx[idx]
        def pair_constraint(params, ii=ii, jj=jj):
            c_p = params.reshape(N, 2)
            return np.linalg.norm(c_p[ii] - c_p[jj]) - 2 * r_target
        constraints.append({'type': 'ineq', 'fun': pair_constraint})
    
    # Boundary constraints
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    close_bounds = np.where(np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up)) < 1.2 * r_target)[0]
    for i in close_bounds:
        def bound_constraint(params, i=i):
            c_p = params.reshape(N, 2)
            return np.min([0.5 - c_p[i, 0], 0.5 + c_p[i, 0], 0.5 - c_p[i, 1], 0.5 + c_p[i, 1]]) - r_target
        constraints.append({'type': 'ineq', 'fun': bound_constraint})
    
    try:
        res = minimize(
            objective,
            c.flatten(),
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c

def _micro_polish(c, time_limit=5.0):
    """Final micro-adjustment for active constraints."""
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        bdists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        
        min_pair_dist = np.min(dist_matrix)
        min_boundary_dist = np.min(bdists)
        target_r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if target_r > current_r + 1e-15:
            current_r = target_r
        
        tol = 1e-10
        active_pairs = np.argwhere(dist_matrix < min_pair_dist + tol)
        active_bounds = np.where(bdists < min_boundary_dist + tol)[0]
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
        
        forces = np.zeros((N, 2))
        
        for i, j in active_pairs:
            if i == j:
                continue
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-14:
                unit_vec = vec / norm
                forces[i] += unit_vec
                forces[j] -= unit_vec
        
        for i in active_bounds:
            d_left = 0.5 - c[i, 0]
            d_right = c[i, 0] + 0.5
            d_down = 0.5 - c[i, 1]
            d_up = c[i, 1] + 0.5
            
            min_d = min(d_left, d_right, d_down, d_up)
            if d_left == min_d:
                forces[i, 0] += 1.0
            elif d_right == min_d:
                forces[i, 0] -= 1.0
            elif d_down == min_d:
                forces[i, 1] += 1.0
            else:
                forces[i, 1] -= 1.0
        
        moved = False
        for step in [1e-6, 5e-7, 2e-7, 1e-7, 5e-8, 2e-8]:
            if moved:
                break
            for i in range(N):
                if np.linalg.norm(forces[i]) > 1e-14:
                    direction = forces[i] / np.linalg.norm(forces[i])
                    new_pos = c[i] + direction * step
                    if np.all(np.abs(new_pos) < 0.499999):
                        old_pos = c[i].copy()
                        c[i] = new_pos
                        new_r = _compute_radius(c)
                        if new_r > current_r:
                            current_r = new_r
                            moved = True
                        else:
                            c[i] = old_pos
    
    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Global optimization with increasing beta
    for beta, gamma in [(40.0, 0.3), (80.0, 0.5), (150.0, 0.8), (250.0, 1.0)]:
        if time.time() > deadline - 80:
            break
        c_opt = _optimize_global(best_c, beta=beta, gamma=gamma, maxiter=400)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # SLSQP polish
    if time.time() < deadline - 40:
        c_opt = _slsqp_polish(best_c)
        r_opt = _compute_radius(c_opt)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Another round of global optimization from improved position
    if time.time() < deadline - 30:
        c_opt = _optimize_global(best_c, beta=300.0, gamma=0.5, maxiter=300)
        r_opt = _compute_radius(c_opt)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # SLSQP polish again
    if time.time() < deadline - 20:
        c_opt = _slsqp_polish(best_c)
        r_opt = _compute_radius(c_opt)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Micro polish
    if time.time() < deadline - 5:
        c_polished, r_polished = _micro_polish(best_c, time_limit=min(deadline - time.time() - 2, 10.0))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
    
    # Final safety check
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c
