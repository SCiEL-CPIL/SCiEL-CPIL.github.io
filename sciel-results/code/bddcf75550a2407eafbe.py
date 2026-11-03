"""
Pack 200 equal circles in a unit square using gradient-based optimization 
with a soft-min objective to handle the max-min constraint.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 135  # Leave margin for 150s limit (15s safety)

def _get_radius(pts):
    """
    Calculate the maximum possible radius for given centers.
    Returns the min of (wall distances) and (pairwise distances / 2).
    """
    if len(pts) == 0:
        return 0.0
    
    # Wall distances: distance to each of the 4 walls
    # pts is (N, 2)
    # min_x = np.min(pts[:, 0]) -> dist to left wall = min_x - (-0.5) = min_x + 0.5
    # max_x = np.max(pts[:, 0]) -> dist to right wall = 0.5 - max_x
    # min_y = np.min(pts[:, 1]) -> dist to bottom wall = min_y + 0.5
    # max_y = np.max(pts[:, 1]) -> dist to top wall = 0.5 - max_y
    
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # cdist is O(N^2) which is 40000 for N=200. This is fast enough.
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min(vals, beta):
    """
    Compute the soft minimum of a set of values.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    # Handle numerical stability
    # We want to minimize the negative of the soft min, so we maximize the soft min.
    # To avoid overflow in exp, we shift by the max value.
    # Note: for soft min, we use -beta * x. 
    # Let y_i = -beta * x_i.
    # soft_min = -1/beta * log(sum(exp(y_i)))
    # max_y = max(y_i) = -beta * min(x_i)
    # sum(exp(y_i)) = exp(max_y) * sum(exp(y_i - max_y))
    # log(sum) = max_y + log(sum(exp(y_i - max_y)))
    # soft_min = -1/beta * (max_y + log(sum(exp(y_i - max_y))))
    
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta

def _objective_and_grad(x, beta):
    """
    Objective function for minimization.
    We want to MAXIMIZE the soft-min of constraints.
    So we MINIMIZE -soft_min.
    
    x: flattened array of shape (2*N,)
    beta: sharpness parameter for soft-min
    """
    pts = x.reshape((N, 2))
    
    # Calculate constraints
    # 1. Wall distances
    # dist to left: x_i + 0.5
    # dist to right: 0.5 - x_i
    # dist to bottom: y_i + 0.5
    # dist to top: 0.5 - y_i
    
    # We can compute these vectorially
    # But for gradient, it's easier to think about individual terms.
    
    # Let's compute the values for the soft min
    # There are 4*N wall terms and N*(N-1)/2 pair terms.
    
    # Wall terms
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair terms
    # We need distances between all pairs i < j
    # Using cdist is convenient for values, but for gradients, we need to be careful.
    # However, L-BFGS-B uses finite differences or analytical gradients if provided.
    # Providing analytical gradients is much faster and more stable.
    
    # Let's compute the soft min value and its gradient w.r.t x.
    
    # Combine all constraint values into a single array C
    # C has size 4N + N(N-1)/2
    # For N=200, this is 800 + 19900 = 20700 values. This is manageable.
    
    # We will compute the values and the Jacobian (gradient of C w.r.t x)
    # Then grad_obj = -1/beta * sum( w_i * grad_C_i ) where w_i are weights from log-sum-exp.
    
    # Step 1: Compute C
    # Wall parts are easy.
    # Pair parts:
    # D = cdist(pts, pts)
    # We need upper triangle indices.
    
    # Instead of building the full C array explicitly which might be memory heavy or slow to index,
    # let's compute the log-sum-exp components directly.
    
    # log_sum_exp(-beta * C) = max(-beta*C) + log(sum(exp(-beta*C - max(-beta*C))))
    
    # Let's identify the critical terms.
    # Actually, for N=200, building the full C array is fine (20k floats).
    
    # Indices for pairs
    # i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    # Pair distances
    # diff = pts[i] - pts[j]
    # dist = norm(diff)
    # val = dist / 2
    
    # Let's compute all values C
    C = np.empty(4*N + num_pairs)
    
    # Fill wall values
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]
    
    # Fill pair values
    # Vectorized distance calculation for pairs
    # This is a bit tricky to do purely vectorized without loops for the norm, 
    # but we can use cdist and extract the upper triangle.
    D = cdist(pts, pts)
    # Extract upper triangle
    pair_dists = D[i_idx, j_idx]
    C[4*N:] = pair_dists / 2.0
    
    # Step 2: Compute soft min and weights
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        # Fallback to hard min if numerical issues
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    # Weights for gradient
    # w_i = exp(y_i) / sum(exp(y_j)) = exp(shifted_i) / sum_exp
    weights = exp_shifted / sum_exp
    
    # Step 3: Compute Gradient
    # grad_obj = - (1/beta) * sum( w_i * grad_C_i )
    # Note: grad_C_i is the gradient of the i-th constraint value w.r.t x.
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    # C[0:N] = x_0 + 0.5 -> grad is [1, 0] for each point's x
    # C[N:2*N] = 0.5 - x_0 -> grad is [-1, 0]
    # C[2*N:3*N] = x_1 + 0.5 -> grad is [0, 1]
    # C[3*N:4*N] = 0.5 - x_1 -> grad is [0, -1]
    
    # We can accumulate these efficiently
    # x-coord contributions
    grad_x_left = weights[0:N] * 1.0
    grad_x_right = weights[N:2*N] * (-1.0)
    grad_y_bottom = weights[2*N:3*N] * 1.0
    grad_y_top = weights[3*N:4*N] * (-1.0)
    
    # Add to grad array
    # x is (2N,). Indices 0, 2, 4... are x-coords. 1, 3, 5... are y-coords.
    # Let's reshape to (N, 2) for easier handling, then flatten back.
    grad_pts = np.zeros((N, 2))
    grad_pts[:, 0] += grad_x_left + grad_x_right
    grad_pts[:, 1] += grad_y_bottom + grad_y_top
    
    # Pair gradients
    # C[4*N + k] = dist(i, j) / 2
    # dist = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    # grad_x_i = (x_i - x_j) / (2 * dist)
    # grad_x_j = (x_j - x_i) / (2 * dist)
    # Similarly for y.
    
    # We need to iterate over pairs or use vectorized accumulation.
    # With ~20k pairs, a loop might be slow in Python. 
    # Let's try to vectorize the accumulation using np.add.at
    
    # Get the coordinates of the pairs
    # i_idx, j_idx are the indices of points.
    # x_i = pts[i_idx, 0], etc.
    
    # Calculate differences
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    # Gradients for each pair term
    # d(dist/2)/dx_i = dx / (2 * dist)
    # d(dist/2)/dx_j = -dx / (2 * dist)
    
    grad_pair_x_i = dx_safe / (2.0 * dists_safe)
    grad_pair_x_j = -dx_safe / (2.0 * dists_safe)
    grad_pair_y_i = dy_safe / (2.0 * dists_safe)
    grad_pair_y_j = -dy_safe / (2.0 * dists_safe)
    
    # Mask out invalid pairs (dist ~ 0)
    grad_pair_x_i = np.where(valid, grad_pair_x_i, 0.0)
    grad_pair_x_j = np.where(valid, grad_pair_x_j, 0.0)
    grad_pair_y_i = np.where(valid, grad_pair_y_i, 0.0)
    grad_pair_y_j = np.where(valid, grad_pair_y_j, 0.0)
    
    # Accumulate weights
    pair_weights = weights[4*N:]
    
    # np.add.at for accumulating gradients into the correct point indices
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    # Total gradient
    # Remember: we are minimizing -soft_min.
    # d(-soft_min)/dx = - (1/beta) * sum(w_i * dC_i/dx)
    # The sum calculated above is sum(w_i * dC_i/dx).
    # So we need to negate and divide by beta.
    
    grad_pts = -grad_pts / beta
    
    return obj_val, grad_pts.flatten()

def _generate_hex_initial():
    """
    Generate a centered hexagonal lattice packing as initial guess.
    Target spacing ~0.0732 (2 * best_known_radius) for better initial density.
    """
    s = 0.0732
    points = []
    row_h = s * np.sqrt(3) / 2
    
    # Generate a grid larger than the square to ensure we have enough points
    y = -0.6
    row_idx = 0
    while y < 0.6:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.6 + shift
        while x < 0.6:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
        
    pts = np.array(points)
    
    # Filter points inside the square (with small margin for safety)
    margin = 0.001
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]
    
    if len(pts) < N:
        # Fallback: generate more points with slightly larger spacing
        s2 = 0.070
        points2 = []
        row_h2 = s2 * np.sqrt(3) / 2
        y = -0.6
        row_idx = 0
        while y < 0.6:
            shift = (s2 / 2.0) if row_idx % 2 == 1 else 0.0
            x = -0.6 + shift
            while x < 0.6:
                points2.append([x, y])
                x += s2
            y += row_h2
            row_idx += 1
        pts = np.array(points2)
        mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
               (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
        pts = pts[mask]
        if len(pts) < N:
            return np.random.uniform(-0.4, 0.4, (N, 2))
        
    # Select N points closest to the center for balanced distribution
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]
    
    return pts

def _generate_random_initial():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.4, 0.4, (N, 2))

def _force_directed_phase(pts, time_limit, iterations=None):
    """
    Force-directed repulsion to spread out overlapping circles.
    Uses vectorized inverse-square repulsion between pairs and boundary repulsion.
    """
    if iterations is None:
        iterations = max(50, int(time_limit * 30))
    
    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()
    
    step = 0.0005
    lr_decay = 0.995
    
    for it in range(iterations):
        if time.time() - t_start > time_limit:
            break
        
        # Compute pairwise differences vectorized
        # diff[i,j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        dists = np.sqrt(np.sum(diff**2, axis=2))  # (N, N)
        np.fill_diagonal(dists, np.inf)
        
        # Only consider close pairs
        threshold = 0.08
        close_mask = dists < threshold
        close_mask = close_mask & ~np.eye(N, dtype=bool)
        
        # Compute forces: for each pair i,j where dist < threshold,
        # force on i from j = (pts[i] - pts[j]) / dist^3 * strength
        # This is equivalent to (diff / dist) * (1/dist^2) = diff / dist^3
        
        # Safe division
        dists_safe = np.where(close_mask, dists, 1.0)
        force_mag = np.where(close_mask, 1.0 / (dists_safe**2 + 1e-8), 0.0)  # (N, N)
        
        # Force on point i from all j: sum over j of diff[i,j] * force_mag[i,j]
        forces = np.sum(diff * force_mag[:, :, np.newaxis], axis=1)  # (N, 2)
        
        # Boundary repulsion (vectorized)
        margin = 0.03
        # Left wall
        close_left = pts[:, 0] < -0.5 + margin
        if np.any(close_left):
            forces[close_left, 0] += (margin - (pts[close_left, 0] + 0.5)) * 5.0
        # Right wall
        close_right = pts[:, 0] > 0.5 - margin
        if np.any(close_right):
            forces[close_right, 0] -= (pts[close_right, 0] - (0.5 - margin)) * 5.0
        # Bottom wall
        close_bottom = pts[:, 1] < -0.5 + margin
        if np.any(close_bottom):
            forces[close_bottom, 1] += (margin - (pts[close_bottom, 1] + 0.5)) * 5.0
        # Top wall
        close_top = pts[:, 1] > 0.5 - margin
        if np.any(close_top):
            forces[close_top, 1] -= (pts[close_top, 1] - (0.5 - margin)) * 5.0
        
        # Update positions
        pts += step * forces
        pts = np.clip(pts, -0.5, 0.5)
        
        # Decay step size
        step *= lr_decay
        
        # Check improvement every few iterations to save time
        if it % 5 == 0:
            r = _get_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    # Final check
    r = _get_radius(pts)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
    
    return best_r, best_pts


def _run_optimization(pts, time_limit, betas=None):
    if betas is None:
        betas = [50, 100, 200, 500, 1000, 2000]
        
    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    # Phase 1: Force-directed to fix initial overlaps
    fd_time = min(3.0, time_limit * 0.12)
    if fd_time > 0.5:
        r_fd, pts_fd = _force_directed_phase(pts, time_limit=fd_time)
        if r_fd > best_r:
            best_r = r_fd
            best_x = pts_fd.flatten()
            x0 = pts_fd.flatten()
    
    # Phase 2: Gradient-based optimization with increasing beta
    for beta in betas:
        elapsed = time.time() - t_start
        if elapsed > time_limit:
            break
        remaining = time_limit - elapsed
        if remaining < 3.0:
            break
            
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
            
        try:
            maxiter = 150
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 30}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
                
        except Exception:
            pass
            
    # Phase 3: Final force-directed polish
    elapsed = time.time() - t_start
    remaining = time_limit - elapsed
    if remaining > 2.0:
        r_fd2, pts_fd2 = _force_directed_phase(best_x.reshape((N, 2)), time_limit=min(2.0, remaining * 0.4))
        if r_fd2 > best_r:
            best_r = r_fd2
            best_x = pts_fd2.flatten()
            
    return best_r, best_x.reshape((N, 2))

def _generate_hex_initial_with_spacing(s):
    points = []
    row_h = s * np.sqrt(3) / 2
    y = -0.5 + row_h/2
    row_idx = 0
    while y < 0.5:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + s/2 + shift
        while x < 0.5:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
    pts = np.array(points)
    if len(pts) >= N:
        return pts[:N]
    else:
        # Fallback to random if hex grid fails
        return np.random.uniform(-0.4, 0.4, (N, 2))

def _generate_jittered_hex(s, jitter_scale, seed=0):
    """Generate hexagonal lattice with small random jitter."""
    rng = np.random.RandomState(seed)
    points = []
    row_h = s * np.sqrt(3) / 2
    y = -0.55
    row_idx = 0
    while y < 0.55:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.55 + shift
        while x < 0.55:
            # Add jitter
            jx = rng.normal(0, jitter_scale)
            jy = rng.normal(0, jitter_scale)
            points.append([x + jx, y + jy])
            x += s
        y += row_h
        row_idx += 1
    pts = np.array(points)
    
    margin = 0.001
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]
    
    if len(pts) < N:
        return None
    
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    return pts[indices[:N]]

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Define multiple initializations
    inits = []
    
    # Init 1: Standard Hex
    pts_hex1 = _generate_hex_initial()
    if pts_hex1 is not None and len(pts_hex1) == N:
        inits.append(pts_hex1)
        
    # Init 2: Jittered Hex (small jitter)
    pts_jitter1 = _generate_jittered_hex(0.0732, 0.002, seed=1)
    if pts_jitter1 is not None:
        inits.append(pts_jitter1)
        
    # Init 3: Jittered Hex (larger jitter)
    pts_jitter2 = _generate_jittered_hex(0.072, 0.004, seed=2)
    if pts_jitter2 is not None:
        inits.append(pts_jitter2)
        
    # Init 4: Denser Hex with jitter
    pts_dense = _generate_jittered_hex(0.070, 0.003, seed=3)
    if pts_dense is not None:
        inits.append(pts_dense)

    # Run optimization on each init, allocating time
    num_inits = len(inits)
    if num_inits == 0:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        best_r = _get_radius(best_pts)
        inits = [best_pts]
        num_inits = 1
        
    # Reserve 10s for final polish
    time_per_init = (_TIME_LIMIT - 10) / num_inits
    
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - t0
        remaining_for_phase = _TIME_LIMIT - 10 - elapsed
        if remaining_for_phase <= 5:
            break
            
        t_limit = min(time_per_init, remaining_for_phase)
        
        r_opt, pts_opt = _run_optimization(init_pts, time_limit=t_limit)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt
            
    # Phase: Polish the best configuration with high-beta gradient optimization
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining_time = _TIME_LIMIT - 3 - elapsed
        if remaining_time > 4:
            betas_polish = [1000, 2000, 5000]
            r_polish, pts_polish = _run_optimization(best_pts, time_limit=remaining_time, betas=betas_polish)
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish
                
    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
