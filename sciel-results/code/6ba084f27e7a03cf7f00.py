"""
Pack 200 equal circles in a unit square using gradient-based optimization 
with a soft-min objective to handle the max-min constraint.
Includes force-directed phase, projected gradient descent, and multi-start.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 145  # Leave margin for 150s limit

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
    Use spacing slightly larger than 2*best_known_radius for optimizer freedom.
    """
    s = 0.074  # Slightly larger than 2*0.0366128 to give room
    points = []
    row_h = s * np.sqrt(3) / 2
    
    # Generate a grid larger than the square to ensure we have enough points
    y = -0.55
    row_idx = 0
    while y < 0.55:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.55 + shift
        while x < 0.55:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
        
    pts = np.array(points)
    
    # Filter points inside the square (with small margin)
    mask = (pts[:, 0] > -0.495) & (pts[:, 0] < 0.495) & (pts[:, 1] > -0.495) & (pts[:, 1] < 0.495)
    pts = pts[mask]
    
    if len(pts) < N:
        # If not enough, generate more with smaller spacing
        s = 0.072
        points = []
        row_h = s * np.sqrt(3) / 2
        y = -0.55
        row_idx = 0
        while y < 0.55:
            shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
            x = -0.55 + shift
            while x < 0.55:
                points.append([x, y])
                x += s
            y += row_h
            row_idx += 1
        pts = np.array(points)
        mask = (pts[:, 0] > -0.495) & (pts[:, 0] < 0.495) & (pts[:, 1] > -0.495) & (pts[:, 1] < 0.495)
        pts = pts[mask]
        if len(pts) < N:
            return np.random.uniform(-0.4, 0.4, (N, 2))
            
    # Select N points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]
    
    return pts

def _force_directed_phase(pts, time_limit, target_r=None):
    """
    Force-directed relaxation: repel overlapping circles, attract to fill space.
    Uses simple physics with velocity damping.
    """
    if target_r is None:
        target_r = _get_radius(pts)
    
    t_start = time.time()
    pts = pts.copy()
    velocities = np.zeros_like(pts)
    damping = 0.85
    dt = 0.002
    repulsion_strength = 2.0
    attraction_strength = 0.15
    
    # Target distance between centers
    target_dist = 2 * target_r * 1.02  # Slightly more than touching
    
    for step in range(8000):
        if time.time() - t_start > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion (only for close pairs for efficiency)
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        # Only consider pairs within 2.5x target distance
        close_mask = D < 2.5 * target_dist
        close_mask = np.triu(close_mask, k=1)
        
        i_idx, j_idx = np.where(close_mask)
        
        for i, j in zip(i_idx, j_idx):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Add random small perturbation
                diff = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(diff)
            
            # Repulsion: strong when overlapping, weak when close
            if dist < target_dist:
                force_mag = repulsion_strength * (target_dist - dist) / target_dist
            elif dist < 1.5 * target_dist:
                force_mag = 0.1 * repulsion_strength * (1.5 * target_dist - dist) / target_dist
            else:
                force_mag = 0.0
            
            if force_mag > 0:
                force_dir = diff / dist
                forces[i] += force_mag * force_dir
                forces[j] -= force_mag * force_dir
        
        # Wall forces: push away from walls
        wall_margin = 0.03
        for i in range(N):
            # Left wall
            if pts[i, 0] < -0.5 + wall_margin:
                forces[i, 0] += attraction_strength * (-0.5 + wall_margin - pts[i, 0])
            # Right wall
            if pts[i, 0] > 0.5 - wall_margin:
                forces[i, 0] -= attraction_strength * (pts[i, 0] - (0.5 - wall_margin))
            # Bottom wall
            if pts[i, 1] < -0.5 + wall_margin:
                forces[i, 1] += attraction_strength * (-0.5 + wall_margin - pts[i, 1])
            # Top wall
            if pts[i, 1] > 0.5 - wall_margin:
                forces[i, 1] -= attraction_strength * (pts[i, 1] - (0.5 - wall_margin))
        
        # Update velocities and positions
        velocities = damping * velocities + dt * forces
        pts += dt * velocities
        
        # Project back into square
        pts = np.clip(pts, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Early stopping if converged
        if step > 200 and np.max(np.abs(velocities)) < 1e-9:
            break
    
    return pts

def _projected_gradient_phase(pts, time_limit, lr=0.001, momentum=0.9):
    """
    Projected gradient descent with momentum to maximize sum of squared pairwise distances.
    This encourages uniform spatial distribution and can escape local minima.
    """
    t_start = time.time()
    pts = pts.copy()
    vel = np.zeros_like(pts)
    
    for step in range(10000):
        if time.time() - t_start > time_limit:
            break
        
        # Compute gradient of sum of squared pairwise distances
        # For each pair (i,j), contribution is 2 * ||p_i - p_j||^2
        # Gradient w.r.t p_i: 4 * (p_i - p_j) for each j != i
        # Gradient w.r.t p_j: 4 * (p_j - p_i) for each i != j
        
        # Vectorized computation
        # D[i,j] = ||p_i - p_j||^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        D2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(D2, 0)
        
        # Gradient: grad_i = 4 * sum_j (p_i - p_j) = 4 * (N * p_i - sum_j p_j)
        # = 4 * (N * p_i - row_sum)
        row_sum = np.sum(pts, axis=0)
        grad = 4 * (N * pts - row_sum)
        
        # Wall penalty: push points away from walls
        wall_margin = 0.05
        wall_grad = np.zeros_like(pts)
        
        # Left wall: if x < -0.5 + margin, push right
        left_mask = pts[:, 0] < -0.5 + wall_margin
        wall_grad[left_mask, 0] += 10 * (-0.5 + wall_margin - pts[left_mask, 0])
        
        # Right wall: if x > 0.5 - margin, push left
        right_mask = pts[:, 0] > 0.5 - wall_margin
        wall_grad[right_mask, 0] -= 10 * (pts[right_mask, 0] - (0.5 - wall_margin))
        
        # Bottom wall: if y < -0.5 + margin, push up
        bottom_mask = pts[:, 1] < -0.5 + wall_margin
        wall_grad[bottom_mask, 1] += 10 * (-0.5 + wall_margin - pts[bottom_mask, 1])
        
        # Top wall: if y > 0.5 - margin, push down
        top_mask = pts[:, 1] > 0.5 - wall_margin
        wall_grad[top_mask, 1] -= 10 * (pts[top_mask, 1] - (0.5 - wall_margin))
        
        # Total gradient (we want to maximize, so add gradient)
        total_grad = grad + wall_grad
        
        # Update velocity with momentum
        vel = momentum * vel + lr * total_grad
        
        # Update positions
        pts = pts + vel
        
        # Project back into square
        pts = np.clip(pts, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Check convergence
        if step > 100 and np.max(np.abs(vel)) < 1e-10:
            break
    
    return pts

def _generate_random_initial():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.4, 0.4, (N, 2))

def _run_optimization(pts, time_limit, betas=None):
    if betas is None:
        betas = [5, 10, 20, 50, 100, 200]
        
    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
            
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
            
        try:
            # Use more iterations for lower beta (exploration), fewer for higher (polish)
            if beta <= 50:
                maxiter = 200
            elif beta <= 100:
                maxiter = 120
            else:
                maxiter = 60
                
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 100}
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

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Multi-start: try multiple initial configurations
    initial_configs = []
    
    # Hex initial
    pts_hex = _generate_hex_initial()
    if pts_hex is not None and len(pts_hex) == N:
        initial_configs.append(pts_hex)
    
    # Random initial
    pts_rand = _generate_random_initial()
    initial_configs.append(pts_rand)
    
    # Slightly perturbed hex
    if len(initial_configs) > 0:
        pts_perturbed = initial_configs[0].copy()
        pts_perturbed += np.random.randn(N, 2) * 0.01
        pts_perturbed = np.clip(pts_perturbed, -0.45, 0.45)
        initial_configs.append(pts_perturbed)
    
    # Process each initial configuration with time budget
    num_configs = len(initial_configs)
    per_config_time = (_TIME_LIMIT - 20) / num_configs  # Leave 20s for final polish
    
    for config_idx, pts_init in enumerate(initial_configs):
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - elapsed - 20  # Reserve 20s for final phase
        if remaining < 5:
            break
        
        config_time = min(per_config_time, remaining)
        
        # Phase 1: Force-directed relaxation
        fd_time = config_time * 0.3
        pts_fd = _force_directed_phase(pts_init, time_limit=fd_time)
        r_fd = _get_radius(pts_fd)
        if r_fd > best_r:
            best_r = r_fd
            best_pts = pts_fd
        
        # Phase 2: Projected gradient descent
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - elapsed - 20
        if remaining < 5:
            continue
        
        pg_time = config_time * 0.2
        pts_pg = _projected_gradient_phase(best_pts if best_pts is not None else pts_fd, time_limit=pg_time)
        r_pg = _get_radius(pts_pg)
        if r_pg > best_r:
            best_r = r_pg
            best_pts = pts_pg
        
        # Phase 3: Gradient optimization
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - elapsed - 20
        if remaining < 5:
            continue
        
        grad_time = config_time * 0.5
        if best_pts is not None:
            r_grad, pts_grad = _run_optimization(best_pts, time_limit=grad_time)
            if r_grad > best_r:
                best_r = r_grad
                best_pts = pts_grad
    
    # Final phase: High-precision polish
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining_time = _TIME_LIMIT - elapsed - 3
        if remaining_time > 5:
            # Use progressively higher betas for final polish
            betas_polish = [300, 500, 1000, 2000, 5000]
            r_polish, pts_polish = _run_optimization(best_pts, time_limit=remaining_time, betas=betas_polish)
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish
    
    # Additional force-directed pass if time permits
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - elapsed - 5
        if remaining > 10:
            pts_fd2 = _force_directed_phase(best_pts, time_limit=remaining, target_r=best_r * 1.005)
            r_fd2 = _get_radius(pts_fd2)
            if r_fd2 > best_r:
                best_r = r_fd2
                best_pts = pts_fd2
    
    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        
    # Final safety: clip to ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
