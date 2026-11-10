import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_radius(centers):
    """
    Compute the radius of the largest equal circles that can be placed 
    at the given centers in the square [-0.5, 0.5]^2.
    """
    # centers shape (N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask the diagonal and lower triangle
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[iu])
    
    # Distance to boundaries
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary_dist = np.min(np.concatenate([d_left, d_right, d_bottom, d_top]))
    
    r = 0.5 * min(min_pair_dist, min_boundary_dist)
    
    return r

def objective_and_grad(centers_flat):
    """
    Objective: -min(min_pair_dist, min_boundary_dist)
    Gradient: computed analytically for the bottleneck constraints.
    """
    centers = centers_flat.reshape(N, 2)
    
    # 1. Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12) # small epsilon for stability
    
    # 2. Boundary distances
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    # 3. Find bottlenecks
    # We need the minimum of all pair distances and all boundary distances.
    # Let's collect them into a single array to find the global min easily?
    # No, that's O(N^2) memory. N=50 is small, so it's fine, but let's be smart.
    
    # Mask diagonal
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Get min pair dist and indices
    min_pair_dist = np.min(dist[mask])
    # Find indices of min pair
    # np.argmin returns flat index. We need i, j.
    flat_idx = np.argmin(dist)
    i, j = np.unravel_index(flat_idx, (N, N))
    if i > j: i, j = j, i # ensure i < j for consistent gradient sign
    
    # Get min boundary dist and which boundary/point
    # Create a vector of all boundary distances
    b_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    min_b_dist = np.min(b_dists)
    
    # Determine which is the global bottleneck
    if min_pair_dist < min_b_dist:
        bottleneck = min_pair_dist
        is_pair = True
        # Gradient of -dist_ij wrt centers
        # d(-dist)/dc_i = -(c_i - c_j) / dist
        # d(-dist)/dc_j = -(c_j - c_i) / dist = (c_i - c_j) / dist
        
        grad = np.zeros_like(centers)
        v = centers[i] - centers[j]
        # dist is min_pair_dist
        grad[i] -= v / bottleneck
        grad[j] += v / bottleneck
    else:
        bottleneck = min_b_dist
        is_pair = False
        # Find which boundary distance is min
        # b_dists indices:
        # 0..N-1: left (x + 0.5) -> grad x = +1
        # N..2N-1: right (0.5 - x) -> grad x = -1
        # 2N..3N-1: bottom (y + 0.5) -> grad y = +1
        # 3N..4N-1: top (0.5 - y) -> grad y = -1
        
        idx = np.argmin(b_dists)
        grad = np.zeros_like(centers)
        
        if idx < N:
            # Left boundary: d = x + 0.5. We want to maximize d, so minimize -d.
            # grad of -d wrt x is -1.
            grad[idx, 0] = -1.0
        elif idx < 2*N:
            # Right boundary: d = 0.5 - x.
            # grad of -d wrt x is +1.
            p = idx - N
            grad[p, 0] = 1.0
        elif idx < 3*N:
            # Bottom boundary: d = y + 0.5.
            # grad of -d wrt y is -1.
            p = idx - 2*N
            grad[p, 1] = -1.0
        else:
            # Top boundary: d = 0.5 - y.
            # grad of -d wrt y is +1.
            p = idx - 3*N
            grad[p, 1] = 1.0
            
    val = -bottleneck
    return val, grad.flatten()

def _force_directed(centers, iterations=500, start_time=0, deadline=0):
    """
    Force-directed relaxation to spread points out.
    Uses soft repulsion for all pairs and strong repulsion for close pairs.
    """
    centers = centers.copy()
    # Initial temperature/step size
    step_size = 0.01
    
    for it in range(iterations):
        if time.time() > deadline - 5.0:
            break
            
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise forces
        # Vectorized:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        
        # Avoid division by zero
        np.fill_diagonal(dist, 1.0)
        
        # Force magnitude: k / dist^2 (Coulomb-like) or k * (r0 - dist) if dist < r0
        # Let's use a potential that pushes apart strongly when close.
        # F = k * (1/dist^2) * direction
        # To make it smoother, maybe F = k * (1/dist - 1/r_max)? No, standard repulsion is good.
        # Let's use F = k / dist^2.
        k = 0.01
        # Force on i from j: -k / dist^2 * (c_i - c_j) / dist = -k * (c_i - c_j) / dist^3
        # Sum over j.
        
        # Normalize diff by dist^3
        # dist^3 shape (N, N)
        dist3 = dist**3
        np.fill_diagonal(dist3, 1.0)
        
        force_matrix = k * diff / dist3[:, :, np.newaxis]
        forces -= np.sum(force_matrix, axis=1)
        
        # Boundary forces
        # Push away from boundaries if too close
        # Ideal boundary distance is ~0.1? No, just keep them inside.
        # If x < -0.4, push right. If x > 0.4, push left.
        margin = 0.4
        # Left
        mask_l = centers[:, 0] < -margin
        forces[mask_l, 0] += 0.5 * (margin - centers[mask_l, 0])
        # Right
        mask_r = centers[:, 0] > margin
        forces[mask_r, 0] -= 0.5 * (centers[mask_r, 0] - margin)
        # Bottom
        mask_b = centers[:, 1] < -margin
        forces[mask_b, 1] += 0.5 * (margin - centers[mask_b, 1])
        # Top
        mask_t = centers[:, 1] > margin
        forces[mask_t, 1] -= 0.5 * (centers[mask_t, 1] - margin)
        
        # Update positions
        centers += step_size * forces
        
        # Decay step size
        step_size *= 0.995
        
    return centers

def _gradient_refine(centers, start_time, deadline):
    """
    Use L-BFGS-B to maximize the bottleneck radius.
    """
    if time.time() > deadline - 2.0:
        return centers
        
    # Flatten centers
    x0 = centers.flatten()
    
    # Bounds: centers must be in [-0.5, 0.5]
    # Actually, if a center is on the boundary, radius is 0. 
    # We want to maximize radius, so we want centers strictly inside.
    # Bounds [-0.5, 0.5] are sufficient for L-BFGS-B to handle, 
    # but the objective is non-smooth at the boundary if we just use min.
    # However, our gradient function handles the bottleneck specifically.
    # If the bottleneck is a boundary, the gradient pushes it in.
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_centers = res.x.reshape(N, 2)
        
        # Verify validity
        if np.all(np.isfinite(new_centers)):
            # Check if it's better
            r_old = compute_radius(centers)
            r_new = compute_radius(new_centers)
            if r_new > r_old:
                return new_centers
    except Exception:
        pass
        
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Initial Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    best_radius = compute_radius(initial_centers)
    best_centers = initial_centers.copy()
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Start from grid.
    # 2. Apply force-directed relaxation to break symmetry and spread out.
    # 3. Apply gradient refinement to maximize bottleneck.
    # 4. Repeat with random perturbations if time permits.
    
    # Phase 1: Grid + Force + Gradient
    c = initial_centers.copy()
    c = _force_directed(c, iterations=300, start_time=start_time, deadline=deadline)
    c = _gradient_refine(c, start_time, deadline)
    
    r = compute_radius(c)
    if r > best_radius:
        best_radius = r
        best_centers = c
    
    # Phase 2: Randomized restarts with Force + Gradient
    # We have time for maybe 5-10 restarts.
    # Each restart: Random init or Perturbed Grid -> Force -> Gradient
    
    max_restarts = 10
    for i in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Random initialization inside square
        # To ensure they don't overlap too much initially, we can use a jittered grid
        # or just random points. Random points might have very small radius initially.
        # Force directed will push them apart.
        
        # Let's try perturbing the current best or the grid.
        # Perturbing the best is local. Perturbing the grid is global-ish.
        # Let's try a mix.
        
        if i % 2 == 0:
            # Perturb grid
            noise_scale = 0.05
            c = initial_centers + rng.normal(0, noise_scale, size=initial_centers.shape)
        else:
            # Random init
            c = rng.uniform(-0.4, 0.4, size=(N, 2))
            
        # Force directed relaxation
        # Fewer iterations for random starts as they need to settle more?
        # Or more? Let's stick to 200.
        c = _force_directed(c, iterations=200, start_time=start_time, deadline=deadline)
        
        # Gradient refinement
        c = _gradient_refine(c, start_time, deadline)
        
        r = compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
            
    return best_centers
