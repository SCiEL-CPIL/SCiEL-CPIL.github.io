import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate initial hexagonal lattice packing."""
    # We can improve the initial guess.
    # For N=50, a hexagonal lattice is good.
    # Let's use a slightly tighter spacing initially.
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _build_grid(centers, cell_size):
    """Build a spatial grid for neighbor lookup."""
    grid = {}
    for i, (x, y) in enumerate(centers):
        gx = int(np.floor(x / cell_size))
        gy = int(np.floor(y / cell_size))
        if (gx, gy) not in grid:
            grid[(gx, gy)] = []
        grid[(gx, gy)].append(i)
    return grid

def _get_neighbors(grid, x, y, cell_size, search_radius):
    """Get neighbors within search_radius using grid."""
    neighbors = []
    gx = int(np.floor(x / cell_size))
    gy = int(np.floor(y / cell_size))
    n_cells = int(np.ceil(search_radius / cell_size))
    for dx in range(-n_cells, n_cells + 1):
        for dy in range(-n_cells, n_cells + 1):
            key = (gx + dx, gy + dy)
            if key in grid:
                neighbors.extend(grid[key])
    return neighbors

def _soft_min_force(centers, r_target, beta):
    """
    Compute forces using a soft-min surrogate for the objective.
    We maximize r = min(d_ij/2, 1-|c_i|).
    We want to move points to increase the minimum.
    Using log-sum-exp approximation for min:
    min(x) ≈ -1/beta * log(sum(exp(-beta*x)))
    Gradient w.r.t. positions can be derived.
    Here we use a simpler heuristic: repel pairs that are close, attract to center if near boundary.
    But specifically, we use the grid to find close pairs efficiently.
    """
    forces = np.zeros_like(centers)
    cell_size = 0.2  # Heuristic cell size
    grid = _build_grid(centers, cell_size)
    
    # Search radius for interactions
    # We care about pairs that are close to the current minimum or slightly larger
    # Let's use a fixed interaction radius based on expected spacing
    search_radius = 0.5
    
    for i in range(N):
        xi, yi = centers[i]
        neighbors = _get_neighbors(grid, xi, yi, cell_size, search_radius)
        
        # Exclude self
        neighbors = [j for j in neighbors if j != i]
        
        for j in neighbors:
            xj, yj = centers[j]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < 1e-9:
                continue
            
            # Interaction strength: strong if close, decays with distance
            # Use a smooth kernel. If dist < 2*r_target, they are colliding or close.
            # We want to push them apart if dist < 2*r_target.
            # Even if dist > 2*r_target, if it's small, it contributes to the "min" being small.
            # Let's use a force proportional to 1/dist^2 or similar, but only for close pairs.
            # To match the "soft min" idea, we can weight by exp(-dist^2 / (2*r_target^2)) or similar.
            # But simpler: just repel if dist < threshold.
            
            # Heuristic: force = k * (1/dist) * (direction)
            # We want to increase the minimum distance.
            # If dist is small, force should be large.
            
            # Let's use a force that is active if dist < 2*r_target * 1.5
            threshold = 2 * r_target * 1.5
            if dist < threshold:
                # Repulsive force
                f_mag = 1.0 / (dist + 1e-6)
                forces[i, 0] += dx / dist * f_mag
                forces[i, 1] += dy / dist * f_mag
        
        # Boundary force
        dist_center = np.sqrt(xi**2 + yi**2)
        if dist_center > 1e-9:
            # Push towards center if close to boundary
            # Force proportional to how close to boundary
            # 1 - dist_center is the margin. If margin < r_target, push in.
            margin = 1.0 - dist_center
            if margin < r_target:
                f_mag = 1.0 / (margin + 1e-6)
                forces[i, 0] -= xi / dist_center * f_mag
                forces[i, 1] -= yi / dist_center * f_mag
    
    return forces

def _gradient_step(centers, r, step_size):
    """
    Perform a directed gradient ascent step to increase r using grid-based neighbor lookup.
    """
    new_centers = centers.copy()
    
    # Compute forces using grid-based method
    forces = _soft_min_force(centers, r, beta=10.0)
    
    # Update centers
    new_centers += step_size * forces
    
    # Ensure they don't fly out too far?
    new_dists = np.hypot(new_centers[:, 0], new_centers[:, 1])
    if np.any(new_dists > 1.0):
        # Project back to unit circle or reject
        # Rejection is safer
        return centers, 0.0
        
    return new_centers, 1.0 # 1.0 indicates success

def _optimize(centers, deadline):
    """
    Directed repulsion-based local search.
    """
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 500
    min_step_size = 1e-6
    
    while time.time() < deadline:
        # Perform a batch of gradient steps
        for _ in range(10):
            new_centers, success = _gradient_step(current, current_r, step_size)
            if not success:
                break
            
            new_r = _compute_radius(new_centers)
            
            if new_r > current_r + 1e-9:
                current = new_centers
                current_r = new_r
                no_improve_count = 0
                if current_r > best_r:
                    best = current.copy()
                    best_r = current_r
            else:
                no_improve_count += 1
                if no_improve_count > 10:
                    break
                
                if new_r > current_r - 1e-6:
                    current = new_centers
                    current_r = new_r
                else:
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
        # Check for stall
        if no_improve_count > max_no_improve:
            # Reheat: random perturbation
            perturbation = np.random.uniform(-step_size, step_size, size=(N, 2))
            perturbed = current + perturbation
            p_dists = np.hypot(perturbed[:, 0], perturbed[:, 1])
            if np.any(p_dists > 1.0):
                scale = 1.0 / np.max(p_dists)
                perturbed *= scale
            
            p_r = _compute_radius(perturbed)
            if p_r > 0:
                current = perturbed
                current_r = p_r
            no_improve_count = 0
            step_size = max(min_step_size, step_size * 0.5)

    return best, best_r

def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    
    # Run optimization
    best_centers, best_r = _optimize(centers, deadline)
    
    return best_centers
