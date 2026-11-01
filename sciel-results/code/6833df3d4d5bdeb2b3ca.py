"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary_dist = np.min(dist_to_boundary)
    
    return min(min_pair_dist, min_boundary_dist) / 2.0

def _compute_forces(centers, k=1.0):
    """
    Compute repulsive forces between spheres and attractive forces to center.
    Returns force vector for each sphere.
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    
    # Repulsive forces between pairs
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Avoid division by zero
                diff = np.random.randn(3) * 1e-10
                dist = np.linalg.norm(diff)
            # Inverse square law repulsion
            force_mag = k / (dist * dist)
            force_dir = diff / dist
            forces[i] += force_mag * force_dir
            forces[j] -= force_mag * force_dir
    
    # Attractive force to center (weak)
    for i in range(n):
        forces[i] -= 0.01 * centers[i]
    
    return forces

def _gradient_ascent(centers, max_iter=500, step_size=0.01):
    """
    Perform gradient ascent on the minimum distance objective using finite differences.
    """
    centers = centers.copy()
    current_radius = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    n = centers.shape[0]
    
    for it in range(max_iter):
        # Compute gradient via finite differences
        eps = 0.001
        grad = np.zeros_like(centers)
        
        for i in range(n):
            for d in range(3):
                # Forward difference
                centers_plus = centers.copy()
                centers_plus[i, d] += eps
                if np.any(np.abs(centers_plus) > 0.5):
                    centers_plus[i, d] = 0.499
                r_plus = _compute_radius(centers_plus)
                
                # Central difference
                centers_minus = centers.copy()
                centers_minus[i, d] -= eps
                if np.any(np.abs(centers_minus) > 0.5):
                    centers_minus[i, d] = -0.499
                r_minus = _compute_radius(centers_minus)
                
                grad[i, d] = (r_plus - r_minus) / (2 * eps)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Update
        new_centers = centers + step_size * grad
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius >= current_radius - 1e-10:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            # Reduce step size
            step_size *= 0.95
        
        if step_size < 1e-6:
            break
    
    return best_centers

def _force_directed(centers, max_iter=200, k=1.0, damping=0.5):
    """
    Force-directed relaxation: spheres repel each other and are attracted to center.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    n = centers.shape[0]
    velocity = np.zeros_like(centers)
    
    for it in range(max_iter):
        forces = _compute_forces(centers, k)
        
        # Update velocity with damping
        velocity = damping * velocity + 0.1 * forces
        
        # Update positions
        new_centers = centers + velocity
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        # Check if we improved
        new_radius = _compute_radius(new_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
        
        centers = new_centers
    
    return best_centers

def _hybrid_optimize(centers, time_budget=10.0):
    """
    Hybrid optimization: force-directed followed by gradient ascent.
    """
    start_time = time.time()
    
    # Phase 1: Force-directed relaxation
    centers_fd = _force_directed(centers, max_iter=100, k=1.0, damping=0.5)
    radius_fd = _compute_radius(centers_fd)
    
    if time.time() - start_time > time_budget * 0.5:
        return centers_fd
    
    # Phase 2: Gradient ascent from force-directed result
    remaining_time = time_budget - (time.time() - start_time)
    max_iter = int(remaining_time * 100)
    centers_grad = _gradient_ascent(centers_fd, max_iter=max_iter, step_size=0.005)
    radius_grad = _compute_radius(centers_grad)
    
    if radius_grad >= radius_fd:
        return centers_grad
    else:
        return centers_fd

def _random_init(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-0.45, 0.45, (N, 3))
    return centers

def _structured_init(seed):
    """
    Generate a structured initial configuration based on known packing patterns.
    Uses a body-centered cubic lattice with perturbations.
    """
    rng = np.random.RandomState(seed)
    
    # BCC lattice: corners + center of cube
    # Scale to fit in [-0.5, 0.5]
    # BCC unit cell has 2 atoms: (0,0,0) and (0.5,0.5,0.5)
    
    # Create a 3x3x3 grid of BCC cells
    positions = []
    
    # Corner positions
    for i in range(3):
        for j in range(3):
            for k in range(3):
                positions.append([i * 0.333 - 0.333, j * 0.333 - 0.333, k * 0.333 - 0.333])
    
    # Center positions
    for i in range(3):
        for j in range(3):
            for k in range(3):
                positions.append([i * 0.333 - 0.166, j * 0.333 - 0.166, k * 0.333 - 0.166])
    
    # We have 54 positions, take first 50
    positions = np.array(positions[:50])
    
    # Add random perturbation
    positions += rng.randn(50, 3) * 0.02
    
    # Clip to cube
    positions = np.clip(positions, -0.49, 0.49)
    
    return positions

def construct():
    np.random.seed(42)
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initialization strategies
    init_strategies = []
    
    # Strategy 1: Random initializations
    for seed in range(10):
        init_strategies.append(_random_init(seed))
    
    # Strategy 2: Structured initializations
    for seed in range(5):
        init_strategies.append(_structured_init(seed))
    
    # Optimize each initialization
    for i, init_centers in enumerate(init_strategies):
        if time.time() - start_time > time_limit:
            break
        
        # Calculate time budget for this optimization
        remaining_time = time_limit - (time.time() - start_time)
        n_remaining = len(init_strategies) - i
        time_budget = min(remaining_time / n_remaining, 15.0)
        
        if time_budget < 1.0:
            break
        
        optimized = _hybrid_optimize(init_centers, time_budget=time_budget)
        radius = _compute_radius(optimized)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized.copy()
    
    # Final refinement with more iterations on best result
    if best_centers is not None and time.time() - start_time < time_limit - 5.0:
        remaining_time = time_limit - (time.time() - start_time)
        refined = _hybrid_optimize(best_centers, time_budget=remaining_time * 0.8)
        refined_radius = _compute_radius(refined)
        if refined_radius >= best_radius:
            best_centers = refined
    
    return best_centers

# EVOLVE-BLOCK-END
