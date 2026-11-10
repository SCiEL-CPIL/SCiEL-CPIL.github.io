"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_grid(points, cell_size):
    """Build a spatial hash grid for fast neighbor lookups."""
    grid = {}
    for i, (x, y) in enumerate(points):
        cx = int(np.floor(x / cell_size))
        cy = int(np.floor(y / cell_size))
        key = (cx, cy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid


def _get_neighbors(grid, point, cell_size):
    """Get indices of points in neighboring grid cells."""
    cx = int(np.floor(point[0] / cell_size))
    cy = int(np.floor(point[1] / cell_size))
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            key = (cx + dx, cy + dy)
            if key in grid:
                neighbors.extend(grid[key])
    return neighbors


def _compute_forces(points, r_target, boundary_strength=10.0):
    """Compute repulsive forces between circles and boundary forces."""
    n = len(points)
    forces = np.zeros_like(points)
    cell_size = max(r_target * 2.5, 0.05)
    grid = _build_grid(points, cell_size)
    
    for i in range(n):
        neighbors = _get_neighbors(grid, points[i], cell_size)
        for j in neighbors:
            if i == j:
                continue
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            min_dist = 2.0 * r_target
            if dist < min_dist and dist > 1e-10:
                # Repulsive force proportional to overlap
                overlap = min_dist - dist
                force_mag = overlap * 5.0
                forces[i] += (diff / dist) * force_mag
        
        # Boundary force
        dist_to_center = np.linalg.norm(points[i])
        max_allowed = 1.0 - r_target
        if dist_to_center > max_allowed:
            # Push inward
            inward = -points[i] / dist_to_center
            excess = dist_to_center - max_allowed
            forces[i] += inward * excess * boundary_strength
    
    return forces


def _relax(points, r_target, max_iters=5000, lr=0.001, cooling=0.999):
    """Iteratively relax the configuration to minimize overlaps."""
    points = points.copy()
    n = len(points)
    
    for it in range(max_iters):
        forces = _compute_forces(points, r_target)
        
        # Adaptive learning rate
        effective_lr = lr * (cooling ** it)
        points += effective_lr * forces
        
        # Project back inside the container
        dists = np.linalg.norm(points, axis=1)
        max_allowed = 1.0 - r_target
        for i in range(n):
            if dists[i] > max_allowed:
                points[i] = points[i] * (max_allowed / dists[i])
        
        # Check convergence
        if np.max(np.abs(forces)) < 1e-6:
            break
    
    return points


def _score(points):
    """Compute the score: minimum of pairwise distances/2 and distance to boundary."""
    n = len(points)
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    min_boundary = np.min(1.0 - np.linalg.norm(points, axis=1))
    
    return min(min_pair_dist / 2.0, min_boundary)


def _hexagonal_init(n, r_target):
    """Generate a hexagonal lattice initial configuration."""
    spacing = 2.0 * r_target
    points = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - r_target:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) >= n:
        # Select n points closest to center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
    else:
        # Fall back to random
        return np.random.randn(n, 2) * 0.5


def _random_init(n, r_target):
    """Generate a random initial configuration."""
    points = []
    for _ in range(n):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if np.hypot(x, y) <= 1.0 - r_target:
                points.append([x, y])
                break
    return np.array(points)


def _ring_init(n, r_target):
    """Generate a ring-based initial configuration."""
    points = []
    # Place points in concentric rings
    ring_radii = [0.0]
    r = r_target * 2.5
    while r < 1.0 - r_target:
        ring_radii.append(r)
        r += r_target * 2.2
    
    points.append([0.0, 0.0])
    for radius in ring_radii[1:]:
        # Number of points that fit on this ring
        circumference = 2 * np.pi * radius
        n_on_ring = max(1, int(circumference / (2.2 * r_target)))
        for i in range(n_on_ring):
            angle = 2 * np.pi * i / n_on_ring
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    return points


def construct():
    """Construct the optimal packing of 50 circles in a unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Try different initial configurations and relaxation strategies
    init_methods = [_hexagonal_init, _random_init, _ring_init]
    
    for method_idx, init_method in enumerate(init_methods):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for each init method
        n_restarts = 3 if method_idx == 0 else 2
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Start with a reasonable estimate of the radius
            r_target = 0.12
            
            # Generate initial configuration
            points = init_method(N, r_target)
            
            # Relax the configuration
            points = _relax(points, r_target, max_iters=3000, lr=0.002, cooling=0.9995)
            
            # Compute score
            score = _score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
            
            # Try to increase the radius if we have room
            if score > r_target * 0.95:
                # Increase target radius slightly
                r_new = r_target * 1.02
                points = _relax(points, r_new, max_iters=1000, lr=0.001, cooling=0.999)
                score_new = _score(points)
                if score_new > best_score:
                    best_score = score_new
                    best_points = points.copy()
    
    # Final refinement with a higher precision relaxation
    if best_points is not None and time.time() < deadline:
        current_score = _score(best_points)
        # Try to fine-tune
        for r_try in [current_score * 0.99, current_score * 1.01]:
            if time.time() > deadline:
                break
            refined = _relax(best_points, r_try, max_iters=2000, lr=0.0005, cooling=0.9999)
            refined_score = _score(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined
    
    if best_points is None:
        # Fallback: simple hexagonal
        points = _hexagonal_init(N, 0.12)
        return points
    
    return best_points
# EVOLVE-BLOCK-END

