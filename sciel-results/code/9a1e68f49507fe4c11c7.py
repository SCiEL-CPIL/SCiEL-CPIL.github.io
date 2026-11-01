"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0
    
    # Min distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))
    
    # Min pair distance via vectorized computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))
    
    return min(min_pair / 2.0, min_boundary)

def _soft_min_gradient(points, beta=50.0):
    """Compute gradient of soft-min of pair distances and boundary distances.
    
    Uses log-sum-exp trick for numerical stability.
    Returns (soft_min_value, gradient) where gradient points in direction to increase soft-min.
    """
    n = len(points)
    
    # Compute all pair distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    
    # Combine all constraints: pair distances / 2 and boundary distances
    # We want to maximize the minimum of these
    all_dists = np.concatenate([dist_matrix[dist_matrix < np.inf] / 2.0, boundary_dists])
    
    # Soft-min using log-sum-exp
    max_val = np.max(all_dists)
    exp_vals = np.exp(-beta * (all_dists - max_val))
    sum_exp = np.sum(exp_vals)
    soft_min = max_val - np.log(sum_exp) / beta
    
    # Gradient computation
    # d(soft_min)/d(all_dists) = -exp_vals / sum_exp (negative because we want to increase min)
    weights = -exp_vals / sum_exp
    
    grad = np.zeros_like(points)
    
    # Boundary contribution
    boundary_weights = weights[-n:]
    for i in range(n):
        r = np.linalg.norm(points[i])
        if r > 1e-10:
            grad[i] += boundary_weights[i] * points[i] / r
    
    # Pair distance contribution
    pair_weights = weights[:-n]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                w = pair_weights[idx]
                grad[i] += w * d / dist
                grad[j] -= w * d / dist
            idx += 1
    
    return soft_min, grad

def _local_optimize(points, max_iter=2000, lr=0.001, beta=50.0):
    """Local optimization using soft-min gradient approach."""
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    best_soft_min, _ = _soft_min_gradient(best_points, beta)
    
    for iteration in range(max_iter):
        soft_min, grad = _soft_min_gradient(points, beta)
        
        # Gradient ascent with adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        grad_normalized = grad / grad_norm
        points_new = points + lr * grad_normalized
        
        # Ensure points stay inside unit circle
        for i in range(n):
            r = np.linalg.norm(points_new[i])
            if r > 1.0:
                points_new[i] = points_new[i] * (1.0 / r)
        
        score = _compute_score(points_new)
        if score > best_score:
            best_score = score
            best_points = points_new.copy()
            points = points_new
            lr = min(lr * 1.05, 0.01)
        else:
            lr *= 0.95
        
        # Check convergence
        if iteration > 100 and abs(score - best_score) < 1e-8:
            break
    
    return best_points

def _radial_ring_init():
    """Initialize points using radial-ring decomposition.
    
    Place circles in concentric rings, optimizing the number of circles per ring
    and radial distances to maximize the minimum distance.
    """
    # Start with a reasonable number of rings
    # For N=300, we can have rings with decreasing circumference
    # Ring k has radius r_k and n_k circles
    
    # Try different ring configurations
    best_points = None
    best_score = -np.inf
    
    # Configuration: list of (radius, num_circles, angular_offset)
    # We'll optimize these parameters
    
    # Initial guess: use hexagonal spacing as starting point
    target_spacing = 0.106  # Approximately 2 * 0.053
    max_radius = 1.0 - target_spacing / 2.0
    
    # Create rings
    ring_radii = []
    ring_counts = []
    remaining = N
    
    r = 0.0
    while remaining > 0 and r < max_radius:
        # Circumference at radius r
        if r < 1e-6:
            # Center point
            n_rings = 1
        else:
            circumference = 2 * np.pi * r
            n_rings = int(circumference / target_spacing)
            n_rings = max(n_rings, 1)
        
        # Don't exceed remaining
        n_rings = min(n_rings, remaining)
        
        ring_radii.append(r)
        ring_counts.append(n_rings)
        remaining -= n_rings
        
        # Next ring radius
        r += target_spacing * 0.866  # Hexagonal spacing
    
    # Build points
    points_list = []
    for radius, count in zip(ring_radii, ring_counts):
        if count == 1 and radius < 1e-6:
            points_list.append([0.0, 0.0])
        elif count == 1:
            points_list.append([radius, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            for x, y in zip(xs, ys):
                points_list.append([x, y])
    
    points = np.array(points_list)
    
    # Adjust to exactly N points
    if len(points) > N:
        points = points[:N]
    elif len(points) < N:
        # Add more points on the outer ring
        deficit = N - len(points)
        outer_radius = ring_radii[-1] if ring_radii else 0.5
        angles = np.linspace(0, 2 * np.pi, deficit, endpoint=False)
        extra_xs = outer_radius * np.cos(angles)
        extra_ys = outer_radius * np.sin(angles)
        extra_points = np.column_stack([extra_xs, extra_ys])
        points = np.vstack([points, extra_points])
    
    return points

def _hex_lattice_init():
    """Initialize points using hexagonal lattice."""
    best_points = None
    best_score = -np.inf
    
    for angle_deg in [0, 30, 60, 90]:
        angle = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        for spacing_factor in [0.95, 0.97, 0.99, 1.0, 1.02]:
            spacing = 2.0 * spacing_factor / (np.sqrt(N) + 1.0)
            
            k = int(1.0 / spacing) + 3
            rows = []
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    xr = x * cos_a - y * sin_a
                    yr = x * sin_a + y * cos_a
                    rows.append((xr, yr))
            
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            
            if len(points) < N:
                continue
            
            radius = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radius, kind="stable")[:N]
            points = points[idx]
            
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    return best_points

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Try radial ring initialization
    if time.time() < deadline - 60:
        ring_points = _radial_ring_init()
        if ring_points is not None and len(ring_points) == N:
            score = _compute_score(ring_points)
            if score > best_score:
                best_score = score
                best_points = ring_points.copy()
    
    # Try hexagonal lattice initialization
    if time.time() < deadline - 60:
        hex_points = _hex_lattice_init()
        if hex_points is not None and len(hex_points) == N:
            score = _compute_score(hex_points)
            if score > best_score:
                best_score = score
                best_points = hex_points.copy()
    
    # Local optimization with soft-min gradient
    if best_points is not None and time.time() < deadline - 30:
        optimized = _local_optimize(best_points, max_iter=1500, lr=0.0005, beta=50.0)
        opt_score = _compute_score(optimized)
        if opt_score > best_score:
            best_points = optimized
            best_score = opt_score
    
    # Final validation
    if best_points is None:
        # Emergency fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0, 0.9, N)
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
