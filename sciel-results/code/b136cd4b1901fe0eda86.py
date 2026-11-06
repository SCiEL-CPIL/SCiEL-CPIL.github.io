import numpy as np
import time

N = 200

def _compute_radius(pts):
    """Compute the maximum radius for given points."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_c = np.min(dist) / 2.0
    
    # Boundary distances
    dx = np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0])
    dy = np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    min_b = np.min(np.minimum(dx, dy))
    
    return min(min_c, min_b)

def _initial_hex_grid():
    """Generate initial hexagonal grid configuration."""
    points_list = []
    # Hexagonal packing: d ~ 0.076 for 200 points in unit square
    d = 0.076
    dy = d * np.sqrt(3) / 2
    
    # Generate rows
    j = 0
    while True:
        y = -0.5 + j * dy
        if y > 0.5:
            break
        x_offset = (j % 2) * d / 2
        i = 0
        while True:
            x = -0.5 + x_offset + i * d
            if x > 0.5:
                break
            points_list.append([x, y])
            i += 1
        j += 1
    
    points = np.array(points_list)
    
    # Select N points - use a stratified approach for good distribution
    if len(points) >= N:
        # Sort by distance from center to prefer central points
        dists = np.sqrt(np.sum(points**2, axis=1))
        # Shuffle with some randomness but prefer spread
        # Use a simple approach: divide into cells and pick one from each
        # For now, just take a random subset but ensure good spread
        # Better: use k-means-like selection
        # Simple approach: sort by angle and distance, pick evenly
        angles = np.arctan2(points[:, 1], points[:, 0])
        sorted_idx = np.lexsort((dists, angles))
        
        # Pick N points evenly from the sorted list
        step = len(points) / N
        selected = []
        for k in range(N):
            idx = int(k * step)
            if idx < len(points):
                selected.append(sorted_idx[idx])
        
        # If we have duplicates, fix them
        selected = list(dict.fromkeys(selected))
        while len(selected) < N:
            # Add random points
            random_idx = np.random.randint(len(points))
            if random_idx not in selected:
                selected.append(random_idx)
        
        points = points[selected[:N]].copy()
    else:
        # Pad with random points
        remaining = N - len(points)
        new_points = np.random.uniform(-0.45, 0.45, size=(remaining, 2))
        points = np.vstack([points, new_points])
    
    # Clip to ensure inside
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    return points

def _repulsion_step(points, r_target, dt, start_time, deadline):
    """Perform one repulsion step with wall forces."""
    n = len(points)
    forces = np.zeros_like(points)
    
    # Pairwise repulsion
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Only consider pairs that are too close
    mask = dist < 2 * r_target
    if np.any(mask):
        # Force is proportional to overlap
        overlap = np.maximum(0, 2 * r_target - dist)
        # Unit vectors
        unit = diff / (dist + 1e-10)[:, :, np.newaxis]
        # Sum forces
        forces += np.sum(overlap[:, :, np.newaxis] * unit, axis=1)
    
    # Wall forces
    # Left wall: x < -0.5 + r
    mask_left = points[:, 0] < -0.5 + r_target
    forces[mask_left, 0] += r_target - points[mask_left, 0] - 0.5 + 0.5  # = r_target - (x + 0.5)
    # Right wall: x > 0.5 - r
    mask_right = points[:, 0] > 0.5 - r_target
    forces[mask_right, 0] -= points[mask_right, 0] - 0.5 + r_target
    # Bottom wall: y < -0.5 + r
    mask_bottom = points[:, 1] < -0.5 + r_target
    forces[mask_bottom, 1] += r_target - points[mask_bottom, 1] - 0.5 + 0.5
    # Top wall: y > 0.5 - r
    mask_top = points[:, 1] > 0.5 - r_target
    forces[mask_top, 1] -= points[mask_top, 1] - 0.5 + r_target
    
    # Update positions
    points = points + dt * forces
    
    # Clip to boundary
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    return points

def _optimize_repulsion(points, start_time, deadline):
    """Optimize using repulsion dynamics with increasing target radius."""
    current_r = _compute_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # Start with a small target radius and increase
    r_target = current_r * 0.95
    dt = 0.001
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Perform several steps
        for _ in range(10):
            if time.time() > deadline:
                break
            points = _repulsion_step(points, r_target, dt, start_time, deadline)
        
        # Check current radius
        current_r = _compute_radius(points)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Increase target radius if we're making progress
        if current_r > r_target * 0.98:
            r_target = min(r_target * 1.01, 0.04)  # Cap at known best
        else:
            # Decrease dt or restart
            dt *= 0.9
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    return best_points, best_r

def _bottleneck_local_opt(points, start_time, deadline):
    """Local optimization focusing on bottleneck pairs."""
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    max_iter = 1000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute radius and find bottleneck
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find minimum distance pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        min_dist = dist[i, j]
        current_r = min_dist / 2
        
        # Also check boundary
        dx = np.minimum(points[:, 0] + 0.5, 0.5 - points[:, 0])
        dy = np.minimum(points[:, 1] + 0.5, 0.5 - points[:, 1])
        min_b = np.min(np.minimum(dx, dy))
        current_r = min(current_r, min_b)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Try to move i and j apart
        if min_dist < 2 * best_r * 1.05:
            # Move them apart
            vec = points[i] - points[j]
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 1e-10:
                vec = vec / vec_norm
                points[i] += 0.001 * vec
                points[j] -= 0.001 * vec
        
        # Also try random perturbations
        if it % 100 == 0:
            idx = np.random.randint(N)
            points[idx] += np.random.normal(0, 0.001, 2)
            points[idx] = np.clip(points[idx], -0.49, 0.49)
    
    return best_points, best_r

def _polish(points, start_time, deadline):
    """Final polish with small steps."""
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    dt = 0.0001
    for it in range(5000):
        if time.time() > deadline:
            break
        
        current_r = _compute_radius(points)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        points = _repulsion_step(points, best_r, dt, start_time, deadline)
    
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    # Initialize
    points = _initial_hex_grid()
    
    # Phase 1: Repulsion optimization
    points, r1 = _optimize_repulsion(points, start_time, deadline)
    
    # Phase 2: Bottleneck local optimization
    if time.time() < deadline - 20:
        points, r2 = _bottleneck_local_opt(points, start_time, deadline)
        if r2 > r1:
            r1 = r2
    
    # Phase 3: Polish
    if time.time() < deadline - 5:
        points, r3 = _polish(points, start_time, deadline)
        if r3 > r1:
            r1 = r3
    
    # Final check
    final_r = _compute_radius(points)
    if final_r < r1:
        # Use the best we had
        pass
    
    return points
