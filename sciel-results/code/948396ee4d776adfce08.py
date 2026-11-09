<<<<<<< SEARCH
def _generate_hex_init():
    """
    Generate an initial hexagonal packing.
    """
    # Target N=200
    # Hex grid spacing s
    # Area per point in hex grid is s^2 * sqrt(3)/2
    # Total Area = 1.0
    # N * s^2 * sqrt(3)/2 = 1.0  => s = sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate points in a grid larger than the square
    # Square is [-0.5, 0.5]
    # We need to cover this area.
    
    points = []
    # y range
    dy = s * np.sqrt(3.0) / 2.0
    # Start from -0.6 to 0.6 to ensure coverage
    y = -0.6
    row = 0
    while y <= 0.6:
        # x offset for alternating rows
        offset = 0.0 if row % 2 == 0 else s / 2.0
        x = -0.6 + offset
        while x <= 0.6:
            points.append((x, y))
            x += s
        y += dy
        row += 1
    
    points = np.array(points)
    
    # Filter points inside the square [-0.5, 0.5]
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N points, remove the ones with the worst local packing
    if len(points) > N:
        # Calculate min distance to neighbor for each point
        # This is O(M^2) where M is number of points. M is approx 200-300.
        # 300^2 = 90,000. Fast.
        min_dists = np.full(len(points), np.inf)
        for i in range(len(points)):
            dists = np.linalg.norm(points[i] - points, axis=1)
            dists[i] = np.inf
            min_dists[i] = np.min(dists)
        
        # Also consider wall distance?
        # Let's just use min neighbor distance for now.
        # Sort indices by min_dist, remove the smallest ones
        idx = np.argsort(min_dists)
        # Keep the largest N
        keep_idx = idx[N:]
        points = points[keep_idx]
        
    # If we have fewer than N points (unlikely with this s), we might need to add
    # But for N=200, s is small enough that we usually have >200 points in the square.
    # If len(points) < N, we just pad with random points? Or just return what we have?
    # The task requires exactly N.
    if len(points) < N:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
        
    # Jitter slightly to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    return points
=======
def _generate_hex_init():
    """
    Generate an initial hexagonal packing.
    """
    # Target N=200
    # Hex grid spacing s
    # Area per point in hex grid is s^2 * sqrt(3)/2
    # Total Area = 1.0
    # N * s^2 * sqrt(3)/2 = 1.0  => s = sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate points in a grid larger than the square
    # Square is [-0.5, 0.5]
    # We need to cover this area.
    
    points = []
    # y range
    dy = s * np.sqrt(3.0) / 2.0
    # Start from -0.6 to 0.6 to ensure coverage
    y = -0.6
    row = 0
    while y <= 0.6:
        # x offset for alternating rows
        offset = 0.0 if row % 2 == 0 else s / 2.0
        x = -0.6 + offset
        while x <= 0.6:
            points.append((x, y))
            x += s
        y += dy
        row += 1
    
    points = np.array(points)
    
    # Filter points inside the square [-0.5, 0.5]
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N points, remove the ones with the worst local packing
    if len(points) > N:
        # Calculate min distance to neighbor for each point
        # This is O(M^2) where M is number of points. M is approx 200-300.
        # 300^2 = 90,000. Fast.
        min_dists = np.full(len(points), np.inf)
        for i in range(len(points)):
            dists = np.linalg.norm(points[i] - points, axis=1)
            dists[i] = np.inf
            min_dists[i] = np.min(dists)
        
        # Also consider wall distance?
        # Let's just use min neighbor distance for now.
        # Sort indices by min_dist, remove the smallest ones
        idx = np.argsort(min_dists)
        # Keep the largest N
        keep_idx = idx[N:]
        points = points[keep_idx]
        
    # If we have fewer than N points (unlikely with this s), we might need to add
    # But for N=200, s is small enough that we usually have >200 points in the square.
    # If len(points) < N, we just pad with random points? Or just return what we have?
    # The task requires exactly N.
    if len(points) < N:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
        
    # Jitter slightly to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    return points

def _repulsive_relaxation(centers, n_iters=100, step_size=0.01, time_limit=None, start_time=None):
    """
    Perform repulsive relaxation to improve local packing.
    Each point is pushed away from its neighbors and walls.
    """
    if time_limit is None or start_time is None:
        return centers
    
    centers = centers.copy()
    n = len(centers)
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion force: proportional to 1/dist^2 in the direction away
        # Only consider close neighbors to save computation
        mask = dists < 0.1  # Only consider points within 0.1
        for i in range(n):
            for j in range(n):
                if i != j and mask[i, j]:
                    d = dists[i, j]
                    if d > 1e-10:
                        # Force magnitude: 1/d^2
                        f_mag = 1.0 / (d * d)
                        # Direction: from j to i
                        direction = (centers[i] - centers[j]) / d
                        forces[i] += f_mag * direction
        
        # Wall repulsion
        for i in range(n):
            for k in range(2):
                # Distance to wall
                if centers[i, k] > 0:
                    wall_dist = 0.5 - centers[i, k]
                    if wall_dist < 0.05:
                        forces[i, k] -= 1.0 / (wall_dist + 1e-10)
                else:
                    wall_dist = 0.5 + centers[i, k]
                    if wall_dist < 0.05:
                        forces[i, k] += 1.0 / (wall_dist + 1e-10)
        
        # Update positions
        centers += step_size * forces
        
        # Clip to bounds
        centers = np.clip(centers, -0.5, 0.5)
        
        # Reduce step size
        step_size *= 0.95
    
    return centers
>>>>>>> REPLACE

