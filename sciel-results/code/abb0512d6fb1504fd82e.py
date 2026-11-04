<<<<<<< SEARCH
def _generate_fps_initialization(rng, n_points=N, n_candidates=300):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    points[0] = (V0 + V1 + V2) / 3.0
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
    return points
=======
def _generate_fps_initialization(rng, n_points=N, n_candidates=300):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    points[0] = (V0 + V1 + V2) / 3.0
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
    return points

def _structured_inits():
    """
    Generate structured initial configurations:
    1. Grid-like patterns
    2. Symmetric arrangements
    3. Known good starting points
    """
    inits = []
    
    # Grid pattern in barycentric coordinates
    # For 11 points, try a 3-3-3-2 grid or similar
    # Barycentric coords (a, b, c) with a+b+c=1
    # Point = a*V0 + b*V1 + c*V2
    
    # Symmetric arrangement: center + 6 around + 4 more
    # Center
    pts = np.array([
        [0.5, np.sqrt(3)/6],  # center
        # 6 points forming a hexagon
        [0.5, np.sqrt(3)/6 + 0.15],
        [0.5 + 0.15*np.sqrt(3)/2, np.sqrt(3)/6 + 0.075],
        [0.5 + 0.15*np.sqrt(3)/2, np.sqrt(3)/6 - 0.075],
        [0.5, np.sqrt(3)/6 - 0.15],
        [0.5 - 0.15*np.sqrt(3)/2, np.sqrt(3)/6 - 0.075],
        [0.5 - 0.15*np.sqrt(3)/2, np.sqrt(3)/6 + 0.075],
        # 4 more points
        [0.25, np.sqrt(3)/12],
        [0.75, np.sqrt(3)/12],
        [0.5, np.sqrt(3)/4],
        [0.2, 0.05]
    ])
    # Clip to triangle
    for i in range(len(pts)):
        x, y = pts[i]
        # Check if inside triangle: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
        if y < 0: y = 0
        if y > np.sqrt(3)*x: y = np.sqrt(3)*x * 0.99
        if y > np.sqrt(3)*(1-x): y = np.sqrt(3)*(1-x) * 0.99
        pts[i] = [x, y]
    inits.append(_points_to_param(pts))
    
    # Another symmetric arrangement
    pts2 = np.array([
        [0.5, np.sqrt(3)/6],
        [0.5, np.sqrt(3)/3],
        [0.25, np.sqrt(3)/12],
        [0.75, np.sqrt(3)/12],
        [0.1, np.sqrt(3)/20],
        [0.9, np.sqrt(3)/20],
        [0.3, np.sqrt(3)/8],
        [0.7, np.sqrt(3)/8],
        [0.5, np.sqrt(3)/16],
        [0.15, np.sqrt(3)/10],
        [0.85, np.sqrt(3)/10]
    ])
    for i in range(len(pts2)):
        x, y = pts2[i]
        if y < 0: y = 0
        if y > np.sqrt(3)*x: y = np.sqrt(3)*x * 0.99
        if y > np.sqrt(3)*(1-x): y = np.sqrt(3)*(1-x) * 0.99
        pts2[i] = [x, y]
    inits.append(_points_to_param(pts2))
    
    return inits
>>>>>>> REPLACE

