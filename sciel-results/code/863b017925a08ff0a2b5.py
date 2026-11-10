<<<<<<< SEARCH
def _generate_random(n, rng):
    """Generate a random compact cluster."""
    radius = 1.1 * n ** (1.0/3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 for q in points):
            points.append(p)
        attempts += 1
    
    # Fallback: add points with decreasing spacing requirement
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        else:
            # Try closer
            for _ in range(10):
                p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
                    break
    
    return np.array(points[:n])
=======
def _generate_random(n, rng):
    """Generate a random compact cluster."""
    radius = 1.1 * n ** (1.0/3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in points):
            points.append(p)
        attempts += 1
    
    # Fallback: add points with decreasing spacing requirement
    min_dist = 0.5
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        else:
            # Try closer
            for _ in range(20):
                p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
                    break
    
    return np.array(points[:n])
>>>>>>> REPLACE

