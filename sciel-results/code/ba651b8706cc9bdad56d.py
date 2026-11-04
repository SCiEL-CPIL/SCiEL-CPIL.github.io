# EVOLVE-BLOCK-START
def _score_vec(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns negative score for minimization.
    Vectorized for speed.
    """
    points = np.asarray(points)
    n = len(points)
    
    # Cache indices for combinations
    if not hasattr(_score_vec, 'indices'):
        _score_vec.indices = np.array(list(itertools.combinations(range(N), 3)))
    idx = _score_vec.indices
        
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    if min_area <= 1e-12:
        return 0.0
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    score = min_area / hull_area
    return -score

def construct():
    # Multi-start optimization
    best_fun = float('inf')
    best_points = None
    
    # Generate initial points
    starts = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    starts.append(np.column_stack((x, y)).flatten())
    
    # 2. Jittered Regular 13-gon
    for _ in range(5):
        noise = np.random.normal(0, 0.1, N * 2)
        starts.append(np.column_stack((x, y)).flatten() + noise)
        
    # 3. Random points in unit disk
    for _ in range(10):
        r = np.sqrt(np.random.uniform(0, 1, N))
        theta = np.random.uniform(0, 2 * np.pi, N)
        px = r * np.cos(theta)
        py = r * np.sin(theta)
        starts.append(np.column_stack((px, py)).flatten())
        
    # Optimize each start
    for x0 in starts:
        res = minimize(_score_vec, x0, method='Nelder-Mead', 
                       options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.fun < best_fun:
            best_fun = res.fun
            best_points = res.x.reshape(-1, 2)
            
    if best_points is None:
        # Fallback to regular polygon
        best_points = np.column_stack((x, y))
        
    return best_points
# EVOLVE-BLOCK-END
