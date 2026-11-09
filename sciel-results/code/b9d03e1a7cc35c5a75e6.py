# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize


def _compute_score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using vectorization."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area calculation
    # i < j < k
    # We can use broadcasting to compute areas for all triples.
    # However, memory for 13 choose 3 is small (286), so we can just loop over i, j and vectorize k?
    # Or simply generate indices.
    
    # Efficient way:
    # p_i, p_j, p_k
    # Area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    
    min_area = float('inf')
    # For N=13, 286 triangles is very small. Vectorizing the whole thing at once is easy.
    # Generate all triples
    idx = np.triu_indices(n, k=1)
    # This gives pairs (i, j) where i < j. We need triples.
    
    # Let's stick to a slightly optimized loop or full vectorization.
    # Full vectorization for N=13:
    # i from 0 to N-3
    # j from i+1 to N-2
    # k from j+1 to N-1
    
    # We can create arrays of indices
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area == 0:
        return 0.0
    
    return min_area / A_hull


def _neg_score(points_flat: np.ndarray) -> float:
    """Negative score for minimization."""
    n = 13
    points = points_flat.reshape(n, 2)
    score = _compute_score(points)
    return -score


def _optimize_lbfgsb(points: np.ndarray, deadline: float, max_iter: int = 200) -> np.ndarray:
    """Optimize points using L-BFGS-B."""
    n = 13
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Check time before starting
    if time.time() > deadline:
        return points
    
    # L-BFGS-B is good for this kind of non-smooth but structured problem if we are careful.
    # The score function is non-smooth due to the min() and abs() and hull area changes.
    # However, L-BFGS-B often finds good local minima.
    
    try:
        result = minimize(
            _neg_score, 
            points.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 20}
        )
        
        if time.time() > deadline:
            return points
            
        if result.success or result.fun < -_compute_score(points):
            return result.x.reshape(n, 2)
    except Exception:
        pass
        
    return points


def _random_perturb(points: np.ndarray, magnitude: float = 0.1) -> np.ndarray:
    """Randomly perturb a point configuration."""
    perturbed = points.copy()
    noise = np.random.randn(*perturbed.shape) * magnitude
    perturbed += noise
    # Clip to [0, 1]
    np.clip(perturbed, 0, 1, out=perturbed)
    return perturbed


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Uses vectorized scoring and L-BFGS-B optimization with multiple restarts.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin: 20s before 600s limit
    
    best_points = None
    best_score = 0.0
    
    # Initial structured configurations
    initial_configs = []
    
    # Config 1: Grid
    pts = []
    for i in range(4):
        for j in range(4):
            x = 0.1 + 0.2 * i
            y = 0.1 + 0.2 * j
            pts.append([x, y])
    initial_configs.append(np.array(pts[:13]))
    
    # Config 2: Hexagonal
    hex_pts = []
    row_height = 0.1732
    for row in range(5):
        y = 0.1 + row * row_height
        if row % 2 == 0:
            for col in range(3):
                x = 0.15 + col * 0.3
                hex_pts.append([x, y])
        else:
            for col in range(3):
                x = 0.3 + col * 0.3
                hex_pts.append([x, y])
    if len(hex_pts) >= 13:
        initial_configs.append(np.array(hex_pts[:13]))
        
    # Config 3: Concentric
    center = [0.5, 0.5]
    concentric_pts = [center]
    for r in [0.2, 0.35]:
        for angle in np.linspace(0, 2 * np.pi, 5, endpoint=False):
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            concentric_pts.append([x, y])
    if len(concentric_pts) >= 13:
        initial_configs.append(np.array(concentric_pts[:13]))

    # Add random initializations
    rng = np.random.default_rng(seed=42)
    for seed in range(10):
        rng = np.random.default_rng(seed=seed)
        random_pts = rng.random((n, 2)) * 0.9 + 0.05
        initial_configs.append(random_pts)
        
    # Evaluate and optimize each config
    for idx, config in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
            
        # Initial score
        score = _compute_score(config)
        if score > best_score:
            best_score = score
            best_points = config.copy()
            
        # Optimize
        optimized = _optimize_lbfgsb(config, deadline, max_iter=100)
        opt_score = _compute_score(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()
            
    # If we have time, do more random restarts from the best solution
    if best_points is not None and time.time() < deadline - 20:
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Perturb the best solution
            perturbed = _random_perturb(best_points, magnitude=0.1)
            
            # Optimize
            optimized = _optimize_lbfgsb(perturbed, deadline, max_iter=50)
            score = _compute_score(optimized)
            
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
                
    # Final safety
    if best_points is None or not np.all(np.isfinite(best_points)):
        xs = np.linspace(0.1, 0.9, 4)
        ys = np.linspace(0.1, 0.9, 4)
        fallback = []
        for x in xs:
            for y in ys:
                fallback.append([x, y])
        best_points = np.array(fallback[:13])
        
    return best_points


# EVOLVE-BLOCK-END
