"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _compute_facet_data(points):
    """
    Computes the circumradius and circumcenter direction for each facet of the convex hull.
    Returns:
        radii: array of shape (M,)
        centers: array of shape (M, 3) unit vectors
        simplices: array of shape (M, 3)
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        # Degenerate facets, handle gracefully
        radii = np.full(len(simplices), np.pi)
        centers = np.zeros((len(simplices), 3))
        centers[~valid] = [0, 0, 1]
        return radii, centers, simplices
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices


def _covering_radius(pts):
    """
    Compute the covering radius in degrees using convex hull facets.
    """
    pts = _normalize(pts)
    radii, _, _ = _compute_facet_data(pts)
    if len(radii) == 0:
        # Fallback to sampling
        return _covering_radius_fallback(pts)
    return np.degrees(np.max(radii))


def _covering_radius_fallback(pts):
    """Fallback: approximate covering radius by sampling the sphere."""
    n_samples = 5000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = dots.max(axis=1)
    min_angles = np.arccos(min_dots)
    return np.degrees(min_angles.max())


def _fibonacci_init():
    """Generate Fibonacci sphere initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _greedy_init(seed=42, n_candidates=2000):
    """
    Greedy initialization: start with a random point, then repeatedly add the point
    that maximizes the minimum distance to existing points.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(N - 1):
        candidates = np.random.randn(n_candidates, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        existing = np.array(points)
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
    
    return np.array(points)


def _local_optimize(points, time_budget, alpha_init=0.05):
    """
    Facet-based local optimization: move vertices of large-radius facets
    toward the facet circumcenter.
    """
    start = time.time()
    best_local = points.copy()
    best_local_radius = _covering_radius(points)
    alpha = alpha_init
    no_improve = 0
    
    while time.time() - start < time_budget:
        radii, centers, simplices = _compute_facet_data(points)
        if len(radii) == 0:
            break
        
        max_radius = np.max(radii)
        
        if max_radius < best_local_radius:
            best_local_radius = max_radius
            best_local = points.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 30:
                alpha *= 0.9
                if alpha < 1e-5:
                    break
            if no_improve > 150:
                break
        
        # Identify facets to move (top 20% by radius)
        if len(radii) > 5:
            threshold = np.percentile(radii, 80)
            mask = radii > threshold
        else:
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True
        
        if not np.any(mask):
            break
        
        updates = np.zeros_like(points)
        
        for m in np.where(mask)[0]:
            simplex = simplices[m]
            i_idx, j_idx, k_idx = simplex
            
            a = points[i_idx]
            b = points[j_idx]
            c = points[k_idx]
            
            center = centers[m]
            
            step_size = alpha * radii[m]
            
            # Project (center - vertex) onto tangent space
            for v, idx in [(a, i_idx), (b, j_idx), (c, k_idx)]:
                tangent = center - np.dot(center, v) * v
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-12:
                    updates[idx] += step_size * (tangent / norm_t)
        
        points += updates
        points = _normalize(points)
        
        # Decay step size
        alpha *= 0.997
    
    return best_local


def _polish(points, time_budget):
    """
    High-precision polish pass with tighter tolerance and adaptive step size.
    """
    start = time.time()
    alpha = 0.01
    best_local = points.copy()
    best_local_radius = _covering_radius(points)
    
    while time.time() - start < time_budget:
        radii, centers, simplices = _compute_facet_data(points)
        if len(radii) == 0:
            break
            
        max_radius = np.max(radii)
        
        if max_radius < best_local_radius:
            best_local_radius = max_radius
            best_local = points.copy()
        
        # Only move the largest facets (top 10%)
        if len(radii) > 5:
            threshold = np.percentile(radii, 90)
            mask = radii > threshold
        else:
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True
            
        if not np.any(mask):
            break
            
        updates = np.zeros_like(points)
        
        for m in np.where(mask)[0]:
            simplex = simplices[m]
            i_idx, j_idx, k_idx = simplex
            
            a = points[i_idx]
            b = points[j_idx]
            c = points[k_idx]
            
            center = centers[m]
            
            step_size = alpha * radii[m]
            
            for v, idx in [(a, i_idx), (b, j_idx), (c, k_idx)]:
                tangent = center - np.dot(center, v) * v
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-12:
                    updates[idx] += step_size * (tangent / norm_t)
        
        points += updates
        points = _normalize(points)
        
        # Decay step size
        alpha *= 0.995
        
        if alpha < 1e-6:
            break
            
    return best_local


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_radius = np.pi
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    init_fib = _fibonacci_init()
    initializations.append(init_fib)
    
    # 2. Random points
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand = _normalize(init_rand)
    initializations.append(init_rand)
    
    # 3. Greedy initialization
    init_greedy = _greedy_init(seed=42, n_candidates=1000)
    initializations.append(init_greedy)
    
    # 4. Another greedy with different seed
    init_greedy2 = _greedy_init(seed=123, n_candidates=1000)
    initializations.append(init_greedy2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise = _normalize(init_fib_noise)
    initializations.append(init_fib_noise)
    
    # 6-10. More randomized restarts
    for s in range(5):
        np.random.seed(1000 + s * 7)
        init = init_fib + np.random.randn(N, 3) * (0.1 + 0.05 * s)
        init = _normalize(init)
        initializations.append(init)
    
    # 11-15. Different noise levels on Fibonacci
    for s in range(5):
        np.random.seed(2000 + s * 13)
        init = init_fib + np.random.randn(N, 3) * (0.05 + 0.03 * s)
        init = _normalize(init)
        initializations.append(init)
    
    # 16-20. Pure random with different seeds
    for s in range(5):
        np.random.seed(3000 + s * 23)
        init = np.random.randn(N, 3)
        init = _normalize(init)
        initializations.append(init)
    
    # Sort initializations by their initial covering radius (best first)
    init_radii = []
    for init in initializations:
        r = _covering_radius(init)
        init_radii.append((r, init))
    init_radii.sort(key=lambda x: x[0])
    
    # Local optimization on each initialization
    for radius, init_pts in init_radii:
        if time.time() > deadline - 10.0:
            break
        
        # Time budget for this restart
        remaining = deadline - time.time()
        time_budget = min(remaining * 0.3, 20.0)  # Use at most 20s per restart
        
        if time_budget < 2.0:
            break
        
        # Local optimization
        optimized = _local_optimize(init_pts.copy(), time_budget)
        opt_radius = _covering_radius(optimized)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = optimized.copy()
    
    # Final polish on the best result
    if best_pts is not None and time.time() < deadline - 5.0:
        polish_budget = min(deadline - time.time() - 5.0, 15.0)
        polished = _polish(best_pts.copy(), polish_budget)
        polished_radius = _covering_radius(polished)
        
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_pts = polished.copy()
    
    if best_pts is None:
        best_pts = init_radii[0][1]
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
