"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of unit points.
    """
    if len(points) < 3:
        return np.pi
    try:
        hull = ConvexHull(points)
    except:
        return np.pi
    
    max_angle = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        if angle > max_angle:
            max_angle = angle
            
    return max_angle * 180.0 / np.pi


def _smooth_covering_radius(points, beta=100.0):
    """
    Smooth approximation of the covering radius using log-sum-exp.
    This provides a differentiable surrogate for optimization.
    """
    try:
        hull = ConvexHull(points)
    except:
        return np.pi
    
    angles = []
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        angles.append(angle)
    
    if not angles:
        return np.pi
    
    angles = np.array(angles)
    # Log-sum-exp for smooth max
    max_a = np.max(angles)
    if max_a == 0:
        return 0.0
    log_sum = max_a + np.log(np.sum(np.exp((angles - max_a) * beta)) / beta)
    return log_sum * 180.0 / np.pi


def _smooth_objective(params, beta=50.0):
    """Smooth objective for gradient-based optimization."""
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    return _smooth_covering_radius(pts, beta=beta)


def _fibonacci_points():
    """Generate Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    v = np.random.randn(N, 3)
    return _normalize(v)


def _icosahedron_based_points():
    """Generate points based on icosahedron with subdivisions."""
    # Start with icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    # Subdivide faces to get more points
    # For 50 points, we can use a few subdivisions
    points = [verts]
    
    # Simple approach: use Fibonacci as base and perturb
    fib = _fibonacci_points()
    # Add some random perturbation
    noise = np.random.randn(N, 3) * 0.1
    perturbed = _normalize(fib + noise)
    return perturbed


def _spherical_lloyd_iteration(points, max_iter=200, time_limit=60, damping=0.5):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Points are moved toward the centroids of their spherical Voronoi cells.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        new_points = points.copy()
        
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
        
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
            
            normal_sum = np.zeros(3)
            for fi in facets:
                simplex = hull.simplices[fi]
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                
                if np.dot(n, p1) < 0:
                    n = -n
                
                normal_sum += n
            
            if np.linalg.norm(normal_sum) > 1e-10:
                target = normal_sum / np.linalg.norm(normal_sum)
                new_points[pi] = points[pi] + damping * (target - points[pi])
                new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            break
    
    return best_points


def _smooth_gradient_optimize(points, time_limit=30, beta=50.0):
    """
    Gradient-based optimization using smooth surrogate.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _smooth_objective(params, beta=beta)
    
    x0 = points.flatten()
    
    # Use L-BFGS-B for smooth optimization
    result = minimize(
        objective, x0, method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _sa_optimize_covering(points, T_start=1.0, T_end=1e-4, n_iter=5000, time_limit=100):
    """
    Simulated Annealing to minimize covering radius directly.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        T = T_start * (cooling ** i)
        
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        step_size = 0.05 * T / T_start
        noise = np.random.normal(0, step_size, 3)
        new_point = original_point + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        points[idx] = new_point
        
        new_cost = _covering_radius(points)
        delta_C = new_cost - current_cost
        
        if delta_C < 0 or np.random.rand() < np.exp(-delta_C / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            points[idx] = original_point
            
    return best_points


def _local_polish(points, time_limit=15):
    """
    Final high-precision polish using exact covering radius with tight tolerance.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius(pts)
    
    x0 = points.flatten()
    result = minimize(
        objective, x0, method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0
    start_time = time.time()
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # 2. Random points with different seeds
    for seed in [42, 123, 456, 789, 1011]:
        initializations.append(_random_points(seed=seed))
    
    # 3. Perturbed Fibonacci
    fib = _fibonacci_points()
    noise = np.random.randn(N, 3) * 0.15
    initializations.append(_normalize(fib + noise))
    
    # Evaluate all initializations and keep the best
    best_points = None
    best_cost = np.pi
    
    for init in initializations:
        init = _normalize(init)
        cost = _covering_radius(init)
        if cost < best_cost:
            best_cost = cost
            best_points = init.copy()
    
    # Stage 1: Spherical Lloyd's algorithm
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 40:
        points_lloyd = _spherical_lloyd_iteration(
            best_points, max_iter=150, time_limit=remaining_time - 50, damping=0.5
        )
        cost_lloyd = _covering_radius(points_lloyd)
        if cost_lloyd < best_cost:
            best_points = points_lloyd
            best_cost = cost_lloyd
    
    # Stage 2: Smooth gradient optimization
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 30:
        points_smooth = _smooth_gradient_optimize(
            best_points, time_limit=remaining_time - 35, beta=50.0
        )
        cost_smooth = _covering_radius(points_smooth)
        if cost_smooth < best_cost:
            best_points = points_smooth
            best_cost = cost_smooth
    
    # Stage 3: Simulated Annealing
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 25:
        points_sa = _sa_optimize_covering(
            best_points, T_start=0.5, T_end=1e-4, n_iter=8000, time_limit=remaining_time - 30
        )
        cost_sa = _covering_radius(points_sa)
        if cost_sa < best_cost:
            best_points = points_sa
            best_cost = cost_sa
    
    # Stage 4: Another Lloyd's pass with smaller damping
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 15:
        points_lloyd2 = _spherical_lloyd_iteration(
            best_points, max_iter=80, time_limit=remaining_time - 20, damping=0.3
        )
        cost_lloyd2 = _covering_radius(points_lloyd2)
        if cost_lloyd2 < best_cost:
            best_points = points_lloyd2
            best_cost = cost_lloyd2
    
    # Stage 5: Final local polish
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 10:
        points_final = _local_polish(best_points, time_limit=remaining_time - 10)
        cost_final = _covering_radius(points_final)
        if cost_final < best_cost:
            best_points = points_final
            best_cost = cost_final
    
    # Final normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
