# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time


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
        
        # Normal to the plane
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        # Ensure n points away from origin
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        if angle > max_angle:
            max_angle = angle
            
    return max_angle * 180.0 / np.pi


def _lloyd_step(points):
    """
    Perform one step of Spherical Lloyd's Algorithm.
    Moves each point to the spherical centroid of its Voronoi cell.
    """
    try:
        hull = ConvexHull(points)
    except:
        return points

    N = len(points)
    # Map point index to list of incident simplex indices
    point_to_simplices = {i: [] for i in range(N)}
    for idx, simplex in enumerate(hull.simplices):
        for p_idx in simplex:
            point_to_simplices[p_idx].append(idx)

    new_points = np.zeros_like(points)
    
    for i in range(N):
        simplices = point_to_simplices[i]
        if not simplices:
            # If a point is not on the hull (shouldn't happen for covering), keep it
            new_points[i] = points[i]
            continue
            
        centroid_sum = np.zeros(3)
        for s_idx in simplices:
            simplex = hull.simplices[s_idx]
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
            
            # Ensure outward normal
            if np.dot(n, p1) < 0:
                n = -n
                
            centroid_sum += n
            
        norm_centroid = np.linalg.norm(centroid_sum)
        if norm_centroid < 1e-10:
            new_points[i] = points[i]
        else:
            new_points[i] = centroid_sum / norm_centroid
            
    return new_points


def _lloyd_optimize_covering(points, n_iter=100, time_limit=30):
    """
    Optimize using Spherical Lloyd's Algorithm.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
            
        new_points = _lloyd_step(points)
        new_points = _normalize(new_points)
        
        # Check if we converged or improved
        new_cost = _covering_radius(new_points)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = new_points.copy()
            
        # If no improvement for a while, might be stuck, but let's just run
        # A simple convergence check: if points didn't move much
        move = np.linalg.norm(new_points - points)
        points = new_points
        
        if move < 1e-6:
            break
            
    return best_points


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


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0  # Safety margin before 150s kill
    start_time = time.time()
    
    # 1. Initialize with Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = x0
    best_cost = _covering_radius(best_points)
    
    # 2. Initial Lloyd's to settle geometry
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 20:
        points_lloyd1 = _lloyd_optimize_covering(best_points, n_iter=50, time_limit=10)
        cost_lloyd1 = _covering_radius(points_lloyd1)
        if cost_lloyd1 < best_cost:
            best_points = points_lloyd1
            best_cost = cost_lloyd1

    # 3. Simulated Annealing to escape local minima
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 20:
        points_sa = _sa_optimize_covering(best_points, T_start=1.0, T_end=1e-4, n_iter=10000, time_limit=remaining_time - 20)
        cost_sa = _covering_radius(points_sa)
        if cost_sa < best_cost:
            best_points = points_sa
            best_cost = cost_sa

    # 4. Final Lloyd's polish
    time_spent = time.time() - start_time
    remaining_time = total_time_limit - time_spent
    if remaining_time > 5:
        points_final = _lloyd_optimize_covering(best_points, n_iter=100, time_limit=remaining_time - 5)
        cost_final = _covering_radius(points_final)
        if cost_final < best_cost:
            best_points = points_final
            best_cost = cost_final
    
    # 5. Final check and normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
