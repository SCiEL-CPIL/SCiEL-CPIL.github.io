"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _all_triangle_areas(points):
    """Compute areas of all C(n,3) triangles."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    return np.array(areas)

def _score(points):
    """Score = min triangle area / convex hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # in 2D, volume is area
    except Exception:
        # Fallback: compute hull area manually
        from scipy.spatial import ConvexHull as CH
        try:
            hull = CH(points)
            hull_area = hull.volume
        except Exception:
            # Very rough fallback
            hull_area = 1.0
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _soft_min(areas, temperature=0.05):
    """Smooth approximation of min using log-sum-exp."""
    # soft_min = -temperature * log(sum(exp(-area/temperature)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum

def _soft_min_grad(areas, temperature=0.05):
    """Gradient of soft_min with respect to areas."""
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    # d(soft_min)/d(area_i) = weights[i] (since soft_min = -T * log(sum(exp(-a_i/T))))
    # d/da_i [-T * log(sum(exp(-a_j/T)))] = -T * (1/sum) * exp(-a_i/T) * (-1/T) = exp(-a_i/T)/sum = weights[i]
    return weights

def _optimize_from(points, time_limit, temperature=0.05):
    """Optimize points using smooth surrogate + local search."""
    points = points.copy()
    start_time = time.time()
    
    # Phase 1: Gradient-based optimization with soft-min surrogate
    def objective(x):
        if time.time() - start_time > time_limit * 0.5:
            return 1e10
        pts = x.reshape(-1, 2)
        areas = _all_triangle_areas(pts)
        return _soft_min(areas, temperature)
    
    def gradient(x):
        if time.time() - start_time > time_limit * 0.5:
            return np.zeros_like(x)
        pts = x.reshape(-1, 2)
        areas = _all_triangle_areas(pts)
        weights = _soft_min_grad(areas, temperature)
        
        # Compute gradient of each area with respect to each point
        n = len(pts)
        grad = np.zeros((n, 2))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    w = weights[idx]
                    v1 = pts[j] - pts[i]
                    v2 = pts[k] - pts[i]
                    cross = v1[0]*v2[1] - v1[1]*v2[0]
                    # d(area)/d(pts[i]) = -0.5 * sign(cross) * [v2, -v1] (for the cross product)
                    # Actually: area = 0.5*|cross(v1,v2)|
                    # d(cross)/d(pts[i]) = -[v2, -v1] = [-v2[0], v2[1], v1[0], -v1[1]]... 
                    # Let me be more careful.
                    # cross = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
                    # d(cross)/d(xi) = -(yk-yi) + (yj-yi) = yj-yk
                    # d(cross)/d(yi) = -(xk-xi) + (xj-xi) = xj-xk
                    # d(cross)/d(xj) = (yk-yi)
                    # d(cross)/d(yj) = -(xk-xi)
                    # d(cross)/d(xk) = -(yj-yi)
                    # d(cross)/d(yk) = (xj-xi)
                    
                    sign = 1.0 if cross >= 0 else -1.0
                    # d(area)/d(pts[i]) = 0.5 * sign * [-d(cross)/d(xi), -d(cross)/d(yi)]
                    # = 0.5 * sign * [-(yj-yk), -(xj-xk)]
                    grad[i, 0] += w * 0.5 * sign * (-(pts[j,1]-pts[k,1]))
                    grad[i, 1] += w * 0.5 * sign * (-(pts[j,0]-pts[k,0]))
                    
                    # d(area)/d(pts[j]) = 0.5 * sign * [d(cross)/d(xj), d(cross)/d(yj)]
                    # = 0.5 * sign * [(yk-yi), -(xk-xi)]
                    grad[j, 0] += w * 0.5 * sign * (pts[k,1]-pts[i,1])
                    grad[j, 1] += w * 0.5 * sign * (-(pts[k,0]-pts[i,0]))
                    
                    # d(area)/d(pts[k]) = 0.5 * sign * [d(cross)/d(xk), d(cross)/d(yk)]
                    # = 0.5 * sign * [-(yj-yi), (xj-xi)]
                    grad[k, 0] += w * 0.5 * sign * (-(pts[j,1]-pts[i,1]))
                    grad[k, 1] += w * 0.5 * sign * (pts[j,0]-pts[i,0])
                    
                    idx += 1
        
        return grad.flatten()
    
    x0 = points.flatten()
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                         options={'maxiter': 100, 'gtol': 1e-6, 'maxfun': 500})
        if result.x is not None:
            points = result.x.reshape(-1, 2)
    except Exception:
        pass
    
    # Phase 2: Local search targeting worst triangles
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 5:
        return points
    
    # Get all triangle indices
    tri_indices = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_indices.append((i, j, k))
    
    best_score = _score(points)
    
    for iteration in range(200):
        if time.time() - start_time > time_limit - 5:
            break
        
        areas = _all_triangle_areas(points)
        worst_idx = np.argmin(areas)
        i, j, k = tri_indices[worst_idx]
        
        # Try to improve by moving one of the three points
        best_move = None
        best_improvement = 0
        current_min = areas[worst_idx]
        
        for move_point in [i, j, k]:
            for step_size in [0.01, 0.05, 0.1, 0.2]:
                for dx in [-step_size, step_size]:
                    for dy in [-step_size, step_size]:
                        new_points = points.copy()
                        new_points[move_point, 0] += dx
                        new_points[move_point, 1] += dy
                        
                        new_areas = _all_triangle_areas(new_points)
                        new_min = np.min(new_areas)
                        
                        if new_min > current_min + 1e-10:
                            improvement = new_min - current_min
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_move = (move_point, dx, dy)
        
        if best_move is not None:
            move_point, dx, dy = best_move
            points[move_point, 0] += dx
            points[move_point, 1] += dy
        
        # Also try small random perturbations
        for _ in range(10):
            if time.time() - start_time > time_limit - 5:
                break
            perturbation = np.random.randn(N, 2) * 0.01
            new_points = points + perturbation
            new_areas = _all_triangle_areas(new_points)
            if np.min(new_areas) > current_min + 1e-10:
                points = new_points
                current_min = np.min(new_areas)
    
    return points

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s margin for 120s limit
    
    # Start with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Normalize to unit area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        hull_area = 1.0
    
    if hull_area > 1e-12:
        points = points / np.sqrt(hull_area)
    
    # Optimize
    points = _optimize_from(points, time_limit)
    
    # Final check and return
    if time.time() - start_time > time_limit:
        # Return the initial regular polygon if we're over time
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            hull_area = 1.0
        if hull_area > 1e-12:
            points = points / np.sqrt(hull_area)
    
    return points
# EVOLVE-BLOCK-END
