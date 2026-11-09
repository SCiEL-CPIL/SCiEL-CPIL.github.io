# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 570.0  # 60s safety margin

N_POINTS = 13
# Precompute triangle indices for n=13
TRIANGLES = np.array(list(itertools.combinations(range(N_POINTS), 3)))

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
    p1 = points[TRIANGLES[:, 0]]
    p2 = points[TRIANGLES[:, 1]]
    p3 = points[TRIANGLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """
    Calculate the ratio A_min / A_hull.
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull < 1e-10:
        return 0.0
        
    return a_min / a_hull

def _soft_min_triangle_area(points, beta=50.0):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    This makes the objective differentiable for gradient-based optimization.
    """
    p1 = points[TRIANGLES[:, 0]]
    p2 = points[TRIANGLES[:, 1]]
    p3 = points[TRIANGLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp for soft minimum: -log(sum(exp(-beta * areas))) / beta
    # This approximates min(areas) as beta -> infinity
    max_a = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) + beta * max_a
    soft_min = -log_sum / beta
    
    return soft_min

def _soft_score(points, beta=50.0):
    """
    Calculate a smooth approximation of the score ratio.
    """
    a_min_soft = _soft_min_triangle_area(points, beta)
    a_hull = _hull_area(points)
    
    if a_hull < 1e-10:
        return 0.0
        
    return a_min_soft / a_hull

def _objective_soft(params, beta=50.0):
    """
    Objective function for free-form optimization with soft-min surrogate.
    """
    points = params.reshape(N_POINTS, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    
    # Check for coincident points
    min_dist = np.inf
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-8:
        return 1e10 - min_dist
        
    s = _soft_score(points, beta)
    return -s

def _objective_exact(params):
    """
    Objective function for free-form optimization with exact minimum.
    """
    points = params.reshape(N_POINTS, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    
    # Check for coincident points
    min_dist = np.inf
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-8:
        return 1e10 - min_dist
        
    s = _score(points)
    return -s

def _lloyd_relax(points, num_iterations=50, hull_indices=None):
    """
    Apply Lloyd's algorithm (Voronoi centroid iteration) to interior points
    while keeping hull points fixed. This helps enforce uniform point density.
    """
    points = points.copy()
    
    # Identify hull points if not provided
    if hull_indices is None:
        try:
            hull = ConvexHull(points)
            hull_indices = set(hull.vertices)
        except Exception:
            return points
    
    interior_indices = [i for i in range(len(points)) if i not in hull_indices]
    
    for _ in range(num_iterations):
        if time.time() > _DEADLINE - 30:
            break
            
        try:
            # Compute Delaunay triangulation
            tri = Delaunay(points)
            
            # For each interior point, compute its Voronoi cell centroid
            # Approximate by averaging centroids of adjacent Delaunay triangles
            new_interior = points[interior_indices].copy()
            
            for idx, point_idx in enumerate(interior_indices):
                # Find triangles containing this point
                adjacent_triangles = []
                for t_idx, triangle in enumerate(tri.simplices):
                    if point_idx in triangle:
                        # Compute centroid of this triangle
                        centroid = np.mean(points[triangle], axis=0)
                        adjacent_triangles.append(centroid)
                
                if len(adjacent_triangles) > 0:
                    # Weighted average of adjacent triangle centroids
                    # (simple average as approximation of Voronoi centroid)
                    new_centroid = np.mean(adjacent_triangles, axis=0)
                    new_interior[idx] = new_centroid
            
            # Update interior points
            for idx, point_idx in enumerate(interior_indices):
                points[point_idx] = new_interior[idx]
                
        except Exception:
            # If Delaunay fails, break out of relaxation
            break
    
    return points

def _optimize_with_soft_min(x0, beta_schedule=None, maxiter=1000, maxfev=50000):
    """
    Optimize using soft-min surrogate with L-BFGS-B, then polish with exact objective.
    """
    if beta_schedule is None:
        beta_schedule = [10.0, 30.0, 100.0, 300.0]
    
    best_points = x0.reshape(N_POINTS, 2)
    best_score = _score(best_points)
    
    current_params = x0.copy()
    
    for beta in beta_schedule:
        if time.time() > _DEADLINE - 60:
            break
        
        try:
            result = minimize(
                _objective_soft,
                current_params,
                args=(beta,),
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'maxfev': maxfev,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if result.fun < 1e9:
                candidate_points = result.x.reshape(N_POINTS, 2)
                candidate_score = _score(candidate_points)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points
                    current_params = result.x.copy()
                    
        except Exception:
            pass
    
    # Final polish with exact objective using Powell
    if time.time() > _DEADLINE - 30:
        return best_points, best_score
    
    try:
        result_polish = minimize(
            _objective_exact,
            best_points.flatten(),
            method='Powell',
            options={
                'maxiter': 500,
                'maxfev': 20000,
                'ftol': 1e-12,
                'xtol': 1e-12
            }
        )
        
        if result_polish.fun < 1e9:
            candidate_points = result_polish.x.reshape(N_POINTS, 2)
            candidate_score = _score(candidate_points)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
                
    except Exception:
        pass
    
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    """
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # --- Strategy 1: Start from regular polygon and optimize with soft-min ---
    angles_reg = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    points_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    x0_reg = points_reg.flatten()
    
    if time.time() < _DEADLINE - 120:
        pts, score = _optimize_with_soft_min(x0_reg)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # --- Strategy 2: Regular polygon with center + 12 outer ---
    # 1 center + 12 outer points
    points_center_outer = np.zeros((N_POINTS, 2))
    angles_outer = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points_center_outer[0] = [0.0, 0.0]
    points_center_outer[1:] = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    x0_co = points_center_outer.flatten()
    
    if time.time() < _DEADLINE - 100:
        pts, score = _optimize_with_soft_min(x0_co)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # --- Strategy 3: Concentric rings with Lloyd relaxation ---
    # Try a few concentric ring configurations
    topologies = [
        [1, 6, 6],   # 1 center, 6 inner, 6 outer
        [1, 5, 7],   # 1 center, 5 inner, 7 outer
        [1, 4, 8],   # 1 center, 4 inner, 8 outer
    ]
    
    for topology in topologies:
        if time.time() > _DEADLINE - 80:
            break
        
        # Construct initial points
        points_init = np.zeros((N_POINTS, 2))
        idx = 0
        num_rings = len(topology)
        base_radii = np.sqrt(np.linspace(0.2, 1.0, num_rings))
        
        for i, count in enumerate(topology):
            r = base_radii[i] if i > 0 else 0.0
            thetas = np.linspace(0, 2 * np.pi, count, endpoint=False) + rng.uniform(-0.05, 0.05, count)
            
            for theta in thetas:
                if r < 1e-9:
                    points_init[idx] = [0.0, 0.0]
                else:
                    points_init[idx] = [r * np.cos(theta), r * np.sin(theta)]
                idx += 1
        
        # Apply Lloyd relaxation to interior points
        points_lloyd = _lloyd_relax(points_init, num_iterations=30)
        x0_lloyd = points_lloyd.flatten()
        
        if time.time() < _DEADLINE - 60:
            pts, score = _optimize_with_soft_min(x0_lloyd, maxiter=500, maxfev=20000)
            if score > best_score:
                best_score = score
                best_points = pts
    
    # --- Strategy 4: Random restarts with soft-min optimization ---
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > _DEADLINE - 40:
            break
        
        # Random points in unit disk
        theta = rng.uniform(0, 2 * np.pi, N_POINTS)
        r = rng.uniform(0, 1, N_POINTS)
        pts_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        x0_rand = pts_rand.flatten()
        
        try:
            pts, score = _optimize_with_soft_min(x0_rand, beta_schedule=[20.0, 80.0], maxiter=300, maxfev=10000)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass
    
    # Fallback
    if best_points is None or best_points.shape != (N_POINTS, 2):
        best_points = points_reg
        
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = points_reg
        
    return best_points

# EVOLVE-BLOCK-END
