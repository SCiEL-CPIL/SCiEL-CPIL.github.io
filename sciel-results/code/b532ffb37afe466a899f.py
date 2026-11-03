import numpy as np
from scipy.spatial import ConvexHull
import time
import random

N = 130

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _time_ok(start, deadline):
    return time.time() < deadline

def _covering_radius(points):
    """
    Compute covering radius in degrees using a high-resolution Fibonacci spiral.
    """
    M = 200000  # number of test points
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = test_pts @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)
    angles = np.arccos(min_dots)
    return np.max(angles) * 180.0 / np.pi

def _covering_radius_fast(points):
    """
    Faster estimate using a coarser grid.
    """
    M = 50000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = test_pts @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)
    angles = np.arccos(min_dots)
    return np.max(angles) * 180.0 / np.pi

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _worst_voronoi_vertex(points):
    """
    Find the point on the sphere farthest from all sites, using convex hull.
    Returns (vertex_direction, covering_radius, facet_indices)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, 0.0, None
    
    max_rad = -1
    worst_dir = None
    worst_facet = None
    
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Normal to the facet
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # Distance from origin to the plane: d = n.dot(p1)
        d = n.dot(p1)
        
        # The Voronoi vertex is at distance 1/d from origin along n (if d > 0)
        # Covering radius = arccos(1 / (1/d)) = arccos(d) ... wait
        # Actually, the circumcenter C of the triangle on the unit sphere:
        # C is along n. The angular distance theta satisfies cos(theta) = 1/|C|.
        # The plane equation: n.x = d. The closest point to origin on the plane is d*n.
        # But the circumcenter is where the perpendiculars meet.
        # For points on unit sphere, the circumcenter C satisfies C = k*n.
        # |C - p1|^2 = |C|^2 - 2 C.dot(p1) + 1
        # Since C is equidistant: C.dot(p1) = C.dot(p2) = C.dot(p3) = k*d
        # |C - p1|^2 = k^2 - 2kd + 1
        # The circumcenter is the center of the sphere through p1,p2,p3.
        # Actually, the Voronoi vertex direction is n, and the angular radius is arccos(d).
        # Because the cap of radius theta centered at n contains p1,p2,p3 iff cos(theta) >= d.
        # So theta = arccos(d) is the angular radius of the smallest cap centered at n containing the triangle.
        # This is the covering radius for this facet.
        
        if d > -1e-10:  # facet visible from origin
            d_clipped = min(1.0, max(-1.0, d))
            theta = np.arccos(d_clipped)
            if theta > max_rad:
                max_rad = theta
                worst_dir = n.copy()
                worst_facet = simplex
    
    if worst_dir is None:
        return None, 0.0, None
    
    return worst_dir, max_rad * 180.0 / np.pi, worst_facet

def _lloyd_step(points, start, deadline, max_iter=500, tol=1e-8):
    """
    Lloyd's algorithm for spherical k-means:
    1. Compute Voronoi cells (via convex hull facets)
    2. Move each point to the centroid of its Voronoi cell
    """
    points = points.copy()
    
    for it in range(max_iter):
        if not _time_ok(start, deadline):
            break
        
        # Compute Voronoi cells via convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find its Voronoi cell (the facet(s) opposite to it)
        # Actually, in the convex hull, each point is a vertex. The Voronoi cell of point i
        # is the dual of the set of facets containing i.
        # The centroid of the Voronoi cell on the sphere is the direction of the sum of 
        # the facet normals of facets adjacent to vertex i.
        
        # Build vertex-to-facet mapping
        vertex_facets = [[] for _ in range(N)]
        for facet in hull.simplices:
            for v in facet:
                vertex_facets[v].append(facet)
        
        # For each vertex, compute the centroid of its Voronoi cell
        # The Voronoi cell is a spherical polygon. Its centroid direction is approximately
        # the sum of the outward normals of the adjacent facets.
        
        new_points = np.zeros_like(points)
        for i in range(N):
            if not vertex_facets[i]:
                new_points[i] = points[i]
                continue
            
            # Sum the normals of adjacent facets
            normal_sum = np.zeros(3)
            for facet in vertex_facets[i]:
                p1 = points[facet[0]]
                p2 = points[facet[1]]
                p3 = points[facet[2]]
                n = np.cross(p2 - p1, p3 - p1)
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-12:
                    n = n / n_norm
                    # Ensure normal points outward (away from origin)
                    if n.dot(points[i]) < 0:
                        n = -n
                    normal_sum += n
            
            ns_norm = np.linalg.norm(normal_sum)
            if ns_norm > 1e-10:
                new_points[i] = normal_sum / ns_norm
            else:
                new_points[i] = points[i]
        
        new_points = _normalize(new_points)
        
        # Check convergence
        move = np.linalg.norm(new_points - points)
        points = new_points
        
        if move < tol:
            break
    
    return points

def _smooth_max_gradient(points, temperature=0.1):
    """
    Gradient of the smooth maximum of angular distances.
    f(x) = (1/T) * log(sum_i exp(T * d_i))
    where d_i is the angular distance from x to point i.
    The gradient points in the direction that increases the worst distance.
    We want to minimize the worst distance, so we move opposite to this gradient.
    """
    # This is expensive, so we use it sparingly.
    # For each point, compute the gradient of the max distance from that point to all others.
    grad = np.zeros_like(points)
    T = 1.0 / temperature  # larger T = sharper max
    
    for i in range(N):
        # Angular distances from point i to all points
        dots = points @ points[i]
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # Smooth max weights
        max_d = np.max(dists)
        weights = np.exp(T * (dists - max_d))
        weights_sum = np.sum(weights)
        if weights_sum < 1e-10:
            continue
        weights /= weights_sum
        
        # Gradient of arccos(d) w.r.t. x_i where d = x_i . x_j
        # d/dx_i arccos(x_i . x_j) = -x_j / sqrt(1 - (x_i.x_j)^2)
        # But we want the gradient w.r.t. x_i of the function that measures covering.
        # Actually, the covering radius is max over y of min_i d(y, p_i).
        # The subgradient of min_i d(y, p_i) w.r.t. p_i is complex.
        
        # Alternative: use the fact that moving p_i in the direction of the nearest
        # "hole" reduces the local covering radius.
        # The hole is the direction that maximizes distance to all points.
        
        # For now, use a simple repulsion-based gradient.
        # The potential U = sum_{i<j} 1/d_ij^p encourages uniformity.
        for j in range(N):
            if i == j:
                continue
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-10:
                continue
            # Gradient of 1/d^2 = -2 * diff / d^4
            grad[i] += -2.0 * diff / (d2 * d2)
    
    return grad

def _local_refine(points, start, deadline, n_iterations=100):
    """
    Local refinement: for each point, try small perturbations to reduce local covering radius.
    """
    points = points.copy()
    current_score = _covering_radius_fast(points)
    
    for it in range(n_iterations):
        if not _time_ok(start, deadline):
            break
        
        improved = False
        for i in range(N):
            if not _time_ok(start, deadline):
                break
            
            # Try moving point i in a random direction
            n_trials = 5
            best_move = None
            best_score = current_score
            
            for trial in range(n_trials):
                noise = np.random.randn(3)
                noise = _normalize(noise)
                
                for step_size in [0.01, 0.005, 0.02]:
                    new_points = points.copy()
                    new_points[i] = points[i] + step_size * noise
                    new_points[i] = _normalize(new_points[i])
                    
                    score = _covering_radius_fast(new_points)
                    if score < best_score:
                        best_score = score
                        best_move = new_points.copy()
            
            if best_move is not None:
                points = best_move
                current_score = best_score
                improved = True
        
        if not improved:
            break
    
    return points

def _simulated_annealing(points, start, deadline, n_iterations=200):
    """
    Simulated annealing: accept worse moves with decreasing probability.
    """
    points = points.copy()
    current_score = _covering_radius_fast(points)
    best_points = points.copy()
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    cooling = (T_min / T) ** (1.0 / max(n_iterations, 1))
    
    for it in range(n_iterations):
        if not _time_ok(start, deadline):
            break
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Random perturbation
        idx = np.random.randint(N)
        noise = np.random.randn(3)
        noise = _normalize(noise)
        
        step_size = 0.02 * T / T_min
        
        new_points = points.copy()
        new_points[idx] = points[idx] + step_size * noise
        new_points[idx] = _normalize(new_points[idx])
        
        new_score = _covering_radius_fast(new_points)
        
        # Metropolis criterion
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            points = new_points
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_points = points.copy()
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    random.seed(42)
    
    best_points = _fibonacci_spiral()
    best_score = _covering_radius(best_points)
    
    # Phase 1: Multi-restart Lloyd's algorithm
    n_restarts = 30
    for restart in range(n_restarts):
        if not _time_ok(start_time, deadline):
            break
        
        # Initial configuration: Fibonacci spiral with random perturbation
        if restart == 0:
            points = _fibonacci_spiral()
        else:
            points = _fibonacci_spiral()
            noise = np.random.randn(N, 3) * (0.01 * restart)
            points = points + noise
            points = _normalize(points)
        
        # Lloyd's algorithm
        points = _lloyd_step(points, start_time, deadline, max_iter=100)
        
        # Local refinement
        points = _local_refine(points, start_time, deadline, n_iterations=20)
        
        # Evaluate
        score = _covering_radius_fast(points)
        if score < best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 2: Simulated annealing on best solution
    if _time_ok(start_time, deadline):
        sa_points = _simulated_annealing(best_points, start_time, deadline, n_iterations=150)
        sa_score = _covering_radius_fast(sa_points)
        if sa_score < best_score:
            best_score = sa_score
            best_points = sa_points
    
    # Phase 3: Final local refinement
    if _time_ok(start_time, deadline):
        refined = _local_refine(best_points, start_time, deadline, n_iterations=50)
        refined_score = _covering_radius_fast(refined)
        if refined_score < best_score:
            best_score = refined_score
            best_points = refined
    
    # Final Lloyd refinement
    if _time_ok(start_time, deadline):
        lloyd_final = _lloyd_step(best_points, start_time, deadline, max_iter=200)
        lloyd_score = _covering_radius_fast(lloyd_final)
        if lloyd_score < best_score:
            best_score = lloyd_score
            best_points = lloyd_final
    
    # Final evaluation with high precision
    final_score = _covering_radius(best_points)
    
    return best_points
