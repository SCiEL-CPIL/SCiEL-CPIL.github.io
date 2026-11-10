# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
from itertools import combinations


def _triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by three points."""
    return abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])) / 2.0


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(points):
    """Compute the area of the convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the Heilbronn ratio: A_min / A_hull."""
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _delaunay_triangles(points):
    """Get Delaunay triangulation and return list of triangles."""
    try:
        tri = Delaunay(points)
        return tri.simplices
    except Exception:
        return None


def _circumradius(p1, p2, p3):
    """Compute circumradius of triangle p1, p2, p3."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    
    d = 2 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(d) < 1e-12:
        return 0.0
    
    ux = ((ax**2 + ay**2)*(by - cy) + (bx**2 + by**2)*(cy - ay) + (cx**2 + cy**2)*(ay - by)) / d
    uy = ((ax**2 + ay**2)*(cx - bx) + (bx**2 + by**2)*(ax - cx) + (cx**2 + cy**2)*(bx - ax)) / d
    
    return np.sqrt((ux - ax)**2 + (uy - ay)**2)


def _min_circumradius(points):
    """Compute the minimum circumradius among all Delaunay triangles."""
    triangles = _delaunay_triangles(points)
    if triangles is None:
        return 0.0
    
    min_r = np.inf
    for simplex in triangles:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        r = _circumradius(p1, p2, p3)
        if r < min_r:
            min_r = r
    return min_r


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the minimum
    triangle area to the convex hull area.
    
    Uses a constrained Delaunay triangulation approach where internal vertices
    are moved via Laplacian smoothing with repulsive forces to maximize the
    minimum circumradius of adjacent triangles, enforcing the farthest-point
    Voronoi property inherent to optimal Heilbronn configurations.
    """
    n = 13
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(pts)
    
    # Config 2: 7-gon with center + 5 inner ring
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    center = np.array([[0.0, 0.0]])
    inner_angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    inner = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
    pts = np.vstack([outer, center, inner])
    initial_configs.append(pts)
    
    # Config 3: Hexagonal lattice subset
    pts = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [2, 0], [2, 1], [0, 2], [1, 2],
        [2, 2], [3, 0], [3, 1], [3, 2], [0, 3]
    ], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts / np.std(pts)
    initial_configs.append(pts)
    
    # Config 4: Random well-spread points
    rng_init = np.random.default_rng(seed=123)
    pts = rng_init.random((n, 2)) * 2 - 1
    initial_configs.append(pts)
    
    for init_pts in initial_configs:
        # Normalize to unit area hull
        hull_area = _hull_area(init_pts)
        if hull_area <= 0:
            continue
        pts = init_pts / np.sqrt(hull_area)
        
        current_points = pts.copy()
        current_score = _score(pts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = pts.copy()
        
        def _get_hull_indices(points):
            """Get indices of points on the convex hull."""
            try:
                hull = ConvexHull(points)
                return set(hull.vertices)
            except Exception:
                return set(range(len(points)))
        
        def _laplacian_smooth(points, i, adj, weight=0.3):
            """Compute Laplacian smoothing displacement for point i."""
            pi = points[i]
            neighbors = list(adj.get(i, []))
            if not neighbors:
                return np.zeros(2)
            
            centroid = np.mean(points[neighbors], axis=0)
            displacement = centroid - pi
            return weight * displacement
        
        def _repulsion_force(points, i, strength=1.0):
            """Compute repulsive force on point i from all other points."""
            force = np.zeros(2)
            pi = points[i]
            for j in range(len(points)):
                if i == j:
                    continue
                diff = pi - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    dist_sq = 1e-12
                dist = np.sqrt(dist_sq)
                force += strength * diff / (dist_sq * dist)
            return force
        
        def _spring_force(points, i, adj, rest_length=0.5, k=1.0):
            """Compute spring forces from Delaunay neighbors."""
            force = np.zeros(2)
            pi = points[i]
            for j in adj.get(i, []):
                diff = points[j] - pi
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue
                force += k * (dist - rest_length) * diff / dist
            return force
        
        def _identify_worst_triangles(points, num_worst=5):
            """Identify the triangles with smallest circumradii."""
            triangles = _delaunay_triangles(points)
            if triangles is None:
                return []
            
            tri_data = []
            for simplex in triangles:
                p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                r = _circumradius(p1, p2, p3)
                tri_data.append((r, simplex))
            
            tri_data.sort(key=lambda x: x[0])
            return [t[1] for t in tri_data[:num_worst]]
        
        # Iterative optimization with physics-based relaxation
        rng_opt = np.random.default_rng(seed=42)
        
        T = 0.15
        T_min = 0.0003
        cooling_rate = 0.998
        max_iterations = 12000
        
        no_improve_count = 0
        max_no_improve = 800
        
        for it in range(max_iterations):
            triangles = _delaunay_triangles(current_points)
            if triangles is None:
                break
            
            adj = {i: set() for i in range(n)}
            for simplex in triangles:
                for i in simplex:
                    for j in simplex:
                        if i != j:
                            adj[i].add(j)
            
            hull_idx = _get_hull_indices(current_points)
            
            worst_tris = _identify_worst_triangles(current_points, num_worst=5)
            
            candidates = []
            for tri in worst_tris:
                for idx in tri:
                    if idx not in hull_idx:
                        candidates.append(idx)
            
            internal_points = [i for i in range(n) if i not in hull_idx]
            if candidates:
                i = rng_opt.choice(candidates)
            elif internal_points:
                i = rng_opt.choice(internal_points)
            else:
                i = rng_opt.integers(0, n)
            
            lap_force = _laplacian_smooth(current_points, i, adj, weight=0.25)
            rep_force = _repulsion_force(current_points, i, strength=0.4)
            spring_force = _spring_force(current_points, i, adj, rest_length=0.45, k=0.25)
            
            total_force = lap_force + rep_force + spring_force
            
            noise = 0.03 * T * rng_opt.standard_normal(2)
            
            step_size = 0.1 * T
            new_points = current_points.copy()
            new_points[i] = current_points[i] + step_size * (total_force / (np.linalg.norm(total_force) + 1e-10) + noise)
            
            if np.linalg.norm(new_points[i]) > 5.0:
                new_points[i] = current_points[i]
            
            new_score = _score(new_points)
            
            if new_score >= current_score or rng_opt.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            T = max(T * cooling_rate, T_min)
            
            if no_improve_count > max_no_improve and T <= T_min * 10:
                break
        
        def _neg_score(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
        
        x0 = current_points.flatten()
        result = minimize(_neg_score, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < -current_score:
            final_pts = result.x.reshape(-1, 2)
            final_score = -result.fun
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts
    
    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points


# EVOLVE-BLOCK-END
