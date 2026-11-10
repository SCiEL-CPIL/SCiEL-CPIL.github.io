# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize


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


def _circumcenter(p1, p2, p3):
    """Compute circumcenter of triangle p1, p2, p3."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    
    d = 2 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(d) < 1e-12:
        return None
    
    ux = ((ax**2 + ay**2)*(by - cy) + (bx**2 + by**2)*(cy - ay) + (cx**2 + cy**2)*(ay - by)) / d
    uy = ((ax**2 + ay**2)*(cx - bx) + (bx**2 + by**2)*(ax - cx) + (cx**2 + cy**2)*(bx - ax)) / d
    
    return np.array([ux, uy])


def _circumradius(p1, p2, p3):
    """Compute circumradius of triangle p1, p2, p3."""
    cc = _circumcenter(p1, p2, p3)
    if cc is None:
        return 0.0
    return np.linalg.norm(cc - p1)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the minimum
    triangle area to the convex hull area.
    
    Uses physics-based relaxation with Delaunay triangulation to guide point
    movement, maximizing the minimum circumradius of Delaunay triangles.
    """
    n = 13
    
    # Start with a reasonable initial configuration: regular 13-gon with center
    # This gives a good starting point for optimization
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Add some randomization to break symmetry
    rng = np.random.default_rng(seed=12345)
    points = points + 0.05 * rng.random((n, 2))
    
    # Normalize to unit area hull
    hull_area = _hull_area(points)
    if hull_area > 0:
        points = points / np.sqrt(hull_area)
    
    best_points = points.copy()
    best_score = _score(points)
    
    # Physics-based relaxation using Delaunay triangulation
    # We use a simulated annealing approach with local moves guided by
    # the Delaunay structure
    
    def _delaunay_adjacency(points):
        """Get adjacency from Delaunay triangulation."""
        try:
            tri = Delaunay(points)
            adj = {i: set() for i in range(len(points))}
            for simplex in tri.simplices:
                for i in simplex:
                    for j in simplex:
                        if i != j:
                            adj[i].add(j)
            return adj
        except Exception:
            return None
    
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
            # Inverse square law repulsion
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
            # Hooke's law
            force += k * (dist - rest_length) * diff / dist
        return force
    
    # Iterative optimization
    np.random.seed(42)
    rng_opt = np.random.default_rng(seed=999)
    
    current_points = points.copy()
    current_score = best_score
    
    # Simulated annealing parameters
    T = 0.1
    T_min = 0.001
    cooling_rate = 0.995
    max_iterations = 5000
    
    for it in range(max_iterations):
        # Choose a random point to move
        i = rng_opt.integers(0, n)
        
        # Compute forces
        adj = _delaunay_adjacency(current_points)
        if adj is None:
            adj = {j: set() for j in range(n)}
        
        rep_force = _repulsion_force(current_points, i, strength=0.5)
        spring_force = _spring_force(current_points, i, adj, rest_length=0.4, k=0.3)
        
        # Combined force
        total_force = rep_force + spring_force
        
        # Add random perturbation
        noise = 0.01 * T * rng_opt.standard_normal(2)
        
        # Proposed move
        step_size = 0.05 * T
        new_points = current_points.copy()
        new_points[i] = current_points[i] + step_size * (total_force / (np.linalg.norm(total_force) + 1e-10) + noise)
        
        # Evaluate
        new_score = _score(new_points)
        
        if new_score >= current_score or rng_opt.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # Early stopping if converged
        if T <= T_min and it > 1000:
            break
    
    # Final local optimization using scipy
    def _neg_score(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = best_points.flatten()
    result = minimize(_neg_score, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if result.fun < -best_score:
        best_points = result.x.reshape(-1, 2)
        best_score = -result.fun
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points


# EVOLVE-BLOCK-END
