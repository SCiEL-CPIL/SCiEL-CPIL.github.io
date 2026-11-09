# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

def _triangle_areas(points):
    """Compute all triangle areas for 13 points."""
    n = len(points)
    areas = np.zeros(len(list(combinations(range(n), 3))))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                areas[idx] = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                idx += 1
    return areas

def _score(points):
    """Compute A_min / A_hull."""
    try:
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _soft_min(areas, temperature):
    """Smooth approximation of the minimum using log-sum-exp."""
    # Use negative log-sum-exp for soft min
    scaled = -areas / temperature
    max_s = np.max(scaled)
    if max_s == 0:
        return -temperature * np.log(np.mean(np.exp(scaled)))
    return -temperature * (max_s + np.log(np.mean(np.exp(scaled - max_s))))

def _optimize_from_initial(points_init, deadline, max_iter=500):
    """Optimize from an initial configuration using L-BFGS-B with soft-min."""
    n = len(points_init)
    x0 = points_init.flatten()
    
    def objective(x):
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        areas = _triangle_areas(pts)
        # Soft min with decreasing temperature
        temp = 0.01
        soft_min_val = _soft_min(areas, temp)
        # Also penalize if any area is too small
        a_min = np.min(areas)
        if a_min < 1e-6:
            return 1e10
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
        except Exception:
            return 1e10
        if a_hull < 1e-6:
            return 1e10
        # Maximize a_min / a_hull, so minimize negative
        return -(a_min / a_hull) + 10.0 * soft_min_val / a_hull
    
    def gradient(x):
        pts = x.reshape(n, 2)
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = objective(x_plus)
            x_minus = x.copy()
            x_minus[i] -= eps
            f_minus = objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad
    
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    
    best_pts = result.x.reshape(n, 2)
    best_score = _score(best_pts)
    
    # Coordinate descent polishing
    for coord_round in range(3):
        for i in range(n):
            for dim in range(2):
                best_val = best_score
                best_coord = best_pts[i, dim]
                # Try small perturbations
                for delta in np.linspace(-0.05, 0.05, 21):
                    candidate = best_pts.copy()
                    candidate[i, dim] = best_coord + delta
                    s = _score(candidate)
                    if s > best_val:
                        best_val = s
                        best_coord = candidate[i, dim]
                best_pts[i, dim] = best_coord
            best_score = best_pts[i, dim]  # placeholder, recompute
        best_score = _score(best_pts)
    
    return best_pts, best_score

def _hexagonal_lattice_13():
    """Create a hexagonal lattice arrangement for 13 points."""
    points = []
    # Center
    points.append([0.0, 0.0])
    # Ring 1: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    # Ring 2: 6 points
    for i in range(6):
        angle = (i + 0.5) * np.pi / 3
        points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    return np.array(points)

def _concentric_rings_13():
    """Create concentric ring arrangement for 13 points."""
    points = []
    # Center
    points.append([0.0, 0.0])
    # Ring 1: 4 points (square)
    for i in range(4):
        angle = i * np.pi / 2 + np.pi/4
        points.append([0.4 * np.cos(angle), 0.4 * np.sin(angle)])
    # Ring 2: 8 points
    for i in range(8):
        angle = i * np.pi / 4
        points.append([0.8 * np.cos(angle), 0.8 * np.sin(angle)])
    return np.array(points)

def _spiral_13():
    """Create a spiral arrangement for 13 points."""
    points = []
    n = 13
    for i in range(n):
        t = i / (n - 1)
        r = 0.1 + 0.9 * t
        angle = t * 4 * np.pi
        points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)

def _random_13(seed):
    """Create random arrangement."""
    rng = np.random.default_rng(seed)
    return rng.random((13, 2))

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    best_points = None
    best_score = -1.0
    
    # Generate diverse initial configurations
    initializations = [
        ("hex", _hexagonal_lattice_13()),
        ("rings", _concentric_rings_13()),
        ("spiral", _spiral_13()),
        ("rand1", _random_13(42)),
        ("rand2", _random_13(123)),
        ("rand3", _random_13(456)),
    ]
    
    # Evaluate initial configurations
    for name, init_pts in initializations:
        s = _score(init_pts)
        if s > best_score:
            best_score = s
            best_points = init_pts.copy()
    
    # Optimize from each initialization
    for name, init_pts in initializations:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Safety margin
            break
        
        try:
            opt_pts, opt_score = _optimize_from_initial(init_pts, deadline, max_iter=200)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            continue
    
    # Final check
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback to hexagonal lattice
        best_points = _hexagonal_lattice_13()
    
    # Ensure valid shape
    if best_points.shape != (13, 2):
        best_points = _hexagonal_lattice_13()
    
    return best_points

# EVOLVE-BLOCK-END
