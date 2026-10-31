"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

def _min_triangle_area(points):
    """Compute the minimum triangle area among all C(n,3) triples."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = np.inf
    for i, j, k in combinations(range(n), 3):
        # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
        d1 = points[j] - points[i]
        d2 = points[k] - points[i]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return float(min_area)

def _is_inside_triangle(p, eps=1e-9):
    """Check if point p=(x,y) is inside or on the boundary of the equilateral triangle."""
    x, y = p
    # Triangle: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # Conditions:
    # y >= 0
    # y <= sqrt(3)/2
    # y <= sqrt(3) * x  (left side)
    # y <= -sqrt(3) * (x - 1) = sqrt(3)*(1-x)  (right side)
    if y < -eps:
        return False
    if y > np.sqrt(3)/2 + eps:
        return False
    if y > np.sqrt(3) * x + eps:
        return False
    if y > np.sqrt(3) * (1 - x) + eps:
        return False
    return True

def _penalty(points):
    """Penalty for points outside the triangle."""
    pen = 0.0
    for p in points:
        if not _is_inside_triangle(p):
            # Distance to triangle boundary
            x, y = p
            # Compute distance to each edge
            # Edge AB: y=0, distance = max(0, -y)
            d_ab = max(0, -y)
            # Edge AC: y = sqrt(3)*x, or sqrt(3)*x - y = 0, distance = max(0, sqrt(3)*x - y)/2
            # Actually distance to line ax+by+c=0 is |ax+by+c|/sqrt(a^2+b^2)
            # Line AC: sqrt(3)*x - y = 0, so distance = |sqrt(3)*x - y| / sqrt(3+1) = |sqrt(3)*x - y|/2
            d_ac = max(0, np.sqrt(3)*x - y) / 2
            # Line BC: sqrt(3)*(1-x) - y = 0, distance = |sqrt(3)*(1-x) - y|/2
            d_bc = max(0, np.sqrt(3)*(1-x) - y) / 2
            pen += (d_ab + d_ac + d_bc) ** 2
    return pen

def _objective_smooth(flat):
    """Smooth objective: negative min area with penalty, using log-sum-exp for smoothness."""
    points = flat.reshape(N, 2)
    pen = _penalty(points)
    if pen > 1e-6:
        return 1e6 + pen
    min_area = _min_triangle_area(points)
    return -min_area

def _objective_hard(flat):
    """Hard objective: returns large value if any point is outside."""
    points = flat.reshape(N, 2)
    for p in points:
        if not _is_inside_triangle(p):
            return 1e6
    min_area = _min_triangle_area(points)
    return -min_area

def _clip_to_triangle(points):
    """Clip points to be inside the triangle."""
    clipped = points.copy()
    for i in range(len(points)):
        x, y = points[i]
        # Project to triangle
        # Simple approach: if outside, move towards centroid
        centroid = np.array([0.5, np.sqrt(3)/6])
        # Check if inside
        if not _is_inside_triangle(clipped[i]):
            # Move towards centroid
            t = 0.5
            new_p = (1-t)*clipped[i] + t*centroid
            clipped[i] = new_p
    return clipped

def _initial_grid():
    """Generate a structured initial configuration using triangular grid."""
    # Use a 3x3 grid in barycentric coordinates, plus center
    # Actually, let's use a more symmetric arrangement
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, np.sqrt(3)/2])
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, np.sqrt(3)/4])
    points.append([0.75, np.sqrt(3)/4])
    # Inner points - use a smaller triangle grid
    # Center
    points.append([0.5, np.sqrt(3)/6])
    # More inner points
    # Use barycentric coordinates for inner points
    # Let's place 4 more points symmetrically
    # Points at 1/3 and 2/3 along medians
    for frac in [1/3, 2/3]:
        # Along median from A to midpoint of BC
        mid_bc = [0.75, np.sqrt(3)/4]
        p = [frac*mid_bc[0], frac*mid_bc[1]]
        points.append(p)
        # Along median from B to midpoint of AC
        mid_ac = [0.25, np.sqrt(3)/4]
        p = [frac*mid_ac[0] + (1-frac)*1.0, frac*mid_ac[1]]
        points.append(p)
        # Along median from C to midpoint of AB
        mid_ab = [0.5, 0.0]
        p = [frac*mid_ab[0] + (1-frac)*0.5, frac*mid_ab[1] + (1-frac)*np.sqrt(3)/2]
        points.append(p)
    
    # We need exactly 11 points, we have 3+3+1+6=13, so remove 2
    # Let's just use the first 11
    points = points[:11]
    return np.array(points)

def _random_initial(seed):
    """Generate random initial configuration."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(N):
        while True:
            x = rng.uniform(0, 1)
            y = rng.uniform(0, np.sqrt(3)/2)
            if _is_inside_triangle([x, y]):
                points.append([x, y])
                break
    return np.array(points)

def _refine(points, max_iter=2000, timeout=None):
    """Refine points using Nelder-Mead."""
    start_time = time.time()
    def obj(flat):
        if timeout is not None and time.time() - start_time > timeout:
            return 1e6
        pts = flat.reshape(N, 2)
        pen = _penalty(pts)
        if pen > 1e-8:
            return 1e6 + pen
        return -_min_triangle_area(pts)
    
    flat = points.flatten()
    try:
        res = minimize(obj, flat, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        pts = res.x.reshape(N, 2)
        if _penalty(pts) < 1e-6:
            return pts, -res.fun
        else:
            return points, _min_triangle_area(points)
    except Exception:
        return points, _min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Differential Evolution for global search
    try:
        # Bounds for Cartesian coordinates
        bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
        
        de_result = differential_evolution(
            _objective_hard,
            bounds=bounds,
            seed=42,
            maxiter=300,
            popsize=20,
            tol=1e-8,
            polish=False,
            init='latinhypercube',
            updating='deferred'
        )
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            pts = de_result.x.reshape(N, 2)
            if _penalty(pts) < 1e-6:
                score = _min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
    except Exception:
        pass
    
    # Phase 2: Multi-start refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try several initial configurations
        configs = []
        # Structured grid
        configs.append(_initial_grid())
        # Random starts
        for seed in range(5):
            configs.append(_random_initial(seed + 100))
        
        for i, init_pts in enumerate(configs):
            if time.time() > deadline - 10:
                break
            remaining_time = deadline - time.time() - 10
            if remaining_time < 5:
                break
            refined, score = _refine(init_pts, max_iter=1000, timeout=remaining_time)
            if score > best_score:
                best_score = score
                best_points = refined
    
    # Phase 3: If still have time, do more DE with different seed
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
            de_result2 = differential_evolution(
                _objective_hard,
                bounds=bounds,
                seed=123,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='latinhypercube'
            )
            pts = de_result2.x.reshape(N, 2)
            if _penalty(pts) < 1e-6:
                score = _min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass
    
    # Final refinement of best
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < deadline - 5:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            refined, score = _refine(best_points, max_iter=500, timeout=remaining)
            if score > best_score:
                best_score = score
                best_points = refined
    
    if best_points is None:
        # Fallback
        best_points = _random_initial(0)
    
    return best_points
# EVOLVE-BLOCK-END
