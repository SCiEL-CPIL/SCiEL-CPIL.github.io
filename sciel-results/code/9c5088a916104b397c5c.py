import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0
_SQRT3 = np.sqrt(3)

# Precompute triangle indices for vectorization
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

# Precompute pair indices for distance objective
_pair_combos = list(combinations(range(N), 2))
_pidx_i = np.array([c[0] for c in _pair_combos])
_pidx_j = np.array([c[1] for c in _pair_combos])


def _project_to_triangle(pts):
    """
    Project points to stay inside the equilateral triangle 
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    Uses iterative orthogonal projection onto the three half-planes.
    """
    pts = np.array(pts, dtype=float).copy()
    n = len(pts)
    
    for _ in range(20):
        # Constraint 1: y >= 0
        mask1 = pts[:, 1] < 0
        if np.any(mask1):
            pts[mask1, 1] = 0
        
        # Constraint 2: y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
        # Line: y = sqrt(3)*x, normal pointing inward: (-sqrt(3), 1) normalized
        # Distance to line: (sqrt(3)*x - y) / sqrt(3+1) = (sqrt(3)*x - y)/2
        # If violated (sqrt(3)*x - y < 0), project onto line
        d2 = _SQRT3 * pts[:, 0] - pts[:, 1]
        mask2 = d2 < 0
        if np.any(mask2):
            # Project onto line y = sqrt(3)*x
            # Point on line closest to (x,y): use projection formula
            # For line ax + by + c = 0, projection of (x0,y0) is:
            # (x0 - a*(ax0+by0+c)/(a^2+b^2), y0 - b*(ax0+by0+c)/(a^2+b^2))
            # Line: sqrt(3)*x - y = 0, so a=sqrt(3), b=-1, c=0
            # a^2 + b^2 = 3 + 1 = 4
            val = _SQRT3 * pts[mask2, 0] - pts[mask2, 1]
            pts[mask2, 0] -= _SQRT3 * val / 4.0
            pts[mask2, 1] -= (-1.0) * val / 4.0
        
        # Constraint 3: sqrt(3)*x + y <= sqrt(3)  =>  sqrt(3) - sqrt(3)*x - y >= 0
        # Line: sqrt(3)*x + y = sqrt(3), normal pointing inward: (-sqrt(3), -1)
        # Distance: (sqrt(3) - sqrt(3)*x - y) / 2
        d3 = _SQRT3 - _SQRT3 * pts[:, 0] - pts[:, 1]
        mask3 = d3 < 0
        if np.any(mask3):
            # Project onto line sqrt(3)*x + y = sqrt(3)
            # a=sqrt(3), b=1, c=-sqrt(3)
            # a^2 + b^2 = 4
            val = _SQRT3 * pts[mask3, 0] + pts[mask3, 1] - _SQRT3
            pts[mask3, 0] -= _SQRT3 * val / 4.0
            pts[mask3, 1] -= 1.0 * val / 4.0
        
        # Also enforce x >= 0 and x <= 1 as additional constraints
        pts[:, 0] = np.clip(pts[:, 0], 0, 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, _H)
    
    return pts


def _min_triangle_area(points):
    """Compute minimum area of all C(11,3) triangles."""
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)


def _min_dist_sq(points):
    """Compute minimum squared distance between any pair."""
    pi = points[_pidx_i]
    pj = points[_pidx_j]
    diff = pi - pj
    dists = np.sum(diff * diff, axis=1)
    return np.min(dists)


def _coordinate_descent(points, max_iter=1000, tol=1e-10, time_limit=None, start_time=None):
    """
    Coordinate descent: optimize one point at a time to maximize min triangle area.
    For each point, try to move it in small steps in 2D to improve the objective.
    """
    if time_limit is None:
        time_limit = 5.0
    if start_time is None:
        start_time = time.time()
    
    points = np.array(points, dtype=float).copy()
    n = len(points)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            best_score = _min_triangle_area(points)
            best_pos = points[i].copy()
            
            # Try moving point i in small steps
            step = 0.05
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    break
                
                # Try 8 directions
                for angle in range(8):
                    theta = angle * np.pi / 4
                    dx = step * np.cos(theta)
                    dy = step * np.sin(theta)
                    
                    new_pos = best_pos + np.array([dx, dy])
                    if new_pos[0] < -0.1 or new_pos[0] > 1.1 or new_pos[1] < -0.1 or new_pos[1] > _H + 0.1:
                        continue
                    
                    points[i] = new_pos
                    projected = _project_to_triangle(points)
                    score = _min_triangle_area(projected)
                    
                    if score > best_score + tol:
                        best_score = score
                        best_pos = projected[i].copy()
                        improved = True
                        # Reset step for next iteration
                        step = 0.05
                        break
                else:
                    # No improvement in this direction set, reduce step
                    step *= 0.7
                    if step < 1e-6:
                        break
            
            points[i] = best_pos
        
        if not improved:
            break
    
    return _project_to_triangle(points)


def _simulated_annealing(points, time_limit=30.0, start_time=None):
    """
    Simulated annealing to escape local optima.
    """
    if start_time is None:
        start_time = time.time()
    
    points = np.array(points, dtype=float).copy()
    n = len(points)
    rng = np.random.default_rng(42)
    
    best_points = points.copy()
    best_score = _min_triangle_area(points)
    
    temperature = 0.1
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        # Pick a random point to perturb
        i = rng.integers(0, n)
        
        # Generate random perturbation
        scale = temperature * 0.1
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        
        new_points = points.copy()
        new_points[i] += np.array([dx, dy])
        new_points = _project_to_triangle(new_points)
        
        new_score = _min_triangle_area(new_points)
        delta = new_score - best_score
        
        # Accept if better, or with probability exp(delta/temperature)
        if delta > 0 or rng.random() < np.exp(delta / temperature):
            points = new_points
        
        # Update best
        current_score = _min_triangle_area(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        temperature *= cooling
        if temperature < 1e-6:
            break
    
    return best_points


def _generate_symmetric_init():
    """
    Generate a symmetric initialization based on hexagonal packing ideas.
    """
    points = []
    
    # Use a pattern that approximates optimal placement
    # Start with vertices and centroid, then add points in a structured way
    
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, _H])
    
    # Centroid
    points.append([0.5, _H/3])
    
    # Midpoints of edges
    points.append([0.5, 0.0])
    points.append([0.25, _H/2])
    points.append([0.75, _H/2])
    
    # Additional points - use a pattern that spreads out
    # Points at 1/3 and 2/3 along medians
    # Median from (0,0) to (0.5, _H): points at 1/3 and 2/3
    points.append([0.5/3, _H/3])
    points.append([0.5*2/3, _H*2/3])
    
    # Median from (1,0) to (0.5, _H): 
    points.append([1.0 - 0.5/3, _H/3])
    points.append([1.0 - 0.5*2/3, _H*2/3])
    
    return np.array(points[:N])


def _generate_random_points(rng, n=N):
    """Generate random points inside the triangle."""
    # Use rejection sampling or barycentric coordinates
    points = []
    for _ in range(n):
        # Barycentric coordinates
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        x = r1 + 0.5 * r2
        y = _H * r2
        points.append([x, y])
    return np.array(points)


def _local_optimize(points, time_budget=5.0):
    """
    Run coordinate descent followed by simulated annealing.
    """
    start_time = time.time()
    
    # First, coordinate descent
    points = _coordinate_descent(points, time_limit=min(3.0, time_budget * 0.6))
    
    # Then, simulated annealing
    remaining = time_budget - (time.time() - start_time)
    if remaining > 1.0:
        points = _simulated_annealing(points, time_limit=remaining * 0.8, start_time=start_time)
    
    return _project_to_triangle(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initialization candidates
    init_candidates = []
    
    # 1. Symmetric initialization
    sym_pts = _generate_symmetric_init()
    init_candidates.append(sym_pts)
    
    # 2. Random initializations
    for i in range(5):
        rand_pts = _generate_random_points(rng)
        init_candidates.append(rand_pts)
    
    # 3. Distance-optimized initializations
    for i in range(3):
        if time.time() > deadline - 30:
            break
        rand_pts = _generate_random_points(rng)
        
        def objective_dist(flat):
            pts = _project_to_triangle(flat.reshape(N, 2))
            return -_min_dist_sq(pts)
        
        try:
            res = minimize(objective_dist, rand_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            dist_opt_pts = _project_to_triangle(res.x.reshape(N, 2))
            init_candidates.append(dist_opt_pts)
        except Exception:
            continue
    
    # Process each candidate with local optimization
    for init_pts in init_candidates:
        if time.time() > deadline - 10:
            break
        
        try:
            # Allocate time based on remaining budget
            remaining = deadline - time.time() - 5
            time_per_candidate = max(2.0, remaining / max(1, len(init_candidates)))
            
            optimized_pts = _local_optimize(init_pts, time_budget=time_per_candidate)
            score = _min_triangle_area(optimized_pts)
            
            if score > best_score:
                best_score = score
                best_points = optimized_pts.copy()
                
        except Exception:
            continue
    
    # If we have time, do additional rounds of perturbation + optimization
    if best_points is not None and time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution
            perturbed = best_points.copy()
            i = rng.integers(0, N)
            scale = 0.05
            perturbed[i] += rng.normal(0, scale, size=2)
            perturbed = _project_to_triangle(perturbed)
            
            # Optimize
            remaining = deadline - time.time() - 5
            time_budget = max(2.0, remaining / 4)
            optimized = _local_optimize(perturbed, time_budget=time_budget)
            score = _min_triangle_area(optimized)
            
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
    
    if best_points is None:
        # Fallback: Random points
        best_points = _generate_random_points(rng)
        best_points = _project_to_triangle(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
