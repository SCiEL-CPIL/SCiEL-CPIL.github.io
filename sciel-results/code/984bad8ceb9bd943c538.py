"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Area of the unit triangle
AREA = np.sqrt(3) / 4.0


def _inside_triangle(p):
    """Check if point p is inside or on the boundary of the triangle."""
    # Barycentric coordinates
    v0 = V1 - V0
    v1 = V2 - V0
    v2 = p - V0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return False
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    return u >= -1e-9 and v >= -1e-9 and (u + v) <= 1 + 1e-9


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _min_triangle_area_fast(points):
    """Compute minimum triangle area, normalized by triangle area."""
    return _min_triangle_area(points) / AREA


def _objective(params):
    """Objective to minimize: negative of min triangle area (so we maximize it).
       Also add penalty if points are outside triangle."""
    n = len(params) // 2
    points = params.reshape(n, 2)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for p in points:
        if not _inside_triangle(p):
            # Compute distance to triangle boundary roughly
            # For simplicity, use a large penalty
            penalty += 10.0
    
    if penalty > 0:
        return penalty
    
    min_area = _min_triangle_area(points)
    return -min_area


def _objective_with_bounds(params):
    """Objective that also ensures points are within bounds."""
    n = len(params) // 2
    points = params.reshape(n, 2)
    
    # Check bounds
    for p in points:
        if p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > np.sqrt(3) / 2:
            return 100.0
    
    min_area = _min_triangle_area(points)
    return -min_area


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Strategy: Use a combination of heuristic initialization and local optimization
    
    # 1. Try a regular polygon-like distribution on the boundary
    # Place points along the edges of the triangle
    def _boundary_distribution():
        """Distribute points evenly along the three edges."""
        points = []
        # Edge 0->1: 4 points
        for i in range(4):
            t = i / 3.0
            points.append(V0 + t * (V1 - V0))
        # Edge 1->2: 4 points
        for i in range(4):
            t = i / 3.0
            points.append(V1 + t * (V2 - V1))
        # Edge 2->0: 3 points
        for i in range(3):
            t = i / 2.0
            points.append(V2 + t * (V0 - V2))
        return np.array(points)
    
    # 2. Try a grid-like distribution
    def _grid_distribution():
        """Create a grid inside the triangle."""
        points = []
        # Center
        points.append((V0 + V1 + V2) / 3.0)
        # Midpoints of edges
        points.append((V0 + V1) / 2.0)
        points.append((V1 + V2) / 2.0)
        points.append((V2 + V0) / 2.0)
        # Quarter points
        for i in range(4):
            t = (i + 1) / 5.0
            points.append(V0 + t * (V1 - V0))
            points.append(V1 + t * (V2 - V1))
            points.append(V2 + t * (V0 - V2))
        # Take first 11
        return np.array(points[:11])
    
    # 3. Random initializations
    def _random_init(seed):
        rng = np.random.RandomState(seed)
        points = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v <= 1:
                    p = u * V1 + v * V2
                    points.append(p)
                    break
        return np.array(points)
    
    # Candidate initializations
    candidates = []
    candidates.append(_boundary_distribution())
    candidates.append(_grid_distribution())
    for seed in range(50):
        candidates.append(_random_init(seed))
    
    # Evaluate all candidates
    for cand in candidates:
        if len(cand) != N:
            continue
        # Check all points are inside
        valid = True
        for p in cand:
            if not _inside_triangle(p):
                valid = False
                break
        if not valid:
            continue
        score = _min_triangle_area(cand)
        if score > best_score:
            best_score = score
            best_points = cand.copy()
    
    if best_points is None:
        # Fallback
        best_points = _random_init(0)
    
    # 4. Local optimization using Nelder-Mead on the best candidate
    if time.time() < deadline - 30:
        try:
            x0 = best_points.flatten()
            bounds = [(0, 1), (0, np.sqrt(3) / 2)] * N
            
            # Use a constrained optimization approach
            # Project points back into triangle if they go outside
            def _project_to_triangle(p):
                """Project a point to the nearest point in the triangle."""
                # Simple approach: if outside, move towards centroid
                centroid = (V0 + V1 + V2) / 3.0
                if _inside_triangle(p):
                    return p
                # Move towards centroid
                for t in np.linspace(0.1, 1.0, 10):
                    p_new = p + t * (centroid - p)
                    if _inside_triangle(p_new):
                        return p_new
                return centroid
            
            def _objective_constrained(params):
                n = len(params) // 2
                points = params.reshape(n, 2)
                # Project to triangle
                points = np.array([_project_to_triangle(p) for p in points])
                min_area = _min_triangle_area(points)
                return -min_area
            
            result = minimize(
                _objective_constrained,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.success or (hasattr(result, 'fun') and result.fun < -best_score):
                new_points = result.x.reshape(N, 2)
                new_points = np.array([_project_to_triangle(p) for p in new_points])
                new_score = _min_triangle_area(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            pass
    
    # 5. Try differential evolution if time permits
    if time.time() < deadline - 60:
        try:
            bounds = [(0, 1), (0, np.sqrt(3) / 2)] * N
            
            def _de_objective(params):
                n = len(params) // 2
                points = params.reshape(n, 2)
                # Check if all inside
                for p in points:
                    if not _inside_triangle(p):
                        return 10.0
                return -_min_triangle_area(points)
            
            de_result = differential_evolution(
                _de_objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42
            )
            
            if de_result.fun < -best_score:
                new_points = de_result.x.reshape(N, 2)
                new_score = _min_triangle_area(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            pass
    
    # Final check
    if best_points is None:
        best_points = _random_init(0)
    
    return np.array(best_points)
