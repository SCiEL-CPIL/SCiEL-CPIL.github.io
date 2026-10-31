import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute triangle indices for vectorized area computation
_TRI_IDX = np.array(list(combinations(range(N), 3)))
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """
    Compute all triangle areas for N points.
    pts: (N, 2) array
    returns: array of shape (_N_TRI,)
    """
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """
    Compute the minimum area of any triangle formed by three of the points.
    """
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """
    Clamp a point to be inside the equilateral triangle.
    """
    x, y = p
    # Barycentric coordinates relative to V0(0,0), V1(1,0), V2(0.5, H)
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-12
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """
    Clamp all points to the triangle.
    """
    return np.array([_clamp_to_triangle(p) for p in pts])

def _objective(x):
    """
    Objective function for minimization: negative of the minimum area.
    """
    pts = x.reshape(-1, 2)
    # Clamp to ensure validity
    pts_clamped = _clamp_all(pts)
    min_a = _min_area(pts_clamped)
    return -min_a

def _generate_hexagonal_init():
    """
    Generate a hexagonal lattice based initial guess.
    For 11 points, a 3x3 grid with corners cut or similar pattern is often good.
    Let's try a simple grid and clamp.
    """
    # A common good configuration for 10 points is a 3x3 grid missing one corner?
    # Or a hexagonal arrangement.
    # Let's create a 4x4 grid of points in the bounding box and pick 11 well-spaced ones?
    # Simpler: Use a 3x3 grid + 2 center-ish points?
    
    # Let's try a specific known good pattern for 11 points in a triangle.
    # Often related to packing circles.
    # Let's just use a 3x3 grid scaled to fit, plus 2 more.
    # Grid points:
    # (0,0), (1/3,0), (2/3,0), (1,0) -- 4 points on bottom
    # (1/6, H/3), (1/2, H/3), (5/6, H/3) -- 3 points on middle row? No, triangle shrinks.
    
    # Let's use a simpler heuristic: 11 points roughly evenly distributed.
    # 1 center, 3 near vertices, 3 near midpoints, 4 more?
    
    # Let's just return a random valid point set as a fallback if specific logic fails,
    # but DE is robust to bad starts.
    # However, a good start helps.
    
    # Try a 3x3 grid in the triangle
    # y levels: 0, H/3, 2H/3
    # x ranges:
    # y=0: x in [0, 1] -> 4 points: 0, 1/3, 2/3, 1
    # y=H/3: x in [1/6, 5/6] -> 3 points: 1/6, 1/2, 5/6
    # y=2H/3: x in [1/3, 2/3] -> 2 points: 1/3, 2/3
    # Total 4+3+2 = 9 points.
    # Need 2 more.
    # Maybe add points at (1/2, H/2) and somewhere else?
    
    pts = []
    # Row 0
    for x in [0.0, 1/3, 2/3, 1.0]:
        pts.append([x, 0.0])
    # Row 1
    for x in [1/6, 1/2, 5/6]:
        pts.append([x, H/3])
    # Row 2
    for x in [1/3, 2/3]:
        pts.append([x, 2*H/3])
    
    # We have 9 points. Add 2 more.
    # Perhaps shift the whole grid slightly or add points in the gaps?
    # Let's add points at (0.5, H/6) and (0.5, 2H/3 + H/6)? No, outside.
    # Let's add points at (0.5, H/2) and (0.5, H/4)?
    
    # Actually, let's just let DE do the work. A random start is fine for DE.
    # But we can provide a "grid" start.
    return np.array(pts[:N]) # If we had exactly N. We have 9.
    # Let's just pad with random points for the rest in the actual call.

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit (5s buffer)
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Define bounds for DE
    # x in [0, 1], y in [0, H]
    # But we need to ensure points are inside the triangle.
    # DE operates in a hyper-rectangle. We will clamp after each evaluation.
    # However, clamping inside the objective function is crucial for DE to work properly
    # because DE might propose points outside, and clamping them to the boundary
    # effectively creates a "soft" boundary for the optimizer.
    
    bounds = [(0, 1), (0, H)] * N
    
    # Initial population:
    # 1. Hexagonal-ish grid
    # 2. Random points
    
    # Let's create a specific good start
    # 1 center (1/3, H/3)
    # 3 vertices (0,0), (1,0), (0.5, H)
    # 3 midpoints (0.5,0), (0.75, H/2), (0.25, H/2)
    # 4 more?
    
    # Let's just use DE with a large population.
    
    best_result = None
    best_val = -np.inf
    
    # Run Differential Evolution
    # maxiter should be tuned based on time.
    # 11 points * 2 coords = 22 dimensions.
    # DE is expensive.
    
    try:
        # Use a callback to stop early if time runs out?
        # SciPy DE doesn't have a direct time-based callback that stops cleanly in all versions,
        # but we can check time in the objective and raise an exception to stop.
        
        def objective_with_time_check(x):
            if time_left() < 5.0:
                raise TimeoutError()
            return _objective(x)

        # Run DE
        # tol=1e-8, maxiter=500 might be too many if slow.
        # Let's try 200 iterations.
        res = differential_evolution(
            objective_with_time_check,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False  # We will do our own polishing
        )
        
        if res.success or res.fun < best_val:
            best_pts = _clamp_all(res.x.reshape(-1, 2))
            best_val = _min_area(best_pts)
            best_result = best_pts
            
    except TimeoutError:
        pass
    except Exception as e:
        pass

    # If DE failed or didn't run, generate a random fallback
    if best_result is None:
        pts_rand = np.random.rand(N, 2)
        pts_rand[:, 0] *= 1.0
        pts_rand[:, 1] *= H
        best_result = _clamp_all(pts_rand)
        best_val = _min_area(best_result)

    # Phase 2: Polish with Nelder-Mead
    # Nelder-Mead is good for refining the exact non-smooth minimum.
    if time_left() > 10.0:
        def obj_nm(x):
            pts = _clamp_all(x.reshape(-1, 2))
            return -_min_area(pts)
            
        try:
            res_nm = minimize(
                obj_nm,
                best_result.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            candidate_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            val = _min_area(candidate_pts)
            if val > best_val:
                best_result = candidate_pts
                best_val = val
        except Exception:
            pass

    return best_result

if __name__ == "__main__":
    pts = construct()
    print(f"Min Area: {_min_area(pts)}")
    print(f"Points:\n{pts}")
