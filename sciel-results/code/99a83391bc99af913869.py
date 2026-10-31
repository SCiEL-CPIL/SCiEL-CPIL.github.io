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

def _soft_min_area(pts, beta=50.0):
    """
    Smooth approximation of min area using log-sum-exp trick.
    """
    areas = _all_areas(pts)
    m = np.min(areas)
    if m <= 1e-15:
        return m
    # Shift to avoid overflow in exp
    shifted = -beta * (areas - m)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

def _clamp_to_triangle(p):
    """
    Clamp a point to be inside the equilateral triangle.
    Uses barycentric coordinates.
    """
    x, y = p
    # Barycentric coordinates relative to V0(0,0), V1(1,0), V2(0.5, H)
    # p = a*V0 + b*V1 + c*V2
    # x = b + 0.5c
    # y = c * H
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-12
    # Clamp barycentric coords to [eps, 1-2eps] to keep strictly inside
    # but effectively on the boundary if needed
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    # Normalize to ensure they sum to 1
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

def _objective_de(flat_pts):
    """
    Objective for Differential Evolution.
    We use a penalty-free approach by clamping, but DE explores the bounding box.
    To guide DE towards the interior, we can use a soft penalty or just rely on
    the fact that points outside have lower "effective" area if we consider
    the clamped position. However, standard DE works better with bounded search.
    We will search in the bounding box [0,1]x[0,H] and clamp for evaluation.
    """
    pts = flat_pts.reshape(-1, 2)
    # Clamp points to ensure they are valid for area calculation
    pts_clamped = _clamp_all(pts)
    
    # Use soft min for smoothness in DE
    # Beta=20 is a good balance for DE
    area_val = _soft_min_area(pts_clamped, beta=20.0)
    
    # We want to maximize min area, so minimize -min area
    return -area_val

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin for 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bounds for differential evolution
    # x in [0, 1], y in [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    best_pts = None
    current_best = -np.inf
    
    # Phase 1: Differential Evolution for global search
    # DE is robust and good for finding a decent global minimum
    try:
        if time_left() > 60.0:
            # popsize=15, maxiter=50 is a reasonable start for 22 dimensions
            # We can increase iterations if time permits, but let's start conservative
            res_de = differential_evolution(
                _objective_de,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                polish=False,
                updating='immediate'
            )
            # Clamp the result
            candidate_pts = _clamp_all(res_de.x.reshape(-1, 2))
            val = _min_area(candidate_pts)
            if val > current_best:
                best_pts = candidate_pts
                current_best = val
        else:
            # Fallback: symmetric initial placement
            best_pts = _initial_placement_symmetric()
            current_best = _min_area(best_pts)
    except Exception:
        best_pts = _initial_placement_symmetric()
        current_best = _min_area(best_pts)
    
    if best_pts is None:
        best_pts = _initial_placement_symmetric()
        current_best = _min_area(best_pts)

    # Phase 2: L-BFGS-B on smooth objective
    # L-BFGS-B is very fast for smooth functions and handles box constraints well
    if time_left() > 15.0:
        try:
            # Increase beta for sharper approximation of min
            def obj_lbfgs(x):
                pts = _clamp_all(x.reshape(-1, 2))
                return -_soft_min_area(pts, beta=100.0)
            
            res_lbfgs = minimize(
                obj_lbfgs,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            candidate_pts = _clamp_all(res_lbfgs.x.reshape(-1, 2))
            val = _min_area(candidate_pts)
            if val > current_best:
                best_pts = candidate_pts
                current_best = val
        except Exception:
            pass

    # Phase 3: Nelder-Mead on hard objective
    # NM is derivative-free and can handle the non-smoothness of the exact min
    if time_left() > 10.0:
        try:
            def obj_nm(x):
                pts = _clamp_all(x.reshape(-1, 2))
                return -_min_area(pts)
            
            res_nm = minimize(
                obj_nm,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            candidate_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            val = _min_area(candidate_pts)
            if val > current_best:
                best_pts = candidate_pts
                current_best = val
        except Exception:
            pass

    # Phase 4: Powell on hard objective
    # Powell is often better than NM for constrained problems in higher dimensions
    if time_left() > 5.0:
        try:
            def obj_powell(x):
                pts = _clamp_all(x.reshape(-1, 2))
                return -_min_area(pts)
            
            res_powell = minimize(
                obj_powell,
                best_pts.flatten(),
                method='Powell',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-10}
            )
            candidate_pts = _clamp_all(res_powell.x.reshape(-1, 2))
            val = _min_area(candidate_pts)
            if val > current_best:
                best_pts = candidate_pts
                current_best = val
        except Exception:
            pass

    # Phase 5: Final L-BFGS-B polish with higher beta
    if time_left() > 2.0:
        try:
            def obj_lbfgs2(x):
                pts = _clamp_all(x.reshape(-1, 2))
                return -_soft_min_area(pts, beta=200.0)
            
            res_lbfgs2 = minimize(
                obj_lbfgs2,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            candidate_pts = _clamp_all(res_lbfgs2.x.reshape(-1, 2))
            val = _min_area(candidate_pts)
            if val > current_best:
                best_pts = candidate_pts
                current_best = val
        except Exception:
            pass

    # Final clamp to ensure validity
    final_pts = _clamp_all(best_pts)
    return final_pts

def _initial_placement_symmetric():
    """
    Generate a symmetric initial placement.
    1 center, 6 around, 4 near vertices?
    Let's use a simple grid or specific symmetric points.
    """
    pts = []
    # Center
    pts.append([1/3, H/3])
    
    # 3 points near vertices (scaled)
    m = 0.1
    pts.append(m * V0 + (1-m) * (V1+V2)/2) # Midpoint of V1V2 scaled towards V0? No, just mix
    # Better:
    # V0 is (0,0). V1 is (1,0). V2 is (0.5, H).
    # Points close to vertices:
    pts.append([0.1, 0.1])
    pts.append([0.9, 0.1])
    pts.append([0.5, H - 0.1])
    
    # Points in between
    pts.append([0.5, H/3])
    pts.append([0.25, H/3])
    pts.append([0.75, H/3])
    
    pts.append([0.3, 0.3])
    pts.append([0.7, 0.3])
    pts.append([0.5, 0.1])
    
    return _clamp_all(np.array(pts))

if __name__ == "__main__":
    # Simple test
    pts = construct()
    print(f"Min Area: {_min_area(pts)}")
    print(f"Points:\n{pts}")
