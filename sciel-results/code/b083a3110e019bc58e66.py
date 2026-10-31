import numpy as np
import time
from scipy.optimize import minimize
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

def _soft_min_area(pts, beta=100.0):
    """
    Smooth approximation of min area using log-sum-exp trick.
    """
    areas = _all_areas(pts)
    m = np.min(areas)
    if m < 1e-12:
        return m
    # Shift to avoid overflow in exp
    shifted = -beta * (areas - m)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    return soft_min_val

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

def _generate_initial_points():
    """
    Generate diverse initial configurations.
    """
    inits = []
    
    # 1. Symmetric configuration
    # Center
    pts = []
    pts.append([1/3, H/3])
    # 3 points near vertices
    d = 0.15
    pts.append([d, d * np.sqrt(3)]) # Near V0? No, V0 is (0,0). 
    # Let's use barycentric mixing for symmetry
    # 1 center
    pts = np.array([[1/3, H/3]])
    # 3 points at 1/3 from vertices
    for v in [V0, V1, V2]:
        # Move from vertex towards centroid
        centroid = np.array([1/3, H/3])
        p = 0.7 * v + 0.3 * centroid
        pts = np.vstack([pts, [p[0], p[1]]])
    # 3 points at midpoints of sides
    for i in range(3):
        v1 = [V0, V1, V2][i]
        v2 = [V0, V1, V2][(i+1)%3]
        mid = 0.5 * (v1 + v2)
        pts = np.vstack([pts, [mid[0], mid[1]]])
    # 2 more points to reach 11? We have 1+3+3=7. Need 4 more.
    # Place 4 more in a smaller inner ring?
    # Or just add random perturbations to the existing ones for other starts.
    # For the first start, let's just add 4 more points in a square-like pattern in the center?
    # Actually, let's just use a simple grid for the rest.
    
    # Start 1: Symmetric + Random fill
    pts1 = np.array(pts)
    while len(pts1) < N:
        # Add random points
        r = np.random.rand(2)
        p = [r[0] * 1.0, r[1] * H]
        if _clamp_to_triangle(p).any(): # Always true
             pts1 = np.vstack([pts1, p])
    inits.append(_clamp_all(pts1[:N]))
    
    # 2. Hexagonal grid points
    # Generate a grid of points and pick the best 11 inside
    grid_pts = []
    step = 0.1
    for i in range(int(1/step)+1):
        for j in range(int(H/step)+1):
            x = i * step
            y = j * step
            if x >= 0 and x <= 1 and y >= 0 and y <= H:
                # Check if inside triangle
                # y <= sqrt(3) * x (left side)
                # y <= -sqrt(3) * (x - 1) (right side)
                if y <= np.sqrt(3) * x + 1e-9 and y <= -np.sqrt(3) * (x - 1) + 1e-9:
                    grid_pts.append([x, y])
    grid_pts = np.array(grid_pts)
    if len(grid_pts) >= N:
        # Pick N points that are roughly evenly spaced? 
        # For simplicity, just take the first N or random N?
        # Let's take a subset that maximizes min distance to start
        # But that's another optimization. Let's just take random 11 from the grid.
        idx = np.random.choice(len(grid_pts), N, replace=False)
        inits.append(grid_pts[idx])
        
    # 3. Random starts
    for _ in range(10):
        pts_rand = np.random.rand(N, 2)
        pts_rand[:, 0] *= 1.0
        pts_rand[:, 1] *= H
        inits.append(_clamp_all(pts_rand))
        
    return inits

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit (5s buffer)
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    best_pts = None
    current_best = -np.inf
    
    # Generate initial points
    inits = _generate_initial_points()
    
    # Phase 1: Multi-start L-BFGS-B with soft-min objective
    # We will run L-BFGS-B from each initial point
    # To save time, we might limit the number of starts or iterations per start
    
    # Objective function for L-BFGS-B
    # We use a soft-min with increasing beta to sharpen the minimum
    # Start with beta=50, then maybe increase
    
    for i, init_pts in enumerate(inits):
        if time_left() < 5.0:
            break
            
        # Flatten points for L-BFGS-B
        x0 = init_pts.flatten()
        
        # Define objective
        def obj_lbfgs(x, beta=50.0):
            pts = x.reshape(-1, 2)
            # Clamp to ensure validity
            pts_clamped = _clamp_all(pts)
            return -_soft_min_area(pts_clamped, beta=beta)
        
        # Run L-BFGS-B
        try:
            res = minimize(
                obj_lbfgs,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            candidate_pts = _clamp_all(res.x.reshape(-1, 2))
            val = _min_area(candidate_pts)
            
            if val > current_best:
                current_best = val
                best_pts = candidate_pts
        except Exception:
            pass
            
    # If no points were found (should not happen), fallback
    if best_pts is None:
        best_pts = inits[0]
        current_best = _min_area(best_pts)
        
    # Phase 2: Polish with Nelder-Mead on hard objective
    # Nelder-Mead is derivative-free and can handle the non-smoothness of exact min
    if time_left() > 10.0:
        def obj_nm(x):
            pts = _clamp_all(x.reshape(-1, 2))
            return -_min_area(pts)
            
        try:
            res_nm = minimize(
                obj_nm,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            candidate_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            val = _min_area(candidate_pts)
            if val > current_best:
                best_pts = candidate_pts
                current_best = val
        except Exception:
            pass
            
    # Phase 3: Final L-BFGS-B with higher beta for sharpness
    if time_left() > 5.0:
        def obj_lbfgs2(x, beta=200.0):
            pts = _clamp_all(x.reshape(-1, 2))
            return -_soft_min_area(pts, beta=beta)
            
        try:
            res_lbfgs2 = minimize(
                obj_lbfgs2,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            candidate_pts = _clamp_all(res_lbfgs2.x.reshape(-1, 2))
            val = _min_area(candidate_pts)
            if val > current_best:
                best_pts = candidate_pts
                current_best = val
        except Exception:
            pass

    # Final clamp
    final_pts = _clamp_all(best_pts)
    return final_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Min Area: {_min_area(pts)}")
    print(f"Points:\n{pts}")
