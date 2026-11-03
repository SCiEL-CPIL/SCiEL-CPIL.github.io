import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute all triangle indices
indices = list(itertools.combinations(range(N), 3))
n_tri = len(indices)

def barycentric_coords(p):
    """Return (u, v, w) barycentric coordinates for point p."""
    denom = np.sqrt(3) / 2
    v = (p[0] * V2[1] - p[1] * V2[0]) / denom
    w = (p[1] * V1[0] - p[0] * V1[1]) / denom
    u = 1 - v - w
    return u, v, w

def clamp_to_triangle(p):
    """Project point p to the nearest point inside the equilateral triangle."""
    u, v, w = barycentric_coords(p)
    if u >= 0 and v >= 0 and w >= 0:
        return p
    centroid = (V0 + V1 + V2) / 3
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        test_p = centroid + mid * (p - centroid)
        tu, tv, tw = barycentric_coords(test_p)
        if tu >= -1e-10 and tv >= -1e-10 and tw >= -1e-10:
            lo = mid
        else:
            hi = mid
    return centroid + lo * (p - centroid)

def point_in_triangle(p, eps=1e-9):
    u, v, w = barycentric_coords(p)
    return (u >= -eps) and (v >= -eps) and (w >= -eps)

def compute_areas(points):
    """Compute areas for all C(N,3) triangles. Returns array of shape (n_tri,)."""
    areas = np.zeros(n_tri)
    for idx, (i, j, k) in enumerate(indices):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        areas[idx] = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
    return areas

def min_area(points):
    """Return the minimum triangle area."""
    return compute_areas(points).min()

def objective_and_grad(x):
    """
    Objective: negative min area + penalty for being outside triangle.
    Returns (value, gradient).
    x is flattened 2N coordinates.
    """
    points = x.reshape(N, 2)
    
    # Compute all areas
    areas = compute_areas(points)
    
    # Find the minimum area (active constraint)
    min_idx = np.argmin(areas)
    min_val = areas[min_idx]
    
    # Penalty for being outside
    penalty = 0.0
    grad_penalty = np.zeros_like(x)
    for i in range(N):
        u, v, w = barycentric_coords(points[i])
        if u < 0:
            penalty += u * u
            # Gradient of u^2 w.r.t. x: 2u * du/dx
            # u = 1 - v - w where v, w are linear in p
            # du/dp_x = -(dv/dp_x + dw/dp_x)
            # v = (p_x * V2_y - p_y * V2_x) / denom => dv/dp_x = V2_y/denom, dv/dp_y = -V2_x/denom
            # w = (p_y * V1_x - p_x * V1_y) / denom => dw/dp_x = -V1_y/denom, dw/dp_y = V1_x/denom
            # So du/dp_x = -(V2_y/denom - V1_y/denom) = (V1_y - V2_y)/denom
            # du/dp_y = -(-V2_x/denom + V1_x/denom) = (V2_x - V1_x)/denom
            denom = np.sqrt(3) / 2
            du_dx = (V1[1] - V2[1]) / denom
            du_dy = (V2[0] - V1[0]) / denom
            grad_penalty[2*i] += 2 * u * du_dx
            grad_penalty[2*i+1] += 2 * u * du_dy
        if v < 0:
            penalty += v * v
            denom = np.sqrt(3) / 2
            dv_dx = V2[1] / denom
            dv_dy = -V2[0] / denom
            grad_penalty[2*i] += 2 * v * dv_dx
            grad_penalty[2*i+1] += 2 * v * dv_dy
        if w < 0:
            penalty += w * w
            denom = np.sqrt(3) / 2
            dw_dx = -V1[1] / denom
            dw_dy = V1[0] / denom
            grad_penalty[2*i] += 2 * w * dw_dx
            grad_penalty[2*i+1] += 2 * w * dw_dy
    
    # Gradient of -min_area w.r.t. x
    # min_area = 0.5 * |cross| for the active triangle
    # We need to handle the sign properly
    i, j, k = indices[min_idx]
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # cross = (p2-p1) x (p3-p1) = (p2x-p1x)(p3y-p1y) - (p2y-p1y)(p3x-p1x)
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    # area = 0.5 * |cross|
    # If cross > 0, area = 0.5 * cross, d(area)/d(cross) = 0.5
    # If cross < 0, area = -0.5 * cross, d(area)/d(cross) = -0.5
    sign = 1.0 if cross >= 0 else -1.0
    darea_dcross = 0.5 * sign
    
    # Gradient of cross w.r.t. each point
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d(cross)/d(x_i) = -(y_k - y_i) = y_i - y_k
    # d(cross)/d(y_i) = -(x_k - x_i) = x_i - x_k
    # d(cross)/d(x_j) = (y_k - y_i)
    # d(cross)/d(y_j) = -(x_k - x_i) = x_i - x_k
    # d(cross)/d(x_k) = -(y_j - y_i) = y_i - y_j
    # d(cross)/d(y_k) = (x_j - x_i)
    
    grad_area = np.zeros_like(x)
    grad_area[2*i] = darea_dcross * (p1[1] - p3[1])
    grad_area[2*i+1] = darea_dcross * (p1[0] - p3[0])
    grad_area[2*j] = darea_dcross * (p3[1] - p1[1])
    grad_area[2*j+1] = darea_dcross * (p1[0] - p3[0])
    grad_area[2*k] = darea_dcross * (p1[1] - p2[1])
    grad_area[2*k+1] = darea_dcross * (p2[0] - p1[0])
    
    # Objective is -min_area + penalty
    value = -min_val + 10.0 * penalty
    grad = -grad_area + 10.0 * grad_penalty
    
    return value, grad

def objective_only(x):
    """Objective without gradient, for methods that don't use gradients."""
    points = x.reshape(N, 2)
    min_val = min_area(points)
    
    penalty = 0.0
    for i in range(N):
        u, v, w = barycentric_coords(points[i])
        if u < 0: penalty += u * u
        if v < 0: penalty += v * v
        if w < 0: penalty += w * w
    
    return -min_val + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin within 150s
    
    best_points = None
    best_score = -float('inf')
    
    # Generate multiple diverse starting points
    rng = np.random.RandomState(12345)
    
    def random_points_in_triangle():
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                p = u * V1 + v * V2
                pts.append(p)
        return np.array(pts)
    
    # Structured starts
    starts = []
    
    # Vertices + midpoints + center + some symmetric points
    pts_struct = [V0, V1, V2]
    M01 = (V0 + V1) / 2
    M12 = (V1 + V2) / 2
    M20 = (V2 + V0) / 2
    pts_struct.extend([M01, M12, M20])
    C = (V0 + V1 + V2) / 3
    pts_struct.append(C)
    # Quarter points
    Q01a = V0 + 0.25 * (V1 - V0)
    Q01b = V0 + 0.75 * (V1 - V0)
    pts_struct.extend([Q01a, Q01b])
    starts.append(np.array(pts_struct))
    
    # More structured: use a grid-like approach
    # Place points at barycentric coordinates
    bary_coords = [
        (0, 0, 1), (0, 1, 0), (1, 0, 0),  # vertices
        (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),  # midpoints
        (1/3, 1/3, 1/3),  # center
        (0, 0.25, 0.75), (0, 0.75, 0.25),  # quarter points on edge
        (0.25, 0, 0.75), (0.75, 0, 0.25),  # quarter points on another edge
    ]
    pts_bary = []
    for u, v, w in bary_coords:
        p = v * V1 + w * V2
        pts_bary.append(p)
    starts.append(np.array(pts_bary))
    
    # Random starts
    for _ in range(5):
        starts.append(random_points_in_triangle())
    
    # Run BFGS/L-BFGS-B from each start
    for i, initial_pts in enumerate(starts):
        if time.time() > deadline - 5:
            break
        
        x0 = initial_pts.flatten()
        
        # Try L-BFGS-B with gradient
        try:
            res = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            x_opt = res.x
            points_opt = x_opt.reshape(N, 2)
            
            # Clamp any points that might be slightly outside
            for j in range(N):
                points_opt[j] = clamp_to_triangle(points_opt[j])
            
            valid = True
            for p in points_opt:
                if not point_in_triangle(p, eps=1e-4):
                    valid = False
                    break
            
            if valid:
                score = min_area(points_opt)
                if score > best_score:
                    best_score = score
                    best_points = points_opt.copy()
                    
        except Exception:
            pass
        
        # Also try Nelder-Mead from the BFGS result or original
        if time.time() > deadline - 5:
            break
        
        x_nm = res.x if 'res' in dir() and hasattr(res, 'x') else x0
        try:
            res_nm = minimize(
                objective_only,
                x_nm,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            x_opt_nm = res_nm.x
            points_opt_nm = x_opt_nm.reshape(N, 2)
            
            for j in range(N):
                points_opt_nm[j] = clamp_to_triangle(points_opt_nm[j])
            
            valid = True
            for p in points_opt_nm:
                if not point_in_triangle(p, eps=1e-4):
                    valid = False
                    break
            
            if valid:
                score = min_area(points_opt_nm)
                if score > best_score:
                    best_score = score
                    best_points = points_opt_nm.copy()
                    
        except Exception:
            pass
    
    # If we have time, do a final round of refinement from the best point
    if best_points is not None and time.time() < deadline - 10:
        x_best = best_points.flatten()
        try:
            res_final = minimize(
                objective_and_grad,
                x_best,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_final = res_final.x
            points_final = x_final.reshape(N, 2)
            
            for j in range(N):
                points_final[j] = clamp_to_triangle(points_final[j])
            
            valid = True
            for p in points_final:
                if not point_in_triangle(p, eps=1e-4):
                    valid = False
                    break
            
            if valid:
                score = min_area(points_final)
                if score > best_score:
                    best_score = score
                    best_points = points_final.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        pts_fallback = [V0, V1, V2]
        M01 = (V0 + V1) / 2
        M12 = (V1 + V2) / 2
        M20 = (V2 + V0) / 2
        pts_fallback.extend([M01, M12, M20])
        C = (V0 + V1 + V2) / 3
        pts_fallback.append(C)
        Q01a = V0 + 0.25 * (V1 - V0)
        Q01b = V0 + 0.75 * (V1 - V0)
        pts_fallback.extend([Q01a, Q01b])
        best_points = np.array(pts_fallback)
    
    return best_points
