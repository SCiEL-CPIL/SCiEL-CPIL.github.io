"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples
_IDX_I = None
_IDX_J = None
_IDX_K = None
_N_TRIPLES = None

def _get_indices():
    global _IDX_I, _IDX_J, _IDX_K, _N_TRIPLES
    if _IDX_I is None:
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _IDX_I = i[mask]
        _IDX_J = j[mask]
        _IDX_K = k[mask]
        _N_TRIPLES = len(_IDX_I)
    return _IDX_I, _IDX_J, _IDX_K

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_triangle_areas(points):
    """Compute all triangle areas, vectorized."""
    i_idx, j_idx, k_idx = _get_indices()
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    lambda_3 = 1 - s_i
    """
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    lam1 = s * (1.0 - t)
    lam2 = s * t
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    s = np.clip(s, 1e-10, 1.0 - 1e-10)
    t = np.clip(t, 1e-10, 1.0 - 1e-10)
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _generate_fps_initialization(rng, n_points=N, n_candidates=200):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    points[0] = (V0 + V1 + V2) / 3.0
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
    return points

def _structured_inits(rng):
    """
    Generate structured initializations based on grid or symmetric patterns.
    """
    inits = []
    
    # Grid-like patterns
    for grid_n in [3, 4, 5]:
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                l0 = 1 - (i + j) / (grid_n - 1) if grid_n > 1 else 1
                l1 = i / (grid_n - 1) if grid_n > 1 else 0
                l2 = j / (grid_n - 1) if grid_n > 1 else 0
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(points[idx])
        elif len(points) > 0:
            while len(points) < N:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(points)
    
    # Symmetric patterns: 1 center + 10 around
    center = (V0 + V1 + V2) / 3.0
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.3
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.6
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    
    inits.append(np.array(pts))
    
    return inits

def _scp_optimize(points_init, max_iters=50, tol=1e-10, deadline=None):
    """
    Sequential Convex Programming: iteratively solve an LP to maximize t
    subject to linearized area constraints for the active set of smallest triangles.
    
    At each iteration:
    1. Compute all triangle areas at current points
    2. Identify active set (triangles with area close to minimum)
    3. Linearize area constraints around current points
    4. Solve LP: max t s.t. t <= linearized_area_i for i in active set,
       points in triangle, t >= 0
    5. Update points, repeat until convergence
    """
    points = points_init.copy()
    best_points = points.copy()
    best_min_area = _min_triangle_area_vectorized(points)
    
    i_idx, j_idx, k_idx = _get_indices()
    n_triples = len(i_idx)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all areas
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        
        current_min = np.min(areas)
        if current_min > best_min_area:
            best_min_area = current_min
            best_points = points.copy()
        
        # Check convergence
        if current_min < tol:
            break
        
        # Identify active set: triangles with area within 10% of minimum
        active_threshold = current_min * 1.1
        active_mask = areas <= active_threshold
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            active_indices = np.array([np.argmin(areas)])
        
        # Build LP: maximize t
        # Variables: x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}, t
        # Total: 2N + 1 variables
        n_vars = 2 * N + 1
        
        # Objective: maximize t => minimize -t
        c = np.zeros(n_vars)
        c[-1] = -1.0  # t is the last variable
        
        # Inequality constraints: A_ub @ x <= b_ub
        # For each active triple (i, j, k), we have:
        # area(i,j,k) >= t
        # Linearized: area(i,j,k) at current points + gradient dot (dx, dy) >= t
        # => -area_current - grad_x . dx - grad_y . dy + t <= 0
        
        A_ub_rows = []
        b_ub_rows = []
        
        for idx in active_indices:
            i, j, k = i_idx[idx], j_idx[idx], k_idx[idx]
            pi_pt = points[i]
            pj_pt = points[j]
            pk_pt = points[k]
            
            # Compute cross product and its gradient w.r.t. each point
            cross_val = (pj_pt[0] - pi_pt[0]) * (pk_pt[1] - pi_pt[1]) - (pj_pt[1] - pi_pt[1]) * (pk_pt[0] - pi_pt[0])
            area_val = 0.5 * abs(cross_val)
            
            # Gradient of cross w.r.t. each point coordinate
            # cross = (pjx-pix)*(pky-piy) - (pjy-piy)*(pkx-pix)
            # d/dpix = -(pky-piy) + (pjy-piy)
            # d/dpiy = -(pkx-pix) + (pjx-pix)
            # d/dpjx = (pky-piy)
            # d/dpjy = -(pkx-pix)
            # d/dpkx = -(pjy-piy)
            # d/dpky = (pjx-pix)
            
            d_cross_d_pix = -(pk_pt[1] - pi_pt[1]) + (pj_pt[1] - pi_pt[1])
            d_cross_d_piy = -(pk_pt[0] - pi_pt[0]) + (pj_pt[0] - pi_pt[0])
            d_cross_d_pjx = pk_pt[1] - pi_pt[1]
            d_cross_d_pjy = -(pk_pt[0] - pi_pt[0])
            d_cross_d_pkx = -(pj_pt[1] - pi_pt[1])
            d_cross_d_pky = pj_pt[0] - pi_pt[0]
            
            # Area = 0.5 * |cross|
            # d(area)/d(point) = 0.5 * sign(cross) * d(cross)/d(point)
            sign_cross = np.sign(cross_val) if abs(cross_val) > 1e-15 else 1.0
            
            # We want: area_linearized >= t
            # area_current + sum(d_area/d_x * dx) >= t
            # => -area_current - sum(d_area/d_x * dx) + t <= 0
            
            row = np.zeros(n_vars)
            row[2*i] = -0.5 * sign_cross * d_cross_d_pix
            row[2*i+1] = -0.5 * sign_cross * d_cross_d_piy
            row[2*j] = -0.5 * sign_cross * d_cross_d_pjx
            row[2*j+1] = -0.5 * sign_cross * d_cross_d_pjjy if False else -0.5 * sign_cross * d_cross_d_pjy
            row[2*k] = -0.5 * sign_cross * d_cross_d_pkx
            row[2*k+1] = -0.5 * sign_cross * d_cross_d_pky
            row[-1] = 1.0  # +t
            
            b_ub_rows.append(0.5 * area_val)  # area_current
            
            A_ub_rows.append(row)
        
        # Point-in-triangle constraints (barycentric coordinates >= 0)
        # For each point i:
        # x_i >= 0, y_i >= 0
        # x_i + y_i * (2/sqrt(3)) - 0.5 * y_i * (2/sqrt(3)) <= 1  (this is lam3 >= 0)
        # Actually: lam1 = x - 0.5*lam2, lam2 = 2y/sqrt(3), lam3 = 1 - lam1 - lam2
        # lam1 >= 0 => x - y/sqrt(3) >= 0 => -x + y/sqrt(3) <= 0
        # lam2 >= 0 => 2y/sqrt(3) >= 0 => -2y/sqrt(3) <= 0
        # lam3 >= 0 => 1 - x + y/sqrt(3) - 2y/sqrt(3) >= 0 => x + y/sqrt(3) <= 1
        
        # Add these as inequality constraints: A_ub @ x <= b_ub
        for i in range(N):
            # lam1 >= 0: -x_i + y_i/sqrt(3) <= 0
            row1 = np.zeros(n_vars)
            row1[2*i] = -1.0
            row1[2*i+1] = 1.0/np.sqrt(3)
            A_ub_rows.append(row1)
            b_ub_rows.append(0.0)
            
            # lam2 >= 0: -2y_i/sqrt(3) <= 0
            row2 = np.zeros(n_vars)
            row2[2*i+1] = -2.0/np.sqrt(3)
            A_ub_rows.append(row2)
            b_ub_rows.append(0.0)
            
            # lam3 >= 0: x_i + y_i/sqrt(3) <= 1
            row3 = np.zeros(n_vars)
            row3[2*i] = 1.0
            row3[2*i+1] = 1.0/np.sqrt(3)
            A_ub_rows.append(row3)
            b_ub_rows.append(1.0)
        
        # t >= 0: -t <= 0
        row_t = np.zeros(n_vars)
        row_t[-1] = -1.0
        A_ub_rows.append(row_t)
        b_ub_rows.append(0.0)
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            if res.status == 0:
                new_points = np.zeros((N, 2))
                new_points[:, 0] = res.x[0::2]
                new_points[:, 1] = res.x[1::2]
                new_t = res.x[-1]
                
                # Validate: check if new points give a better min area
                new_min_area = _min_triangle_area_vectorized(new_points)
                if new_min_area > best_min_area:
                    best_min_area = new_min_area
                    best_points = new_points.copy()
                
                # Check for convergence
                if abs(new_t - current_min) < tol and new_min_area <= current_min * (1 + 1e-6):
                    points = new_points
                    break
                
                points = new_points
            else:
                # LP failed, break
                break
        except Exception:
            break
    
    return best_points, best_min_area

def construct():
    start_time = time.time()
    deadline_global = start_time + 145.0
    
    def time_left():
        return deadline_global - time.time()

    best_points = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    initial_points = []
    
    # FPS initializations
    n_fps_inits = 8
    for i in range(n_fps_inits):
        if time_left() < 60.0:
            break
        pts = _generate_fps_initialization(rng)
        initial_points.append(pts)
        
    # Structured inits
    for pts in _structured_inits(rng):
        if time_left() < 60.0:
            break
        initial_points.append(pts)
    
    # Random initializations
    for i in range(5):
        if time_left() < 60.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        lam1 = s * (1 - t)
        lam2 = s * t
        x = lam1 * V1[0] + lam2 * V2[0]
        y = lam1 * V1[1] + lam2 * V2[1]
        initial_points.append(np.column_stack([x, y]))
    
    # Run SCP optimization on each initialization
    for init_pts in initial_points:
        if time_left() < 30.0:
            break
        
        try:
            scp_pts, scp_score = _scp_optimize(init_pts, max_iters=30, tol=1e-10, deadline=deadline_global - 25.0)
            if scp_score > best_score:
                best_score = scp_score
                best_points = scp_pts.copy()
        except Exception:
            pass
    
    # If SCP didn't find anything, fall back to simple optimization
    if best_points is None:
        # Use FPS initialization
        best_points = _generate_fps_initialization(rng)
        best_score = _min_triangle_area_vectorized(best_points)
    
    # Final polish with Nelder-Mead on the best solution
    if time_left() > 20.0:
        try:
            params = _points_to_param(best_points)
            bounds = [(0.0, 1.0)] * (2 * N)
            res = minimize(
                _objective, params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_points = _param_to_points(res.x)
        except Exception:
            pass
    
    # Validate and return
    h = np.sqrt(3) / 2
    for i in range(N):
        x, y = best_points[i]
        lam2 = 2.0 * y / np.sqrt(3)
        lam1 = x - 0.5 * lam2
        lam3 = 1.0 - lam1 - lam2
        if lam1 < -1e-8 or lam2 < -1e-8 or lam3 < -1e-8:
            lams = np.array([lam1, lam2, lam3])
            lams = np.maximum(lams, 0.0)
            lams = lams / np.sum(lams)
            best_points[i] = lams[0] * V0 + lams[1] * V1 + lams[2] * V2
    
    return best_points
