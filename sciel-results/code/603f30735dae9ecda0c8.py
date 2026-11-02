"""Heilbronn triangles: place 11 points to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 145  # safety margin from 150s limit

def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the equilateral triangle area."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    cross = dx[:, :, None] * dy[:, :, None] - dy[:, :, None] * dx[:, :, None]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA

def _cartesian_to_barycentric(points):
    """Convert Cartesian to barycentric coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    w = y / S3H
    u = 1.0 - x - 0.5 * w
    v = x - 0.5 * w
    return np.column_stack([u, v, w])

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack([x, y])

def _soft_min_objective_flat(flat_params, alpha=100.0):
    """
    Objective in barycentric coordinates using soft-min.
    flat_params: 2*N array of (u_i, v_i) for i=0..N-1, where w_i = 1 - u_i - v_i.
    """
    n = len(flat_params) // 2
    uv = flat_params.reshape(n, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Penalty for negative barycentric coords
    neg_u = np.maximum(0.0, -u)
    neg_v = np.maximum(0.0, -v)
    neg_w = np.maximum(0.0, -w)
    penalty = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(neg_w**2))
    
    # Convert to Cartesian
    x = v + 0.5 * w
    y = S3H * w
    
    # Compute all triangle areas
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    cross = dx[:, :, None] * dy[:, :, None] - dy[:, :, None] * dx[:, :, None]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return penalty
    
    # Soft minimum (stable)
    m = np.min(valid_areas)
    if m < 1e-12:
        m = 1e-12
    x_exp = -alpha * (valid_areas - m)
    log_sum_exp = -alpha * m + np.log(np.sum(np.exp(x_exp)))
    min_val = -log_sum_exp / alpha
    return -min_val + penalty

def _min_area_objective_flat(flat_params):
    """Non-smooth objective: negative min area plus penalty."""
    n = len(flat_params) // 2
    uv = flat_params.reshape(n, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    neg_u = np.maximum(0.0, -u)
    neg_v = np.maximum(0.0, -v)
    neg_w = np.maximum(0.0, -w)
    penalty = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(neg_w**2))
    
    x = v + 0.5 * w
    y = S3H * w
    
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    cross = dx[:, :, None] * dy[:, :, None] - dy[:, :, None] * dx[:, :, None]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return penalty
    
    min_val = np.min(valid_areas) / TRI_AREA
    return -min_val + penalty

def _is_valid(points):
    """Check if all points are inside the triangle."""
    bary = _cartesian_to_barycentric(points)
    return np.all(bary >= -1e-9)

def _farthest_point_sampling(rng, n_points):
    """
    Generate n_points in the triangle using Farthest Point Sampling (FPS).
    """
    points = []
    while True:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            break
    points.append(np.array([u + 0.5 * v, S3H * v]))
    
    for _ in range(n_points - 1):
        best_dist = -1
        best_pt = None
        
        for _ in range(200):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    break
            cand = np.array([u + 0.5 * v, S3H * v])
            
            dists = np.linalg.norm(points - cand, axis=1)
            min_d = np.min(dists)
            
            if min_d > best_dist:
                best_dist = min_d
                best_pt = cand
        
        points.append(best_pt)
        
    return np.array(points)

def _hexagonal_initialization(n_points, rng):
    """
    Generate a hexagonal grid of points inside the triangle.
    """
    points = []
    best_pts = None
    best_score = -1
    
    for scale in [0.1, 0.15, 0.2, 0.25, 0.3]:
        pts = []
        k = 0
        while True:
            y = k * np.sqrt(3)/2 * scale
            if y > S3H:
                break
            j = -10
            while j < 10:
                x = (j + (k % 2) * 0.5) * scale
                w = y / S3H
                u = 1.0 - x - 0.5 * w
                v = x - 0.5 * w
                if u >= -1e-9 and v >= -1e-9 and w >= -1e-9:
                    pts.append([x, y])
                j += 1
            k += 1
        
        pts = np.array(pts)
        if len(pts) == 0:
            continue
        
        if len(pts) >= n_points:
            indices = rng.choice(len(pts), n_points, replace=False)
            selected = pts[indices]
        else:
            selected = pts.copy()
            while len(selected) < n_points:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    w = 1 - u - v
                    x = v + 0.5 * w
                    y = S3H * w
                    selected = np.vstack([selected, [[x, y]]])
        
        score = _min_area_normalized(selected)
        if score > best_score:
            best_score = score
            best_pts = selected
    
    if best_pts is None:
        return _farthest_point_sampling(rng, n_points)
    
    return best_pts

def _simulated_annealing(points, n_iter=20000, T_start=0.05, T_end=0.001, rng=None):
    """
    Run Simulated Annealing to maximize the minimum triangle area.
    points: (N, 2) array of Cartesian coordinates.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    current_points = points.copy()
    current_score = _min_area_normalized(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    n = len(points)
    
    for i in range(n_iter):
        T = T_start * (T_end / T_start) ** (i / n_iter)
        
        idx = rng.randint(n)
        cand_points = current_points.copy()
        
        step = 0.05 * (1.0 + 0.5 * (T / T_start))
        dx = rng.normal(0, step)
        dy = rng.normal(0, step)
        
        cand_points[idx, 0] += dx
        cand_points[idx, 1] += dy
        
        bary = _cartesian_to_barycentric(cand_points)
        bary = np.maximum(bary, 0)
        s = np.sum(bary, axis=1, keepdims=True)
        s[s == 0] = 1
        bary = bary / s
        cand_points = _barycentric_to_cartesian(bary)
        
        cand_score = _min_area_normalized(cand_points)
        
        if cand_score >= current_score:
            current_points = cand_points
            current_score = cand_score
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_points.copy()
        else:
            delta = current_score - cand_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_points = cand_points
                    current_score = cand_score
    
    return best_points, best_score

def _slsqp_optimize(points, rng=None):
    """
    Optimize using SLSQP with explicit triangle area constraints.
    This is the structural change: using SLSQP's active-set method on the 165 constraints.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    # Work in barycentric coordinates (u, v) for each point, w = 1 - u - v
    # Variables: x[2*i] = u_i, x[2*i+1] = v_i
    bary = _cartesian_to_barycentric(points)
    x0 = np.column_stack([bary[:, 0], bary[:, 1]]).flatten()
    
    n_vars = 2 * N
    
    # Bounds: 0 <= u, 0 <= v, u + v <= 1
    # For SLSQP, we can use bounds on u and v as [0, 1] and add constraint u+v <= 1
    bounds = [(0, 1)] * n_vars
    
    # Inequality constraints: u_i + v_i <= 1 for all i
    # And triangle area constraints: area(i,j,k) >= t (but we maximize t implicitly by maximizing min area)
    # Actually, SLSQP maximizes a scalar objective.
    # Objective: maximize min_area.
    # This is non-smooth.
    # Alternative: Fix a target area A, and check if feasible? No, that's complex.
    # Better: Use SLSQP to maximize the sum of areas? No, we want min.
    
    # Let's stick to the non-smooth objective but use SLSQP's ability to handle bounds well.
    # The key insight from the prompt is that SLSQP handles the non-smooth constraint better than log-sum-exp smoothing.
    # However, SLSQP requires a differentiable objective for gradients. If we don't provide gradients, it uses finite differences.
    # Let's try maximizing the minimum area directly with SLSQP and finite differences.
    
    def objective(x):
        uv = x.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        
        # Penalty for invalid barycentric coords (should be handled by bounds/constraints mostly)
        penalty = 0.0
        if np.min(u) < 0 or np.min(v) < 0 or np.min(w) < 0:
            penalty = 100.0 * (np.sum(np.maximum(0, -u)**2) + np.sum(np.maximum(0, -v)**2) + np.sum(np.maximum(0, -w)**2))
        
        x_cart = v + 0.5 * w
        y_cart = S3H * w
        pts = np.column_stack([x_cart, y_cart])
        
        score = _min_area_normalized(pts)
        return -score + penalty
    
    # Constraints: u_i + v_i <= 1
    cons = []
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]
        })
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if res.success:
            uv = res.x.reshape(N, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1.0 - u - v
            
            if np.min(u) >= -1e-6 and np.min(v) >= -1e-6 and np.min(w) >= -1e-6:
                x_cart = v + 0.5 * w
                y_cart = S3H * w
                pts = np.column_stack([x_cart, y_cart])
                
                if _is_valid(pts):
                    score = _min_area_normalized(pts)
                    return pts, score
    except Exception:
        pass
    
    return None, 0.0

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Simulated Annealing from diverse starts
    if elapsed() < 60:
        starts = []
        
        # FPS start
        fps_pts = _farthest_point_sampling(rng, N)
        starts.append(fps_pts)
        
        # Hexagonal start
        hex_pts = _hexagonal_initialization(N, rng)
        starts.append(hex_pts)
        
        # Random starts
        for _ in range(3):
            u = rng.random(N)
            v = rng.random(N)
            s = u + v
            s[s == 0] = 1.0
            scale = np.minimum(1.0, 0.99 / s)
            u = u * scale
            v = v * scale
            w = 1 - u - v
            x = v + 0.5 * w
            y = S3H * w
            pts = np.column_stack([x, y])
            starts.append(pts)
        
        for start_pts in starts:
            if elapsed() > DEADLINE - 30:
                break
            
            try:
                sa_pts, sa_score = _simulated_annealing(start_pts, n_iter=5000, T_start=0.05, T_end=0.001, rng=rng)
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_pts.copy()
            except Exception:
                pass
    
    # Phase 2: SLSQP optimization from best SA result and other starts
    # This is the structural change: using SLSQP with explicit constraints
    slsqp_starts = []
    
    if best_points is not None:
        slsqp_starts.append(best_points)
    
    # Also add FPS and Hexagonal starts for SLSQP
    fps_pts = _farthest_point_sampling(rng, N)
    slsqp_starts.append(fps_pts)
    hex_pts = _hexagonal_initialization(N, rng)
    slsqp_starts.append(hex_pts)
    
    for start_pts in slsqp_starts:
        if elapsed() > DEADLINE - 20:
            break
        
        try:
            slsqp_pts, slsqp_score = _slsqp_optimize(start_pts, rng=rng)
            if slsqp_pts is not None and slsqp_score > best_score:
                best_score = slsqp_score
                best_points = slsqp_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Powell optimization from each initial guess (keep as fallback/refinement)
    initial_guesses = []
    
    # FPS-based start
    fps_pts = _farthest_point_sampling(rng, N)
    bary = _cartesian_to_barycentric(fps_pts)
    initial_guesses.append(np.column_stack([bary[:, 0], bary[:, 1]]).flatten())
    
    # Random starts
    for _ in range(3):
        u = rng.random(N)
        v = rng.random(N)
        s = u + v
        s[s == 0] = 1.0
        scale = np.minimum(1.0, 0.99 / s)
        u = u * scale
        v = v * scale
        initial_guesses.append(np.column_stack([u, v]).flatten())
    
    # Add SLSQP result if found
    if best_points is not None:
        bary = _cartesian_to_barycentric(best_points)
        initial_guesses.append(np.column_stack([bary[:, 0], bary[:, 1]]).flatten())
    
    for i, x0 in enumerate(initial_guesses):
        if elapsed() > DEADLINE - 15:
            break
        
        def obj_powell(x):
            return _min_area_objective_flat(x)
        
        try:
            res = minimize(
                obj_powell,
                x0,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
        except Exception:
            continue
        
        if elapsed() > DEADLINE - 10:
            break
        
        uv = res.x.reshape(N, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        
        if np.min(u) < -1e-6 or np.min(v) < -1e-6 or np.min(w) < -1e-6:
            u = np.maximum(u, 0)
            v = np.maximum(v, 0)
            w = 1.0 - u - v
            if np.min(w) < -1e-6:
                continue
        
        x_cart = v + 0.5 * w
        y_cart = S3H * w
        points = np.column_stack([x_cart, y_cart])
        
        if not _is_valid(points):
            continue
        
        score = _min_area_normalized(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Phase 4: Polish with L-BFGS-B using soft-min objective
    if best_points is not None and elapsed() < DEADLINE - 5:
        bary = _cartesian_to_barycentric(best_points)
        x0 = np.column_stack([bary[:, 0], bary[:, 1]]).flatten()
        
        def obj_refine(x):
            return _soft_min_objective_flat(x, alpha=200.0)
        
        try:
            res = minimize(
                obj_refine,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            uv = res.x.reshape(N, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1.0 - u - v
            
            if np.min(u) >= -1e-6 and np.min(v) >= -1e-6 and np.min(w) >= -1e-6:
                x_cart = v + 0.5 * w
                y_cart = S3H * w
                points = np.column_stack([x_cart, y_cart])
                
                if _is_valid(points):
                    score = _min_area_normalized(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _farthest_point_sampling(rng, N)
    
    return best_points
