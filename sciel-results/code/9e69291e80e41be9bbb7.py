import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2, jitter=0.0):
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    if jitter > 0:
        rng = np.random.RandomState(42)
        pts1 += rng.uniform(-jitter, jitter, pts1.shape)
        pts2 += rng.uniform(-jitter, jitter, pts2.shape)
    return np.vstack([pts1, pts2])

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)

    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
    weights = exp_vals / sum_exp
    
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
    
    score = smooth_min / ha
    
    signs = np.sign(cross)
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1])
    d_cross_di_y = (v1[:, 0] + v2[:, 0])
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def _local_refinement(points, max_iter=50):
    """
    Coordinate descent: optimize each point individually to maximize 
    the minimum area of triangles involving that point.
    """
    pts = points.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Define objective for point i: minimize -min_area_triangles_involving_i
            # We approximate this by looking at all triangles involving i.
            # To be efficient, we only consider triangles where i is a vertex.
            # However, the global min might be in a triangle not involving i.
            # A better local objective: maximize the minimum area of all triangles.
            # This is expensive. 
            # Alternative: maximize the minimum area of triangles involving i, 
            # subject to not decreasing the global min too much.
            
            # Let's stick to the global smooth score for the gradient, 
            # but use a line search for step size.
            # Actually, a simpler local move: try small random perturbations 
            # and keep the best one (simulated annealing step).
            
            # Let's use a dedicated local optimizer for point i
            # Objective: -_score(pts)
            # This is non-smooth. 
            # We will use Nelder-Mead on the 2 coordinates of point i.
            
            def _neg_score_fixed_i(params):
                new_pts = pts.copy()
                new_pts[i] = params
                return -_score(new_pts)
            
            x0 = pts[i].copy()
            try:
                res = scipy.optimize.minimize(
                    _neg_score_fixed_i,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 20, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < -_score(pts):
                    pts[i] = res.x
                    improved = True
            except Exception:
                pass
        if not improved:
            break
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _run_optimization(init_points, name=""):
        if time.time() > deadline - 10:
            return
        
        x0 = _normalize(init_points).flatten()
        
        # Stage 1: Coarse optimization with low beta
        # Stage 2: Fine optimization with high beta
        # Stage 3: Local refinement
        
        x = x0.copy()
        
        # Adaptive beta annealing
        betas = [10.0, 50.0, 200.0, 1000.0]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final polish with local refinement
        if time.time() < deadline - 5:
            pts = x.reshape(-1, 2)
            pts = _local_refinement(pts, max_iter=20)
            _update_best(pts)

    # 1. Generate diverse initial configurations
    inits = []
    
    # Regular polygon
    inits.append(_regular_polygon(N, 1.0))
    
    # Two-ring 7-7
    for rot in [0, np.pi/7, np.pi/14]:
        inits.append(_generate_two_ring_config(7, 7, 1.0, 0.5, 0.0, rot))
        inits.append(_generate_two_ring_config(7, 7, 1.0, 0.6, 0.0, rot))
        inits.append(_generate_two_ring_config(7, 7, 1.0, 0.7, 0.0, rot))
        
    # Two-ring 6-8
    for rot in [0, np.pi/6, np.pi/8]:
        inits.append(_generate_two_ring_config(6, 8, 1.0, 0.5, 0.0, rot))
        inits.append(_generate_two_ring_config(6, 8, 1.0, 0.6, 0.0, rot))
        
    # Two-ring 5-9
    inits.append(_generate_two_ring_config(5, 9, 1.0, 0.5, 0.0, 0.0))
    
    # Random
    for _ in range(5):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))

    # 2. Optimize each initial configuration
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        _run_optimization(init, name=f"init_{i}")

    # 3. If time remains, try simulated annealing on the best point
    if best_points is not None and time.time() < deadline - 20:
        pts = best_points.copy()
        current_score = _score(pts)
        T = 0.1
        for step in range(50):
            if time.time() > deadline - 5:
                break
            # Perturb a random point
            i = rng.randint(N)
            new_pts = pts.copy()
            new_pts[i] += rng.uniform(-0.1, 0.1, 2)
            new_score = _score(new_pts)
            
            # Metropolis criterion
            if new_score > current_score or rng.rand() < np.exp((new_score - current_score) / T):
                pts = new_pts
                current_score = new_score
                _update_best(pts)
            T *= 0.9

    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
