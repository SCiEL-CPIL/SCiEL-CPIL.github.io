"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, linprog
import itertools

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]
N_TRIPLES = len(_IDX_I)

def _bary_to_cart(bary):
    """Convert barycentric coords (l0, l1, l2) to Cartesian. bary shape (N, 3)."""
    l1 = bary[:, 1:2]
    l2 = bary[:, 2:3]
    x = l1 * V1[0] + l2 * V2[0]
    y = l1 * V1[1] + l2 * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords. Returns (N, 3)."""
    x = pts[:, 0]
    y = pts[:, 1]
    l2 = 2.0 * y / np.sqrt(3)
    l1 = x - 0.5 * l2
    l0 = 1.0 - l1 - l2
    return np.column_stack([l0, l1, l2])

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _softmin_objective(params, beta=100.0):
    """Smooth surrogate for the min triangle area using soft-min (log-sum-exp based)."""
    points = _param_to_points(params)
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    shifted = areas - min_area
    # Numerically stable log-sum-exp
    max_shift = np.max(shifted)
    log_sum = (max_shift + np.log(np.sum(np.exp(-beta * (shifted - max_shift))))) / beta
    soft_min = min_area - log_sum
    
    return -soft_min

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    """
    s = params[0::2]
    t = params[1::2]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
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
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
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
    
    # Symmetric patterns
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

def _linprog_solve(points, max_iter=50, tol=1e-10):
    """
    Solve the LP for fixed orientation class.
    Given points, determine the sign of each triangle determinant,
    then solve: maximize t subject to signed_area(i,j,k) >= t for all triples,
    and barycentric constraints.
    """
    # Compute signed areas for current points
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    signs = np.sign(cross)
    signs[signs == 0] = 1.0  # Default to positive for degenerate cases
    
    # LP variables: x_0..x_{N-1}, y_0..y_{N-1}, t
    # Total: 2*N + 1 variables
    n_vars = 2 * N + 1
    
    # Objective: minimize -t => c = [0, ..., 0, -1]
    c = np.zeros(n_vars)
    c[-1] = -1.0
    
    # Constraints:
    # 1. For each triple (i,j,k): sign_m * (cross_ij_k) >= t
    #    cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    #    This is bilinear, not linear! 
    #    
    #    Wait, the area is quadratic in the coordinates. The LP approach doesn't work directly
    #    because the area is not linear in the point coordinates.
    #    
    #    Let me reconsider. The strategy mentioned "topological sorting to fix the sign of all 165 
    #    triangle determinants, thereby transforming the non-convex maximin problem into a sequence 
    #    of convex Linear Programs". But the area is quadratic, not linear.
    #    
    #    Actually, I think the idea is different. Let me think about this more carefully.
    #    
    #    The key insight might be that if we fix the combinatorial structure (which points form 
    #    which triangles and their orientation), the problem might be solvable. But the area 
    #    constraint is still quadratic.
    #    
    #    Let me try a different approach: use the LP to find a good lower bound t, then use 
    #    local optimization to improve.
    #    
    #    Actually, let me just use local optimization methods more effectively.
    
    return points, 0.0  # Placeholder - LP approach doesn't directly apply

def _differential_evolution_search(rng, start_time, deadline, n_points=N):
    """
    Use differential evolution for global search.
    """
    from scipy.optimize import differential_evolution
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    def objective(params):
        return -_min_triangle_area_vectorized(_param_to_points(params))
    
    try:
        result = differential_evolution(
            objective, bounds, 
            seed=rng.randint(2**31),
            maxiter=100,
            tol=1e-8,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.8,
            polish=False
        )
        return result.x, -result.fun
    except Exception:
        return None, 0.0

def _local_polish(params, start_time, deadline, max_iter_nm=2000):
    """
    Local polish using Nelder-Mead with tight tolerances.
    """
    if time.time() > deadline - 5.0:
        return params
    
    try:
        res = minimize(
            _objective, params, method='Nelder-Mead',
            options={'maxiter': max_iter_nm, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        return res.x
    except Exception:
        return params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Generate diverse initializations
    initial_params = []
    
    # FPS-based inits
    for i in range(6):
        if time_left() < 40.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
    
    # Structured inits
    for pts in _structured_inits(rng):
        if time_left() < 40.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)
    
    # Random inits
    for i in range(6):
        if time_left() < 40.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)
    
    # Phase 2: Local optimization from each initialization
    bounds = [(0.0, 1.0)] * (2 * N)
    
    for start_p in initial_params:
        if time_left() < 30.0:
            break
        
        try:
            # First SLSQP
            res = minimize(
                _objective, start_p, method='SLSQP', bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
            
            # Then Nelder-Mead polish
            if time_left() > 25.0:
                res2 = minimize(
                    _objective, res.x, method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                )
                score_res2 = -res2.fun
                if score_res2 > best_score:
                    best_score = score_res2
                    best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Differential Evolution for global search
    if time_left() > 50.0:
        de_params, de_score = _differential_evolution_search(rng, start_time, deadline)
        if de_params is not None and de_score > best_score:
            best_score = de_score
            best_params = de_params.copy()
        
        # Polish DE result
        if best_params is not None and time_left() > 20.0:
            polished = _local_polish(best_params, start_time, deadline, max_iter_nm=2000)
            polished_score = -_objective(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_params = polished.copy()
    
    # Phase 4: More local optimization from perturbed best solution
    if best_params is not None and time_left() > 30.0:
        for i in range(5):
            if time_left() < 15.0:
                break
            try:
                noise = rng.normal(0, 0.02, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                res = minimize(
                    _objective, perturbed, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 5: Final high-precision polish
    if best_params is not None and time_left() > 10.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final fallback
    if best_params is None:
        pts = _generate_fps_initialization(rng)
        best_params = _points_to_param(pts)
    
    best_points = _param_to_points(best_params)
    return np.array(best_points)
