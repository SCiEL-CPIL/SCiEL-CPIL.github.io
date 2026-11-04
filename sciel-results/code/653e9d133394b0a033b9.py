"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Uses simulated annealing with topological swaps and local jitter for combinatorial search,
combined with gradient-based polishing via SLSQP and Nelder-Mead.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute indices for all triples (i, j, k) with i < j < k
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]
N_TRIPLES = len(_IDX_TRIPLES)


def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_triangle_areas(points):
    """Compute all triangle areas among all triples."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _param_to_points(params):
    """Map parameters in [0, 1]^{2N} to points inside the triangle using barycentric coords."""
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    lam1 = s * (1.0 - t)
    lam2 = s * t
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V1[1] + lam2 * V2[1]
    return np.column_stack([x, y])


def _points_to_param(points):
    """Map points in the triangle to parameters (s, t)."""
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


def _hard_objective(params):
    """Hard objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)


def _score_params(params):
    """Score a parameter configuration (higher is better)."""
    points = _param_to_points(params)
    return _min_triangle_area_vectorized(points)


def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    pts = points.copy()
    # Clip x to [0, 1]
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    # Clip y to [0, H]
    pts[:, 1] = np.clip(pts[:, 1], 0.0, H)
    # For each point, if y > H * (1 - x) or y > H * x (when x > 0.5), clamp
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        if y > H:
            y = H
        if x <= 0.5:
            max_y = H * x / 0.5
        else:
            max_y = H * (1 - x) / 0.5
        if y > max_y:
            y = max_y
        pts[i] = [x, y]
    return pts


def _slsqp_optimize(params, maxiter=200, ftol=1e-9):
    """Optimize using SLSQP with hard objective."""
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(
            _hard_objective, params, method='SLSQP', bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        return res.x
    except Exception:
        return params


def _nelder_mead_polish(params, maxiter=5000):
    """Polish using Nelder-Mead."""
    try:
        res = minimize(
            _hard_objective, params, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x
    except Exception:
        return params


def _simulated_annealing(points, start_temp=0.5, end_temp=0.01, n_iter=5000, rng=None):
    """Simulated annealing with topological swaps and local jitter.
    
    Two types of moves:
    1. Local jitter: perturb one or two points slightly
    2. Topological swap: swap two points and then optimize locally
    
    Returns improved points and score.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    current = points.copy()
    current_score = _min_triangle_area_vectorized(current)
    best = current.copy()
    best_score = current_score
    
    for it in range(n_iter):
        # Cooling schedule
        alpha = it / max(n_iter - 1, 1)
        temp = start_temp * (end_temp / start_temp) ** alpha
        
        move_type = rng.random()
        if move_type < 0.6:
            # Local jitter: perturb 1-3 random points
            n_perturb = rng.randint(1, 4)
            idx = rng.choice(N, n_perturb, replace=False)
            new_points = current.copy()
            for i in idx:
                dx = rng.normal(0, 0.05)
                dy = rng.normal(0, 0.05)
                new_points[i, 0] += dx
                new_points[i, 1] += dy
            new_points = _clamp_to_triangle(new_points)
        elif move_type < 0.85:
            # Swap two points and jitter both
            i, j = rng.choice(N, 2, replace=False)
            new_points = current.copy()
            new_points[i], new_points[j] = new_points[j].copy(), new_points[i].copy()
            # Add small jitter to break symmetry
            for k in [i, j]:
                dx = rng.normal(0, 0.03)
                dy = rng.normal(0, 0.03)
                new_points[k, 0] += dx
                new_points[k, 1] += dy
            new_points = _clamp_to_triangle(new_points)
        else:
            # Global move: replace 1-2 points with new random positions
            n_replace = rng.randint(1, 3)
            idx = rng.choice(N, n_replace, replace=False)
            new_points = current.copy()
            for i in idx:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                new_points[i] = r1 * V1 + r2 * V2
            new_points = _clamp_to_triangle(new_points)
        
        new_score = _min_triangle_area_vectorized(new_points)
        
        # Accept or reject
        if new_score >= current_score:
            current = new_points
            current_score = new_score
            if new_score > best_score:
                best = new_points.copy()
                best_score = new_score
        else:
            delta = current_score - new_score
            if delta < 1e-15:
                accept = True
            else:
                accept_prob = np.exp(-delta / max(temp, 1e-15))
                accept = rng.random() < accept_prob
            if accept:
                current = new_points
                current_score = new_score
    
    return best, best_score


def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """Generate an initial configuration using Greedy Farthest Point Sampling."""
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
    """Generate structured initializations based on grid or symmetric patterns."""
    inits = []
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


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # 1. Generate diverse initializations
    n_fps_inits = 8
    for i in range(n_fps_inits):
        if time_left() < 35.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # Add structured inits
    for pts in _structured_inits(rng):
        if time_left() < 35.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    # Add random initializations
    for i in range(8):
        if time_left() < 35.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    # 2. Optimize each initialization with SLSQP
    for start_p in initial_params:
        if time_left() < 20.0:
            break
            
        try:
            refined = _slsqp_optimize(start_p, maxiter=150)
            score_res = _score_params(refined)
            if score_res > best_score:
                best_score = score_res
                best_params = refined.copy()
        except Exception:
            pass

    # 3. Run simulated annealing on the best solution found so far
    if best_params is not None and time_left() > 40.0:
        try:
            best_points = _param_to_points(best_params)
            sa_points, sa_score = _simulated_annealing(
                best_points, start_temp=0.3, end_temp=0.005, 
                n_iter=3000, rng=rng
            )
            if sa_score > best_score:
                best_score = sa_score
                best_params = _points_to_param(sa_points)
        except Exception:
            pass

    # 4. Polish with SLSQP after simulated annealing
    if best_params is not None and time_left() > 25.0:
        try:
            refined = _slsqp_optimize(best_params, maxiter=200)
            score_res = _score_params(refined)
            if score_res > best_score:
                best_score = score_res
                best_params = refined.copy()
        except Exception:
            pass

    # 5. Nelder-Mead polish
    if best_params is not None and time_left() > 15.0:
        try:
            res = _nelder_mead_polish(best_params, maxiter=3000)
            score_res = _score_params(res)
            if score_res > best_score:
                best_score = score_res
                best_params = res.copy()
        except Exception:
            pass

    # 6. Try a few more SA runs from perturbed best solution
    if best_params is not None and time_left() > 30.0:
        for i in range(3):
            if time_left() < 15.0:
                break
            try:
                best_points = _param_to_points(best_params)
                noise = rng.normal(0, 0.03, size=2*N)
                perturbed = np.clip(best_params + noise, 0.0, 1.0)
                pert_points = _param_to_points(perturbed)
                sa_points, sa_score = _simulated_annealing(
                    pert_points, start_temp=0.2, end_temp=0.005,
                    n_iter=2000, rng=rng
                )
                if sa_score > best_score:
                    best_score = sa_score
                    best_params = _points_to_param(sa_points)
            except Exception:
                pass

    # 7. Final high-precision polish
    if best_params is not None and time_left() > 10.0:
        try:
            res = _slsqp_optimize(best_params, maxiter=300)
            score_res = _score_params(res)
            if score_res > best_score:
                best_score = score_res
                best_params = res.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: Centered grid
        h = np.sqrt(3) / 2
        fallback_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
        return fallback_points

    # Convert best params to points
    best_points = _param_to_points(best_params)
    
    return best_points
