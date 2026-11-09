"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing

H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_triangles = len(_idx_i)

# Precompute all C(11,2) pair index pairs
_pair_i, _pair_j = [], []
for i in range(N):
    for j in range(i + 1, N):
        _pair_i.append(i)
        _pair_j.append(j)
_pair_i = np.array(_pair_i)
_pair_j = np.array(_pair_j)
_n_pairs = len(_pair_i)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _soft_min_area(points, beta=50.0):
    """Compute soft-min of triangle areas using log-sum-exp for smoothness."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    soft_min = -max_a - (1.0 / beta) * np.log(np.sum(exp_terms))
    return soft_min


def _min_pairwise_dist(points):
    """Compute minimum pairwise distance among all C(N,2) pairs."""
    p1 = points[_pair_i]
    p2 = points[_pair_j]
    d = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
    return d.min()


def _is_inside(p):
    """Check if point p=(x,y) is inside the equilateral triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True


def _symmetric_params_to_points(params):
    """Map 11 parameters to 11 points with reflection symmetry about x=0.5.
    params[0]: y of center point (x=0.5)
    params[1:]: for 5 pairs, (x_i, y_i) where x_i in [0, 0.5], y_i in [0, H]
    Point i left: (x_i, y_i), right: (1-x_i, y_i)
    """
    points = np.zeros((N, 2))
    # Center point
    points[0, 0] = 0.5
    points[0, 1] = np.clip(params[0], 0.0, H)
    
    # 5 pairs
    for i in range(5):
        x = np.clip(params[1 + 2*i], 0.0, 0.5)
        y = np.clip(params[2 + 2*i], 0.0, H)
        # Clamp y to triangle boundary for this x
        max_y = np.sqrt(3) * min(x, 1.0 - x)
        if y > max_y:
            y = max_y
        points[1 + 2*i, 0] = x
        points[1 + 2*i, 1] = y
        points[2 + 2*i, 0] = 1.0 - x
        points[2 + 2*i, 1] = y
    return points


def _objective_symmetric(params):
    """Minimize negative exact min-area for symmetric config."""
    points = _symmetric_params_to_points(params)
    return -_min_area(points)


def _objective_soft_symmetric(params, beta=50.0):
    """Minimize negative soft-min area for symmetric config."""
    points = _symmetric_params_to_points(params)
    return -_soft_min_area(points, beta)


def _objective_packing_symmetric(params):
    """Minimize negative min pairwise distance for symmetric config."""
    points = _symmetric_params_to_points(params)
    return -_min_pairwise_dist(points)


def _symmetric_bounds():
    """Bounds for symmetric parameters: y_center, then (x,y) for 5 pairs."""
    bounds = [(0.0, H)]
    for _ in range(5):
        bounds.append((0.0, 0.5))
        bounds.append((0.0, H))
    return bounds


def _lattice_symmetric_params():
    """Generate initial symmetric params from a lattice-like pattern."""
    params = np.zeros(11)
    # Center point near centroid
    params[0] = H / 3.0
    
    # 5 pairs arranged in roughly symmetric rows
    # Row 1 (top): 1 pair near top
    params[1] = 0.25   # x of left point
    params[2] = H * 0.7
    
    # Row 2: 2 pairs
    params[3] = 0.15
    params[4] = H * 0.5
    params[5] = 0.35
    params[6] = H * 0.5
    
    # Row 3: 2 pairs
    params[7] = 0.1
    params[8] = H * 0.25
    params[9] = 0.4
    params[10] = H * 0.25
    
    return params


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin for 150s limit

    best_params = None
    best_score = -np.inf
    bounds = _symmetric_bounds()
    
    def _update_best(params, score):
        nonlocal best_params, best_score
        if score > best_score:
            best_params = params.copy()
            best_score = score

    # Phase 1: Differential Evolution with soft-min objective (global search)
    try:
        result_de = differential_evolution(
            lambda p: _objective_soft_symmetric(p, beta=50.0),
            bounds=bounds,
            maxiter=200,
            popsize=30,
            tol=1e-8,
            seed=42,
            polish=False,
            updating='immediate',
            init='sobol'
        )
        score_de = -_objective_symmetric(result_de.x)
        _update_best(result_de.x, score_de)
    except Exception:
        pass

    # Phase 2: Dual annealing for additional global exploration
    if time.time() < deadline - 40.0:
        try:
            result_da = dual_annealing(
                _objective_symmetric,
                bounds=bounds,
                maxiter=500,
                seed=123,
                restarts=5,
                polish=True
            )
            score_da = -result_da.fun
            _update_best(result_da.x, score_da)
        except Exception:
            pass

    # Phase 3: Nelder-Mead polish from best solution
    if time.time() < deadline - 30.0 and best_params is not None:
        try:
            res_nm = minimize(
                _objective_symmetric,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_nm = -res_nm.fun
            _update_best(res_nm.x, score_nm)
        except Exception:
            pass

    # Phase 4: Multiple restarts with perturbation
    if time.time() < deadline - 20.0:
        rng = np.random.RandomState(777)
        n_restarts = 15
        for restart in range(n_restarts):
            if time.time() > deadline - 20.0:
                break
            try:
                if restart < 5 and best_params is not None:
                    # Perturb best solution
                    noise = rng.randn(11) * 0.05
                    start = best_params + noise
                    for i, b in enumerate(bounds):
                        start[i] = np.clip(start[i], b[0], b[1])
                elif restart < 10:
                    # Lattice-based start
                    start = _lattice_symmetric_params()
                    noise = rng.randn(11) * 0.1
                    start = start + noise
                    for i, b in enumerate(bounds):
                        start[i] = np.clip(start[i], b[0], b[1])
                else:
                    # Random start
                    start = np.array([rng.uniform(b[0], b[1]) for b in bounds])
                
                # Quick Nelder-Mead polish
                res_r = minimize(
                    _objective_symmetric,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                score_r = -res_r.fun
                _update_best(res_r.x, score_r)
            except Exception:
                pass

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5.0 and best_params is not None:
        try:
            res_final = minimize(
                _objective_symmetric,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            score_f = -res_final.fun
            _update_best(res_final.x, score_f)
        except Exception:
            pass

    # Convert best params to points
    if best_params is not None:
        points = _symmetric_params_to_points(best_params)
        # Verify all points are inside; if not, clamp
        for i in range(N):
            if not _is_inside(points[i]):
                # Move toward centroid
                cx, cy = 0.5, H / 3
                points[i] = 0.9 * points[i] + 0.1 * np.array([cx, cy])
                if not _is_inside(points[i]):
                    points[i] = np.array([cx, cy])
        return points
    else:
        # Fallback: random points
        rng = np.random.RandomState(0)
        pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.99:
                    x = u + 0.5 * v
                    y = H * v
                    if _is_inside(np.array([x, y])):
                        pts.append([x, y])
                        break
        return np.array(pts)

# EVOLVE-BLOCK-END
