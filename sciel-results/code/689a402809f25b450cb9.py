import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 11
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

# Precompute all C(11,2) pair index pairs
_pair_i, _pair_j = [], []
for i in range(N):
    for j in range(i + 1, N):
        _pair_i.append(i)
        _pair_j.append(j)
_pair_i = np.array(_pair_i)
_pair_j = np.array(_pair_j)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


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


def _params_to_points(params):
    """Convert 2*N params (barycentric u,v pairs) to N Cartesian points.
    Each pair (u,v) must satisfy u>=0, v>=0, u+v<=1.
    Point = u*A + v*B + (1-u-v)*C where A=(0,0), B=(1,0), C=(0.5,H)."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        if u < 0: u = 0.0
        if v < 0: v = 0.0
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        w = 1.0 - u - v
        points[i] = np.array([v + w * 0.5, w * H])
    return points


def _points_to_params(points):
    """Convert Cartesian points to barycentric params (u,v) for each point."""
    params = np.zeros(2 * N)
    for i in range(N):
        x, y = points[i]
        # y = w * H => w = y/H
        w = y / H
        # x = v + w * 0.5 => v = x - 0.5*w
        v = x - 0.5 * w
        u = 1.0 - w - v
        params[2*i] = max(0.0, u)
        params[2*i+1] = max(0.0, v)
    return params


def _objective_exact(params):
    """Minimize negative exact min-area."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_min_area(points)


def _objective_packing(params):
    """Minimize negative min pairwise distance (smooth packing proxy)."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_min_pairwise_dist(points)


def _objective_soft_min(params, beta=50.0):
    """Smooth soft-min surrogate for min triangle area using log-sum-exp.
    Returns -soft_min(areas) which is smooth and differentiable."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    # Log-sum-exp for -min: soft_min = -logsumexp(-beta * areas) / beta
    # We want to maximize min, so minimize -soft_min
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    logsumexp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -logsumexp / beta
    return -soft_min_val


def _lattice_initial_points():
    """Generate initial points on a triangular lattice inside the equilateral triangle."""
    points = []
    for row in range(4):
        for col in range(4 - row):
            u = row / 3.0
            v = col / 3.0
            if u + v <= 1.0 + 1e-9:
                w = 1.0 - u - v
                pt = np.array([v + w * 0.5, w * H])
                # Scale slightly inward to avoid boundary issues
                cx, cy = 0.5, H / 3
                pt = 0.98 * pt + 0.02 * np.array([cx, cy])
                points.append(pt)
    # We have 10 points from 4x4 triangular grid (1+2+3+4=10)
    # Add center as 11th
    points.append(np.array([0.5, H / 3]))
    return np.array(points[:N])


def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_params = None
    best_score = -np.inf

    # Phase 1: Lattice-based initial configuration + packing optimization
    try:
        lat_pts = _lattice_initial_points()
        lat_params = _points_to_params(lat_pts)
        
        # Optimize packing (min pairwise distance) from lattice start
        res_pack = minimize(
            _objective_packing,
            lat_params,
            method='Nelder-Mead',
            options={'maxiter': 15000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        best_params = res_pack.x
        best_score = -_objective_exact(best_params)
    except Exception:
        pass

    # Phase 2: Differential evolution on soft-min objective for global search
    if time.time() < deadline - 45.0:
        try:
            bounds = [(0.0, 1.0)] * (2 * N)
            result = differential_evolution(
                lambda p: _objective_soft_min(p, beta=50.0),
                bounds=bounds,
                maxiter=60,
                popsize=20,
                tol=1e-5,
                seed=42,
                polish=False,
                updating='immediate',
                init='sobol'
            )
            score_de = -_objective_exact(result.x)
            if score_de > best_score:
                best_params = result.x
                best_score = score_de
        except Exception:
            pass

    # Phase 3: Polish with exact min-area objective from best solution
    if time.time() < deadline - 25.0:
        try:
            res3 = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score3 = -res3.fun
            if score3 > best_score:
                best_params = res3.x
                best_score = score3
        except Exception:
            pass

    # Phase 4: Multiple restarts - mix of lattice perturbations and random
    if time.time() < deadline - 15.0:
        rng = np.random.RandomState(777)
        for restart in range(8):
            if time.time() > deadline - 15.0:
                break
            try:
                if restart < 4:
                    # Perturb lattice points
                    lat_pts = _lattice_initial_points()
                    noise = rng.randn(N, 2) * 0.08
                    lat_pts = lat_pts + noise
                    for i in range(N):
                        if not _is_inside(lat_pts[i]):
                            lat_pts[i] = 0.9 * lat_pts[i] + 0.1 * np.array([0.5, H/3])
                    params = _points_to_params(lat_pts)
                else:
                    # Random barycentric
                    params = np.zeros(2 * N)
                    for i in range(N):
                        u, v = rng.random(), rng.random()
                        if u + v > 1.0:
                            s = 1.0 / (u + v)
                            u *= s
                            v *= s
                        params[2*i] = u
                        params[2*i+1] = v
                
                # Quick polish with soft-min then exact
                res_r = minimize(
                    lambda p: _objective_soft_min(p, beta=30.0),
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                score_r = -_objective_exact(res_r.x)
                if score_r > best_score:
                    best_params = res_r.x
                    best_score = score_r
            except Exception:
                pass

    # Final refinement with very tight tolerances
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            score_final = -res_final.fun
            if score_final > best_score:
                best_params = res_final.x
                best_score = score_final
        except Exception:
            pass

    if best_params is None:
        best_points = _lattice_initial_points()
    else:
        best_points = _params_to_points(best_params)
        # Ensure all points are inside
        for i in range(N):
            if not _is_inside(best_points[i]):
                best_points[i] = 0.9 * best_points[i] + 0.1 * np.array([0.5, H/3])
    
    return best_points
