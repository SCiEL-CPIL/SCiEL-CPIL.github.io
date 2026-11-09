"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

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


def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) with u+v<=1 to Cartesian.
    Point = u*A + v*B + (1-u-v)*C where A=(0,0), B=(1,0), C=(0.5,H)."""
    u = bary[0]
    v = bary[1]
    w = 1.0 - u - v
    x = u * 0.0 + v * 1.0 + w * 0.5
    y = u * 0.0 + v * 0.0 + w * H
    return np.array([x, y])


def _params_to_points(params):
    """Convert 2*N params (barycentric u,v pairs) to N Cartesian points.
    Each pair (u,v) must satisfy u>=0, v>=0, u+v<=1."""
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
        points[i] = np.array([v + (1 - u - v) * 0.5, (1 - u - v) * H])
    return points


def _objective_exact(params):
    """Minimize negative exact min-area."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_min_area(points)


def _soft_min_area(params, temperature=0.05):
    """Compute soft minimum of triangle areas using log-sum-exp for smoothness."""
    points = _params_to_points(params)
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    # Soft min via log-sum-exp: soft_min = -T * log(sum(exp(-areas/T)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    soft_min = -temperature * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    return soft_min


def _objective_soft(params, temperature=0.05):
    """Minimize negative soft-min area (smooth surrogate)."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_soft_min_area(params, temperature)


def _epigraph_objective(params, t):
    """Epigraph formulation: minimize t such that all triangle areas >= t.
    Uses smooth penalty for violations."""
    points = _params_to_points(params)
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    # Penalty for areas below t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    return t + penalty


def _objective_packing(params):
    """Minimize negative min pairwise distance (smooth packing proxy)."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_min_pairwise_dist(points)


def _lattice_initial_points():
    """Generate initial points on a triangular lattice inside the equilateral triangle.
    Uses barycentric coordinates on a grid."""
    points = []
    # Use a 4x4 grid in barycentric coords, keeping only valid ones
    # Barycentric: (u, v) where u,v >= 0, u+v <= 1
    # Map to rows of the triangle
    for row in range(4):
        for col in range(4 - row):
            u = row / 3.0
            v = col / 3.0
            if u + v <= 1.0 + 1e-9:
                pt = _bary_to_cart(np.array([u, v]))
                # Scale slightly inward to avoid boundary issues
                cx, cy = 0.5, H / 3
                pt = 0.98 * pt + 0.02 * np.array([cx, cy])
                points.append(pt)
    # We have 10 points from 4x4 triangular grid (1+2+3+4=10)
    # Add center as 11th
    points.append(np.array([0.5, H / 3]))
    return np.array(points[:N])


def _lattice_params(points):
    """Convert Cartesian points to barycentric params (u,v) for each point."""
    params = np.zeros(2 * N)
    for i in range(N):
        x, y = points[i]
        # Inverse: x = v + (1-u-v)*0.5, y = (1-u-v)*H
        # y = (1-u-v)*H => 1-u-v = y/H => u+v = 1 - y/H
        # x = v + 0.5*(1-u-v) = v + 0.5 - 0.5*u - 0.5*v = 0.5 + 0.5*v - 0.5*u
        # x - 0.5 = 0.5*(v-u) => v-u = 2*(x-0.5)
        # u+v = 1 - y/H
        s = 1.0 - y / H  # u+v
        d = 2.0 * (x - 0.5)  # v-u
        u = (s - d) / 2.0
        v = (s + d) / 2.0
        params[2*i] = max(0.0, u)
        params[2*i+1] = max(0.0, v)
    return params


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill

    best_params = None
    best_score = -np.inf

    def _elapsed():
        return time.time() - t_start

    # Phase 1: Lattice-based initial configuration + soft-min optimization
    try:
        lat_pts = _lattice_initial_points()
        lat_params = _lattice_params(lat_pts)
        
        # First optimize with soft-min (smooth) for better convergence
        res_soft = minimize(
            lambda p: _objective_soft(p, temperature=0.05),
            lat_params,
            method='Nelder-Mead',
            options={'maxiter': 15000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        # Then polish with exact
        res_exact = minimize(
            _objective_exact,
            res_soft.x,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        best_params = res_exact.x
        best_score = -_objective_exact(best_params)
    except Exception:
        pass

    # Phase 2: Differential evolution on soft-min objective for global search
    if _elapsed() < deadline - 50.0:
        try:
            bounds = [(0.0, 1.0)] * (2 * N)
            result = differential_evolution(
                lambda p: _objective_soft(p, temperature=0.03),
                bounds=bounds,
                maxiter=100,
                popsize=30,
                tol=1e-5,
                seed=42,
                polish=False,
                updating='immediate',
                init='sobol'
            )
            # Polish the DE result
            res_polish = minimize(
                _objective_exact,
                result.x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            score_de = -res_polish.fun
            if score_de > best_score:
                best_params = res_polish.x
                best_score = score_de
        except Exception:
            pass

    # Phase 3: Epigraph-based local optimization
    if _elapsed() < deadline - 30.0 and best_params is not None:
        try:
            # Use current best score as initial t
            t_init = best_score * 1.001
            # Define augmented objective with t as auxiliary variable
            def _epigraph_augmented(x):
                params = x[:-1]
                t = x[-1]
                return _epigraph_objective(params, t)
            
            x_init = np.concatenate([best_params, [t_init]])
            res_epi = minimize(
                _epigraph_augmented,
                x_init,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            epi_params = res_epi.x[:-1]
            epi_score = -_objective_exact(epi_params)
            if epi_score > best_score:
                best_params = epi_params
                best_score = epi_score
        except Exception:
            pass

    # Phase 4: Multiple restarts with soft-min bridge
    if _elapsed() < deadline - 20.0:
        rng = np.random.RandomState(777)
        n_restarts = 8
        for restart in range(n_restarts):
            if _elapsed() > deadline - 20.0:
                break
            try:
                if restart < 4:
                    # Perturb lattice points
                    lat_pts = _lattice_initial_points()
                    noise = rng.randn(N, 2) * 0.08
                    lat_pts = lat_pts + noise
                    # Clamp to triangle
                    for i in range(N):
                        if not _is_inside(lat_pts[i]):
                            lat_pts[i] = 0.9 * lat_pts[i] + 0.1 * np.array([0.5, H/3])
                    params = _lattice_params(lat_pts)
                elif restart < 6:
                    # Perturb current best
                    params = best_params + rng.randn(2 * N) * 0.03
                    # Clamp params to valid range
                    for i in range(N):
                        if params[2*i] < 0: params[2*i] = 0
                        if params[2*i+1] < 0: params[2*i+1] = 0
                        if params[2*i] + params[2*i+1] > 1:
                            s = 1.0 / (params[2*i] + params[2*i+1])
                            params[2*i] *= s
                            params[2*i+1] *= s
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
                
                # Soft-min polish (smooth, faster convergence)
                res_soft_r = minimize(
                    lambda p: _objective_soft(p, temperature=0.04),
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                # Then exact polish
                res_e = minimize(
                    _objective_exact,
                    res_soft_r.x,
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                score_r = -res_e.fun
                if score_r > best_score:
                    best_params = res_e.x
                    best_score = score_r
            except Exception:
                pass

    # Phase 5: Final high-precision polish with tighter tolerances
    if _elapsed() < deadline - 8.0 and best_params is not None:
        try:
            res_final = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            score_f = -res_final.fun
            if score_f > best_score:
                best_params = res_final.x
                best_score = score_f
        except Exception:
            pass

    # Phase 6: Ultra-fine final polish
    if _elapsed() < deadline - 3.0 and best_params is not None:
        try:
            res_uf = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
            )
            score_uf = -res_uf.fun
            if score_uf > best_score:
                best_params = res_uf.x
                best_score = score_uf
        except Exception:
            pass

    if best_params is not None:
        points = _params_to_points(best_params)
        for i in range(N):
            if not _is_inside(points[i]):
                points[i] = 0.95 * points[i] + 0.05 * np.array([0.5, H / 3])
        return points
    else:
        rng = np.random.RandomState(0)
        pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.99:
                    pts.append([u + 0.5 * v, H * v])
                    break
        return np.array(pts)

# EVOLVE-BLOCK-END
