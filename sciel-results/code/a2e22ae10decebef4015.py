"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area.

Strategy:
- Parametrize the 13 points using a smooth radial basis function (RBF) expansion
  over angular coordinates: r(θ) = base_r + Σ w_k * exp(-σ_k * (θ - θ_k)²)
  This structurally enforces a smooth, convex boundary and reduces the search space
  to a small number of learnable parameters (weights, centers, widths).
- Use a smooth log-sum-exp surrogate for the minimum triangle area.
- Optimize the RBF parameters via differential evolution (global search) in the
  low-dimensional parameter space, followed by L-BFGS-B polish with annealing.
- Also retain the independent radial perturbation approach (13 DOF) as a secondary
  search axis, and a full 2D Nelder-Mead polish on the best candidate.
- The RBF representation avoids jagged local minima inherent to independent point
  optimization by enforcing global smoothness of the boundary.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 13
TIME_LIMIT = 140.0  # safety margin before 150s kill

# Precompute all triangle index triples
_TRIS = np.array(list(combinations(range(N), 3)))
_N_TRIS = _TRIS.shape[0]


def _triangle_areas(pts):
    """Vectorized computation of all C(N,3) triangle areas."""
    i, j, k = _TRIS[:, 0], _TRIS[:, 1], _TRIS[:, 2]
    p1 = pts[i]
    p2 = pts[j]
    p3 = pts[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0


def _soft_min_value(areas, beta):
    """Soft-min via log-sum-exp."""
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - log_sum / beta


def _soft_min_score(pts, beta):
    """Smooth surrogate: soft-min of triangle areas divided by hull area."""
    try:
        areas = _triangle_areas(pts)
        if len(areas) == 0:
            return 0.0
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        soft_min_val = _soft_min_value(areas, beta)
        return soft_min_val / hull_area
    except Exception:
        return 0.0


def _neg_soft_min_score(x, beta):
    pts = x.reshape(N, 2)
    return -_soft_min_score(pts, beta)


def _neg_score(x):
    pts = x.reshape(N, 2)
    return -_score(pts)


# ===================== RBF-based parametrization =====================

# Number of Gaussian basis functions
N_BASIS = 6

def _rbf_params_to_points(params, base_angles=None):
    """Convert RBF parameters to 2D points.
    
    params layout: [base_r (1), w_1..w_K (K), theta_1..theta_K (K), sigma_1..sigma_K (K)]
    Total: 1 + 3*K parameters.
    
    r(theta) = base_r + sum_k w_k * exp(-sigma_k * ((theta - theta_k) mod 2pi)^2)
    
    The points are placed at angles base_angles (regular 13-gon angles) with
    radii given by the RBF.
    """
    if base_angles is None:
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    K = N_BASIS
    base_r = params[0]
    weights = params[1:1+K]
    centers = params[1+K:1+2*K]
    sigmas = params[1+2*K:1+3*K]
    
    radii = np.empty(N)
    for i in range(N):
        theta_i = base_angles[i]
        r = base_r
        for k in range(K):
            # Circular distance
            d = theta_i - centers[k]
            d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
            r += weights[k] * np.exp(-sigmas[k] * d * d)
        radii[i] = max(r, 0.05)
    
    pts = np.column_stack([radii * np.cos(base_angles), radii * np.sin(base_angles)])
    return pts


def _rbf_objective(params, beta, base_angles=None):
    """Negative soft-min score for RBF parameters."""
    if base_angles is None:
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = _rbf_params_to_points(params, base_angles)
    return -_soft_min_score(pts, beta)


def _rbf_bounds():
    """Bounds for RBF parameters: [base_r, w_1..w_K, theta_1..theta_K, sigma_1..sigma_K]"""
    K = N_BASIS
    bounds = [(0.3, 2.0)]  # base_r
    bounds += [(-0.5, 0.5)] * K  # weights
    bounds += [(0.0, 2 * np.pi)] * K  # centers (angles)
    bounds += [(0.5, 20.0)] * K  # sigmas (widths)
    return bounds


# ===================== Independent radial parametrization =====================

def _radial_to_points(radii, base_angles=None):
    """Convert radial offsets to 2D points on a regular polygon."""
    if base_angles is None:
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([radii * np.cos(base_angles), radii * np.sin(base_angles)])
    return pts


def _radial_objective(radii, beta, base_angles=None):
    if base_angles is None:
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = _radial_to_points(radii, base_angles)
    return -_soft_min_score(pts, beta)


# ===================== Initializations =====================

def _regular_polygon(n=N, noise=0.0, rng=None):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    if noise > 0 and rng is not None:
        pts = pts + rng.normal(0, noise, (n, 2))
    return pts


def _halton_sequence(n, base):
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        i_temp = i
        while i_temp > 0:
            f /= base
            x += f * (i_temp % base)
            i_temp //= base
        result.append(x)
    return np.array(result)


def _generate_initializations():
    rng = np.random.default_rng(42)
    inits = []
    
    inits.append(_regular_polygon())
    for noise in [0.05, 0.1, 0.2, 0.3, 0.5]:
        inits.append(_regular_polygon(noise=noise, rng=rng))
    
    x = _halton_sequence(N, 2)
    y = _halton_sequence(N, 3)
    inits.append(np.column_stack([x, y]))
    
    for _ in range(5):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        inits.append(pts)
    
    for _ in range(5):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)
    
    return inits


# ===================== Optimization routines =====================

def _lbfgs_polish(pts, time_budget=15.0, start_time=None, betas=(50, 100, 200, 500)):
    """L-BFGS-B polish using soft-min surrogate with increasing beta."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    x0 = pts.flatten().copy()
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    for beta in betas:
        if time.time() > deadline - 2:
            break
        try:
            res = minimize(
                _neg_soft_min_score,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts.copy()
                x0 = res.x.copy()
        except Exception:
            break
    
    return best_pts, best_score


def _rbf_de_search(start_time, time_budget=30.0):
    """Differential evolution in the RBF parameter space (1 + 3*K = 19 dims)."""
    deadline = start_time + time_budget
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    def de_obj(params):
        return _rbf_objective(params, beta=50, base_angles=base_angles)
    
    bounds = _rbf_bounds()
    
    try:
        res = differential_evolution(
            de_obj,
            bounds=bounds,
            seed=42,
            maxiter=80,
            popsize=20,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True,
            init='latinhypercube',
            updating='immediate',
            callback=lambda xk, convergence: time.time() > deadline - 5
        )
        pts = _rbf_params_to_points(res.x, base_angles)
        s = _score(pts)
        return pts, s, res.x
    except Exception:
        return None, 0.0, None


def _rbf_lbfgs_polish(rbf_params, time_budget=15.0, start_time=None, betas=(50, 100, 200, 500)):
    """L-BFGS-B polish on RBF parameters with annealing."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    current = rbf_params.copy()
    best_pts = _rbf_params_to_points(current, base_angles)
    best_score = _score(best_pts)
    
    for beta in betas:
        if time.time() > deadline - 2:
            break
        try:
            res = minimize(
                _rbf_objective,
                current,
                args=(beta, base_angles),
                method='L-BFGS-B',
                bounds=_rbf_bounds(),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            current = res.x
            pts = _rbf_params_to_points(current, base_angles)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            break
    
    return best_pts, best_score


def _radial_optimization(start_time, time_budget=25.0):
    """Optimize radial offsets of a regular 13-gon using L-BFGS-B."""
    deadline = start_time + time_budget
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    best_pts = None
    best_score = -1
    
    rng = np.random.default_rng(7)
    radial_inits = [
        np.ones(N),
        np.ones(N) + rng.normal(0, 0.1, N),
        np.ones(N) + rng.normal(0, 0.2, N),
        np.ones(N) + rng.normal(0, 0.3, N),
        np.ones(N) + rng.uniform(-0.2, 0.2, N),
        np.linspace(0.7, 1.3, N),
    ]
    
    for radii0 in radial_inits:
        if time.time() > deadline - 5:
            break
        current_radii = radii0.copy()
        for beta in [20, 50, 100, 200, 500]:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(
                    _radial_objective,
                    current_radii,
                    args=(beta, base_angles),
                    method='L-BFGS-B',
                    bounds=[(0.1, 2.0)] * N,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                current_radii = res.x
            except Exception:
                break
        
        pts = _radial_to_points(current_radii, base_angles)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    if best_pts is None:
        return None, 0.0
    return best_pts, best_score


def _de_radial_search(start_time, time_budget=20.0):
    """Differential evolution in the 13-D radial space."""
    deadline = start_time + time_budget
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    def de_obj(radii):
        return _radial_objective(radii, beta=50, base_angles=base_angles)
    
    try:
        res = differential_evolution(
            de_obj,
            bounds=[(0.3, 1.8)] * N,
            seed=42,
            maxiter=60,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True,
            init='latinhypercube',
            updating='immediate',
            callback=lambda xk, convergence: time.time() > deadline - 5
        )
        pts = _radial_to_points(res.x, base_angles)
        s = _score(pts)
        return pts, s
    except Exception:
        return None, 0.0


def _targeted_polish(pts, time_budget=10.0, start_time=None):
    """Targeted polish: optimize points involved in worst triangles."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    areas = _triangle_areas(best_pts)
    k_worst = min(5, len(areas))
    worst_indices = np.argsort(areas)[:k_worst]
    
    # Map triangle index to point indices
    def tri_idx_to_pts(tri_idx):
        count = 0
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                for k in range(j + 1, N):
                    if count == tri_idx:
                        return (i, j, k)
                    count += 1
        return (0, 1, 2)
    
    involved_points = set()
    for wi in worst_indices:
        i, j, k = tri_idx_to_pts(wi)
        involved_points.add(i)
        involved_points.add(j)
        involved_points.add(k)
    
    for pt_idx in sorted(involved_points):
        if time.time() > deadline - 2:
            break
        
        fixed_pts = best_pts.copy()
        x0 = fixed_pts[pt_idx].copy()
        
        def obj_single(x, idx=pt_idx, fp=fixed_pts):
            trial = fp.copy()
            trial[idx] = x
            return -_score(trial)
        
        try:
            res = minimize(
                obj_single,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if -res.fun > best_score + 1e-12:
                best_pts[pt_idx] = res.x
                best_score = -res.fun
        except Exception:
            pass
    
    return best_pts, best_score


def _nelder_mead_polish(pts, time_budget=8.0, start_time=None):
    """Nelder-Mead on full 2N-dimensional space."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget
    
    x0 = pts.flatten().copy()
    try:
        result = minimize(
            _neg_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return result.x.reshape(N, 2), -result.fun
    except Exception:
        return pts, _score(pts)


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    inits = _generate_initializations()
    
    best_pts = None
    best_score = -1
    
    # Evaluate all initializations
    for i, init in enumerate(inits):
        if time.time() > deadline - 50:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    if best_pts is None:
        best_pts = _regular_polygon()
        best_score = _score(best_pts)
    
    # Phase 1: RBF differential evolution (19-D parameter space)
    remaining = deadline - 40 - time.time()
    if remaining > 20:
        rbf_pts, rbf_score, rbf_params = _rbf_de_search(start_time, time_budget=min(remaining, 30.0))
        if rbf_pts is not None:
            if rbf_score > best_score:
                best_score = rbf_score
                best_pts = rbf_pts
            # Polish RBF result with L-BFGS-B on RBF params
            if rbf_params is not None and time.time() < deadline - 20:
                pts_pol, score_pol = _rbf_lbfgs_polish(
                    rbf_params,
                    time_budget=min(deadline - 20 - time.time(), 12.0),
                    start_time=start_time,
                    betas=(50, 100, 200, 500)
                )
                if score_pol > best_score:
                    best_score = score_pol
                    best_pts = pts_pol
            # Also polish in full 2D space
            if time.time() < deadline - 15:
                pts_2d, score_2d = _lbfgs_polish(
                    rbf_pts,
                    time_budget=min(deadline - 15 - time.time(), 10.0),
                    start_time=start_time,
                    betas=(100, 200, 500)
                )
                if score_2d > best_score:
                    best_score = score_2d
                    best_pts = pts_2d
    
    # Phase 2: Radial optimization (13-D)
    remaining = deadline - 25 - time.time()
    if remaining > 15:
        rad_pts, rad_score = _radial_optimization(start_time, time_budget=min(remaining, 20.0))
        if rad_pts is not None and rad_score > best_score:
            best_score = rad_score
            best_pts = rad_pts
    
    # Phase 3: L-BFGS-B polish on best candidates
    candidates_to_polish = [best_pts]
    scored_inits = []
    for init in inits:
        s = _score(init)
        scored_inits.append((s, init))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    for s, init in scored_inits[:3]:
        if s > best_score * 0.6:
            candidates_to_polish.append(init)
    
    for cand in candidates_to_polish:
        if time.time() > deadline - 20:
            break
        remaining = deadline - 15 - time.time()
        if remaining < 5:
            break
        pts_opt, score_opt = _lbfgs_polish(
            cand,
            time_budget=min(remaining, 10.0),
            start_time=start_time,
            betas=(50, 100, 200, 500)
        )
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Phase 4: DE in radial space
    remaining = deadline - 15 - time.time()
    if remaining > 12:
        de_pts, de_score = _de_radial_search(start_time, time_budget=min(remaining, 15.0))
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_pts = de_pts
        if de_pts is not None and time.time() < deadline - 10:
            pts_opt, score_opt = _lbfgs_polish(
                de_pts,
                time_budget=min(deadline - 10 - time.time(), 8.0),
                start_time=start_time,
                betas=(100, 300, 500)
            )
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt
    
    # Phase 5: Targeted polish
    remaining = deadline - 8 - time.time()
    if remaining > 5:
        best_pts, best_score = _targeted_polish(
            best_pts,
            time_budget=min(remaining, 8.0),
            start_time=start_time
        )
    
    # Phase 6: Final Nelder-Mead polish
    remaining = deadline - 3 - time.time()
    if remaining > 3:
        best_pts, best_score = _nelder_mead_polish(
            best_pts,
            time_budget=min(remaining, 5.0),
            start_time=start_time
        )
    
    # Final safety check
    if time.time() > deadline - 1:
        return best_pts
    
    final_score = _score(best_pts)
    if final_score > best_score:
        best_score = final_score
    
    return best_pts
