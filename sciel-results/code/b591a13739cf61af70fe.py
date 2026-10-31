import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triple indices for vectorized triangle area computation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

# Fixed angles for radial parameterization
_FIXED_ANGLES = 2 * np.pi * np.arange(N) / N
_FIXED_COS = np.cos(_FIXED_ANGLES)
_FIXED_SIN = np.sin(_FIXED_ANGLES)


def _radial_to_points(radii):
    """Convert N radii to N points at fixed angles."""
    return np.column_stack([radii * _FIXED_COS, radii * _FIXED_SIN])


def _triangle_areas_vectorized(points):
    """Compute areas of all C(N,3) triangles in vectorized form."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area_from_radii(radii):
    """Compute convex hull area from radii at fixed angles (assumes convex position)."""
    pts = _radial_to_points(radii)
    # For points in convex position ordered by angle, hull area = 0.5 * sum of cross products
    # Use the shoelace formula on the ordered points
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return float(area)


def _score_from_radii(radii):
    """Score = min triangle area / convex hull area, from radii."""
    if np.any(radii < 1e-10):
        return 0.0
    hull_area = _hull_area_from_radii(radii)
    if hull_area < 1e-12:
        return 0.0
    pts = _radial_to_points(radii)
    areas = _triangle_areas_vectorized(pts)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = float(hull.volume)
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    """Negative score for minimization."""
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _penalty_objective_radial(radii, t):
    """
    Penalty objective: maximize t such that all triangle areas >= t * hull_area.
    Returns a penalty value to minimize.
    """
    if np.any(radii < 1e-10):
        return 1e10
    hull_area = _hull_area_from_radii(radii)
    if hull_area < 1e-12:
        return 1e10
    pts = _radial_to_points(radii)
    areas = _triangle_areas_vectorized(pts)
    # We want all areas >= t * hull_area
    # Penalty = sum of violations + regularization
    violations = np.maximum(0, t * hull_area - areas)
    penalty = np.sum(violations ** 2)
    # Small regularization to keep radii positive and bounded
    reg = 1e-6 * np.sum((radii - 1.0) ** 2)
    return penalty + reg


def _penalty_optimize_radial(radii_init, deadline, max_iter=500):
    """
    Optimize radii using penalty method with increasing t.
    """
    rng = np.random.RandomState(42)
    best_radii = radii_init.copy()
    best_score = _score_from_radii(best_radii)
    if best_score <= 0:
        best_score = 1e-15
    
    # Start with a small t and increase
    t_start = 0.01
    t_end = 0.5
    n_t_steps = 8
    
    for t_idx in range(n_t_steps):
        if time.time() >= deadline - 15:
            break
        
        t = t_start + (t_end - t_start) * t_idx / max(n_t_steps - 1, 1)
        
        # Optimize radii for this t
        try:
            res = minimize(
                lambda r: _penalty_objective_radial(r, t),
                best_radii,
                method='L-BFGS-B',
                bounds=[(0.3, 2.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            cand_radii = res.x
            s = _score_from_radii(cand_radii)
            if s > best_score:
                best_score = s
                best_radii = cand_radii.copy()
        except Exception:
            pass
        
        # Also try Powell
        if time.time() < deadline - 10:
            try:
                res = minimize(
                    lambda r: _penalty_objective_radial(r, t),
                    best_radii,
                    method='Powell',
                    bounds=[(0.3, 2.0)] * N,
                    options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand_radii = res.x
                s = _score_from_radii(cand_radii)
                if s > best_score:
                    best_score = s
                    best_radii = cand_radii.copy()
            except Exception:
                pass
    
    return best_radii, best_score


def _simulated_annealing_radii(radii_init, deadline, n_restarts=3):
    """
    Simulated annealing on the N radius parameters.
    """
    rng = np.random.RandomState(2024)
    best_radii = radii_init.copy()
    best_s = _score_from_radii(best_radii)
    if best_s <= 0:
        best_s = 1e-15

    T_start = 0.1
    T_end = 1e-6
    n_steps = 2000

    for restart in range(n_restarts):
        if time.time() >= deadline - 20:
            break
        
        cur_radii = radii_init.copy() + rng.randn(N) * 0.05 * (restart + 1)
        cur_radii = np.clip(cur_radii, 0.3, 2.0)
        cur_s = _score_from_radii(cur_radii)
        if cur_s <= 0:
            cur_s = 1e-15
        cur_f = -cur_s

        for step in range(n_steps):
            if time.time() > deadline - 15:
                break

            alpha = step / max(n_steps - 1, 1)
            T = T_start * (T_end / T_start) ** alpha

            # Propose a perturbation on a random subset of radii
            n_pert = rng.randint(1, 4)
            idx = rng.choice(N, size=n_pert, replace=False)
            scale = 0.05 * (1.0 - 0.5 * alpha)
            new_radii = cur_radii.copy()
            new_radii[idx] += rng.randn(n_pert) * scale
            new_radii = np.clip(new_radii, 0.3, 2.0)

            new_s = _score_from_radii(new_radii)
            if new_s <= 0:
                new_s = 1e-15
            new_f = -new_s

            delta_f = new_f - cur_f
            if delta_f < 0 or rng.random() < np.exp(-delta_f / max(T, 1e-12)):
                cur_f = new_f
                cur_s = new_s
                cur_radii = new_radii
                if cur_s > best_s:
                    best_s = cur_s
                    best_radii = cur_radii.copy()

    return best_radii, best_s


def _local_search_radii(radii, deadline, n_iter=1000, sigma_start=0.02, sigma_end=0.0003):
    """
    Coordinate-wise local search on radii with adaptive step size.
    """
    best = radii.copy()
    best_s = _score_from_radii(best)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if time.time() >= deadline - 8:
            break
        
        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        idx = rng.randint(N)
        for scale in [1.0, 0.5, 0.2, 0.1, 2.0, 0.05]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step
            cand = np.clip(cand, 0.3, 2.0)
            
            s = _score_from_radii(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break
    
    return best, best_s


def _seed_regular_radii():
    """Regular N-gon: all radii = 1."""
    return np.ones(N)


def _seed_perturbed_radii(seed=42):
    """Perturbed regular polygon."""
    rng = np.random.RandomState(seed)
    r = 1.0 + 0.05 * rng.randn(N)
    return np.clip(r, 0.5, 1.5)


def _seed_alternating_radii():
    """Alternating radii pattern."""
    r = np.ones(N)
    r[::2] = 1.1
    r[1::2] = 0.9
    return r


def _seed_three_pattern_radii():
    """Three-level pattern."""
    r = np.ones(N)
    r[::3] = 1.15
    r[1::3] = 0.9
    r[2::3] = 1.0
    return r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    best_radii = None
    best_score = -1

    # ---------- Phase A: Penalty optimization from multiple seeds ----------
    seeds = [
        _seed_regular_radii(),
        _seed_perturbed_radii(42),
        _seed_perturbed_radii(7),
        _seed_alternating_radii(),
        _seed_three_pattern_radii(),
    ]
    
    for i, seed_r in enumerate(seeds):
        if time.time() >= deadline - 40:
            break
        try:
            r_opt, s = _penalty_optimize_radial(seed_r, deadline)
            if s > best_score:
                best_score = s
                best_radii = r_opt.copy()
        except Exception:
            pass

    # ---------- Phase B: Simulated annealing from best and seeds ----------
    sa_inits = []
    if best_radii is not None:
        sa_inits.append(best_radii.copy())
    sa_inits.append(_seed_regular_radii())
    sa_inits.append(_seed_perturbed_radii(99))
    
    for r_init in sa_inits:
        if time.time() >= deadline - 20:
            break
        try:
            r_opt, s = _simulated_annealing_radii(r_init, deadline, n_restarts=2)
            if s > best_score:
                best_score = s
                best_radii = r_opt.copy()
        except Exception:
            pass

    # ---------- Phase C: Local search polish ----------
    if best_radii is not None and time.time() < deadline - 10:
        try:
            r_opt, s = _local_search_radii(best_radii, deadline, n_iter=800)
            if s > best_score:
                best_score = s
                best_radii = r_opt.copy()
        except Exception:
            pass

    # ---------- Phase D: Final fine-tuning with Nelder-Mead on full 2D points ----------
    if best_radii is not None and time.time() < deadline - 5:
        pts = _radial_to_points(best_radii)
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            cand_pts = res.x.reshape(N, 2)
            s = _score(cand_pts)
            if s > best_score:
                best_score = s
                # Convert back to radii for consistency
                cx, cy = np.mean(cand_pts[:, 0]), np.mean(cand_pts[:, 1])
                # Use the radial projection
                radii_new = np.sqrt((cand_pts[:, 0] - cx) ** 2 + (cand_pts[:, 1] - cy) ** 2)
                best_radii = radii_new
                best_score = s
        except Exception:
            pass

    # ---------- Fallback ----------
    if best_radii is None:
        best_radii = _seed_regular_radii()

    return _radial_to_points(best_radii)
