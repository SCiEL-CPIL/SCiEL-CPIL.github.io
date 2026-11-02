"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 13
DEADLINE = 145.0  # safety margin before 150s kill

# Pre-compute all C(13,3)=286 triple indices for fast vectorized scoring
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)
_num_triples = len(_triple_i)


def _triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area


def _logsumexp_surrogate(pts, temperature=0.01):
    """
    Smooth surrogate: -logsumexp(-areas/temperature) / hull_area
    This is differentiable and approximates min(areas)/hull_area.
    As temperature -> 0, it converges to the exact objective.
    """
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 1e10
    
    areas = _triangle_areas(pts)
    # Use log-sum-exp for numerical stability
    scaled = areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    # -logsumexp(-areas/T) = -(max(-areas/T) + log(sum(exp(-areas/T - max))))
    # Actually: -logsumexp(-x) where x = areas/T
    # = -(max(-x) + log(sum(exp(-x - max(-x)))))
    # = -(-max(x) + log(sum(exp(x - max(x)))))  ... wait let me redo
    
    # We want a smooth approximation of min(areas)
    # min(x) ≈ -1/T * log(sum(exp(-x/T)))
    # So: -min(areas)/hull_area ≈ (1/T) * log(sum(exp(-areas/T))) / hull_area
    # For minimization, we minimize: log(sum(exp(-areas/T))) / (T * hull_area)
    
    neg_scaled = -areas / temperature
    max_neg = np.max(neg_scaled)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))
    
    return log_sum_exp / (temperature * hull_area)


def _neg_score(params):
    """Negative score for minimization. params is 1D array of length 2*N."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)


def _neg_surrogate(params, temperature=0.01):
    """Negative surrogate for minimization."""
    pts = params.reshape(N, 2)
    return _logsumexp_surrogate(pts, temperature)


def _regular_polygon(n_points, radius=1.0, center=(0.0, 0.0)):
    """Generate vertices of a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    return pts


def _make_two_layer(n_outer=9, n_inner=4, r_outer=1.0, r_inner=0.5, angle_offset=0.0):
    """Two concentric rings: outer regular polygon + inner regular polygon."""
    outer = _regular_polygon(n_outer, radius=r_outer)
    inner = _regular_polygon(n_inner, radius=r_inner, center=(0, 0))
    # Rotate inner ring by half-step to interleave
    if n_inner > 1:
        rot_angle = angle_offset + np.pi / n_inner
        cos_r = np.cos(rot_angle)
        sin_r = np.sin(rot_angle)
        rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        inner = inner @ rot.T
    pts = np.vstack([outer, inner])
    return pts[:N]


def _make_three_layer(n1=5, n2=4, n3=4, r1=1.0, r2=0.6, r3=0.3):
    """Three concentric rings."""
    ring1 = _regular_polygon(n1, radius=r1)
    ring2 = _regular_polygon(n2, radius=r2, center=(0, 0))
    ring3 = _regular_polygon(n3, radius=r3, center=(0, 0))
    # Rotate rings to interleave
    if n2 > 1:
        rot2 = np.pi / n2
        c2, s2 = np.cos(rot2), np.sin(rot2)
        ring2 = ring2 @ np.array([[c2, -s2], [s2, c2]]).T
    if n3 > 1:
        rot3 = np.pi / n3
        c3, s3 = np.cos(rot3), np.sin(rot3)
        ring3 = ring3 @ np.array([[c3, -s3], [s3, c3]]).T
    pts = np.vstack([ring1, ring2, ring3])
    return pts[:N]


def _normalize_to_unit_area(pts):
    """Scale points so convex hull has unit area."""
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        return pts * scale
    return pts


def _optimize_from_init(init_pts, t_start, method='surrogate_then_exact'):
    """
    Multi-phase optimization:
    1. Optimize the smooth surrogate with Powell/L-BFGS-B
    2. Refine with exact objective
    """
    t0 = t_start
    best_params = init_pts.flatten().copy()
    best_score = _score_points(init_pts)
    
    # Phase 1: Surrogate optimization (smooth, differentiable)
    if (time.time() - t0) < (DEADLINE - 30.0):
        for temp in [0.05, 0.02, 0.01, 0.005]:
            if (time.time() - t0) > (DEADLINE - 25.0):
                break
            try:
                res = minimize(
                    _neg_surrogate,
                    best_params,
                    args=(temp,),
                    method='Powell',
                    options={'maxiter': 200, 'maxfun': 5000, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                candidate = res.x.reshape(N, 2)
                score = _score_points(candidate)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 2: Exact objective refinement with L-BFGS-B (using finite differences)
    if (time.time() - t0) < (DEADLINE - 15.0):
        try:
            res2 = minimize(
                _neg_score,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-12}
            )
            candidate = res2.x.reshape(N, 2)
            score = _score_points(candidate)
            if score > best_score:
                best_score = score
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Powell on exact objective (handles non-differentiability)
    if (time.time() - t0) < (DEADLINE - 10.0):
        try:
            res3 = minimize(
                _neg_score,
                best_params,
                method='Powell',
                options={'maxiter': 100, 'maxfun': 5000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            candidate = res3.x.reshape(N, 2)
            score = _score_points(candidate)
            if score > best_score:
                best_score = score
                best_params = res3.x.copy()
        except Exception:
            pass
    
    return best_params, best_score


def construct():
    t_start = time.time()
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Regular 13-gon
    poly13 = _regular_polygon(N, radius=1.0)
    inits.append(_normalize_to_unit_area(poly13))
    
    # 2. Two-layer: 9 outer + 4 inner
    two_layer = _make_two_layer(n_outer=9, n_inner=4, r_outer=1.0, r_inner=0.5)
    inits.append(_normalize_to_unit_area(two_layer))
    
    # 3. Two-layer: 8 outer + 5 inner
    two_layer2 = _make_two_layer(n_outer=8, n_inner=5, r_outer=1.0, r_inner=0.45, angle_offset=0.1)
    inits.append(_normalize_to_unit_area(two_layer2))
    
    # 4. Three-layer: 5+4+4
    three_layer = _make_three_layer(n1=5, n2=4, n3=4, r1=1.0, r2=0.6, r3=0.3)
    inits.append(_normalize_to_unit_area(three_layer))
    
    # 5. Three-layer: 6+4+3
    three_layer2 = _make_three_layer(n1=6, n2=4, n3=3, r1=1.0, r2=0.55, r3=0.25)
    inits.append(_normalize_to_unit_area(three_layer2))
    
    # 6. Two-layer: 7 outer + 6 inner
    two_layer3 = _make_two_layer(n_outer=7, n_inner=6, r_outer=1.0, r_inner=0.55, angle_offset=0.2)
    inits.append(_normalize_to_unit_area(two_layer3))
    
    # 7. Random initialization (seeded)
    rng = np.random.RandomState(123)
    rand_pts = rng.uniform(-1, 1, size=(N, 2))
    inits.append(_normalize_to_unit_area(rand_pts))
    
    # 8. Regular 13-gon with slight perturbation
    poly13_pert = _regular_polygon(N, radius=1.0)
    poly13_pert += rng.normal(0, 0.05, size=(N, 2))
    inits.append(_normalize_to_unit_area(poly13_pert))
    
    # Optimize from each initialization, tracking best
    best_params = None
    best_score = 0.0
    
    for idx, init in enumerate(inits):
        if (time.time() - t_start) > (DEADLINE - 20.0):
            break
        try:
            params, score = _optimize_from_init(init, t_start)
            if score > best_score:
                best_score = score
                best_params = params
        except Exception:
            pass
    
    # If no good result yet, try differential evolution
    if (time.time() - t_start) < (DEADLINE - 30.0) and (best_score < 45.0 or best_params is None):
        try:
            # Use the best initialization as starting point for DE
            de_init = best_params if best_params is not None else inits[0].flatten()
            
            def de_objective(params):
                pts = params.reshape(N, 2)
                # Use surrogate for DE (faster, smoother)
                return _logsumexp_surrogate(pts, temperature=0.03)
            
            bounds = [(-1.5, 1.5)] * (2 * N)
            
            result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=20,
                popsize=12,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='deferred',
                workers=1
            )
            
            if result.x is not None:
                candidate = result.x.reshape(N, 2)
                score = _score_points(candidate)
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
                
                # Refine DE result
                if (time.time() - t_start) < (DEADLINE - 15.0):
                    try:
                        params, score2 = _optimize_from_init(candidate, t_start)
                        if score2 > best_score:
                            best_score = score2
                            best_params = params
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Final refinement if time permits
    if best_params is not None and (time.time() - t_start) < (DEADLINE - 8.0):
        try:
            final_pts = best_params.reshape(N, 2)
            # One more Powell pass on exact objective
            res_final = minimize(
                _neg_score,
                best_params,
                method='Powell',
                options={'maxiter': 50, 'maxfun': 2000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            score_final = _score_points(res_final.x.reshape(N, 2))
            if score_final > best_score:
                best_score = score_final
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Safety: if somehow we have no valid result, return regular polygon
    if best_params is None:
        best_params = inits[0].flatten()
    
    return best_params.reshape(N, 2)
