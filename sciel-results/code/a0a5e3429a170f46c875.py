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


def _triangle_areas_vectorized(points):
    """Compute areas of all C(N,3) triangles in vectorized form."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
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


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    """Score using soft-min instead of hard min."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    """Negative soft score for minimization."""
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Seed generators ----------

def _seed_regular():
    """Regular 13-gon inscribed in unit circle."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    """Two-ring configuration: n_out on unit circle, rest at r_inner."""
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    """Three-ring configuration."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular(seed=42):
    """Quasi-regular: regular polygon with small random perturbations."""
    rng = np.random.RandomState(seed)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.02 * rng.randn(N)
    r = np.clip(r, 0.7, 1.3)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_golden_angle(seed=0):
    """Points with golden angle spacing on unit circle."""
    rng = np.random.RandomState(seed)
    ang = 2 * np.pi * rng.permutation(N) / N
    r = np.ones(N) * (1.0 + 0.01 * rng.randn(N))
    r = np.clip(r, 0.8, 1.2)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_ellipse(n, aspect, seed=0):
    """Points on an ellipse with optimized angular positions."""
    rng = np.random.RandomState(seed)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.randn(n) * 0.02
    x = np.cos(ang)
    y = aspect * np.sin(ang)
    return np.column_stack([x, y])


# ---------- Optimization routines ----------

def _annealed_soft_minimize(points, deadline, max_temp=0.3, min_temp=1e-7, n_steps=12):
    """
    Anneal the soft-min temperature and optimize at each step using L-BFGS-B.
    """
    best = points.copy()
    best_s = _score(best)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 5:
            break
        
        try:
            def obj(x):
                pts = x.reshape(N, 2)
                return _neg_soft_score(pts, temp)
            
            res = minimize(
                obj, best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-14},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def _polish_true_objective(points, deadline):
    """Polish using the true (non-smooth) objective with Powell then Nelder-Mead."""
    best = points.copy()
    best_s = _score(best)
    
    if time.time() < deadline - 4:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-14, 'fatol': 1e-14},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def _local_search(points, deadline, n_iter=1500, sigma_start=0.02, sigma_end=0.0001):
    """
    Coordinate-wise local search with adaptive step size and multi-scale steps.
    """
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(42)
    
    for it in range(n_iter):
        if time.time() >= deadline - 3:
            break
        
        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)
        
        improved = False
        for scale in [1.0, 0.5, 0.25, 0.1, 2.0, 0.05, 4.0, 0.01]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction
            
            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                improved = True
                break
    
    return best, best_s


def _active_constraint_polish(points, deadline):
    """
    Polish by focusing on the active (minimum-area) constraints.
    Identify the k smallest triangles and do targeted optimization.
    """
    best = points.copy()
    best_s = _score(best)
    areas = _triangle_areas_vectorized(best)
    min_area = np.min(areas)
    
    # Get the indices of the smallest triangles
    k = min(20, len(areas))
    small_idx = np.argsort(areas)[:k]
    
    # Try to increase the minimum by moving points that appear in small triangles
    # Get all point indices involved in the small triangles
    involved = set()
    for idx in small_idx:
        i, j, kk = _TRIPLE_IDX[idx]
        involved.add(i)
        involved.add(j)
        involved.add(kk)
    
    if not involved:
        return best, best_s
    
    involved_list = sorted(involved)
    
    # Do a local search focused on the involved points
    rng = np.random.RandomState(123)
    n_iters = 500
    for it in range(n_iters):
        if time.time() >= deadline - 2:
            break
        
        progress = it / max(n_iters - 1, 1)
        sigma = 0.01 * (0.01 ** progress)
        
        idx = rng.choice(involved_list)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)
        
        for scale in [1.0, 0.5, 0.2, 2.0]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction
            
            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break
    
    return best, best_s


def _differential_evolution_global(points, deadline):
    """
    Use differential evolution for global search with soft-min objective.
    """
    from scipy.optimize import differential_evolution
    
    def obj(x):
        pts = x.reshape(N, 2)
        s = _soft_score(pts, 0.05)
        if s <= 0:
            return 1e10
        return -s
    
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    try:
        res = differential_evolution(
            obj, bounds,
            maxiter=30, popsize=15, tol=1e-6,
            polish=False, seed=42,
            init='sobol',
        )
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > _score(points):
            return cand, s
        return points, _score(points)
    except Exception:
        return points, _score(points)


def _two_ring_optimize(deadline):
    """
    Optimize two-ring configuration: outer ring on unit circle, inner ring at radius r.
    Parameters: r (inner radius), offset (angular offset of inner ring), n_out (number outer)
    """
    best_pts = None
    best_s = -1
    
    # Try different numbers of outer points
    for n_out in [6, 7, 8, 9]:
        n_in = N - n_out
        if n_in < 1:
            continue
        
        # Optimize r and offset for this configuration
        def two_ring_params(params):
            r = np.clip(params[0], 0.1, 0.9)
            offset = params[1]
            a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
            a_in = np.linspace(offset, 2 * np.pi + offset, n_in, endpoint=False)
            outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
            inner = np.column_stack([r * np.cos(a_in), r * np.sin(a_in)])
            pts = np.vstack([outer, inner])
            return pts
        
        # Search over r and offset
        for r in np.linspace(0.3, 0.8, 6):
            for offset in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                pts = two_ring_params([r, offset])
                s = _score(pts)
                if s > best_s:
                    best_s = s
                    best_pts = pts.copy()
        
        # Refine with local optimization
        if best_pts is not None and time.time() < deadline - 5:
            # Find best r, offset near the best found
            def refine_obj(params):
                r = np.clip(params[0], 0.1, 0.9)
                offset = params[1]
                pts = two_ring_params(params)
                s = _soft_score(pts, 0.01)
                if s <= 0:
                    return 1e10
                return -s
            
            # Find initial best
            best_r, best_off = 0.5, 0.0
            best_refine_s = -1
            for r in np.linspace(0.3, 0.8, 10):
                for offset in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    pts = two_ring_params([r, offset])
                    s = _score(pts)
                    if s > best_refine_s:
                        best_refine_s = s
                        best_r = r
                        best_off = offset
            
            if time.time() < deadline - 3:
                try:
                    res = minimize(
                        refine_obj,
                        [best_r, best_off],
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
                    )
                    pts = two_ring_params(res.x)
                    s = _score(pts)
                    if s > best_s:
                        best_s = s
                        best_pts = pts.copy()
                except Exception:
                    pass
    
    if best_pts is None:
        return _seed_regular(), _score(_seed_regular())
    return best_pts, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Generate diverse initializations
    inits = []
    inits.append(_seed_regular())
    inits.append(_seed_two_ring(7, 0.5, 0.0))
    inits.append(_seed_two_ring(6, 0.4, np.pi / 6))
    inits.append(_seed_two_ring(8, 0.6, np.pi / 4))
    inits.append(_seed_two_ring(7, 0.55, np.pi / 12))
    inits.append(_seed_two_ring(9, 0.45, np.pi / 8))
    inits.append(_seed_two_ring(6, 0.5, np.pi / 3))
    inits.append(_seed_two_ring(8, 0.5, np.pi / 2))
    inits.append(_seed_three_ring(5, 4, 4, 0.5, 0.8, 0.0, np.pi / 5))
    inits.append(_seed_three_ring(6, 3, 4, 0.4, 0.7, np.pi / 6, np.pi / 3))
    inits.append(_seed_three_ring(7, 3, 3, 0.55, 0.75, np.pi / 7, np.pi / 4))
    inits.append(_seed_quasi_regular())
    inits.append(_seed_quasi_regular(seed=99))
    inits.append(_seed_golden_angle(0))
    inits.append(_seed_golden_angle(1))
    inits.append(_seed_golden_angle(7))
    inits.append(_seed_ellipse(N, 1.0, seed=0))
    inits.append(_seed_ellipse(N, 1.2, seed=1))
    inits.append(_seed_ellipse(N, 0.8, seed=2))
    
    # Add some random inits
    rng = np.random.RandomState(42)
    for _ in range(3):
        inits.append(rng.random((N, 2)))
    
    best_points = None
    best_score = -1

    # Phase 1: Quick evaluation of all seeds
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()
    
    # Phase 2: Annealed soft-min optimization on top seeds
    # Sort seeds by score and optimize the best ones
    seed_scores = [(init, _score(init)) for init in inits]
    seed_scores.sort(key=lambda x: -x[1])
    
    n_optimize = min(8, len(seed_scores))
    for i in range(n_optimize):
        if time.time() >= deadline - 30:
            break
        
        init, s_init = seed_scores[i]
        try:
            pts, s = _annealed_soft_minimize(init, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 3: Two-ring optimization (structured search)
    if time.time() < deadline - 25:
        try:
            pts, s = _two_ring_optimize(deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 4: Differential evolution global search
    if best_points is not None and time.time() < deadline - 20:
        try:
            pts, s = _differential_evolution_global(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 5: Polish the best result with true objective
    if best_points is not None:
        if time.time() < deadline - 10:
            try:
                pts, s = _polish_true_objective(best_points, deadline)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Phase 5.5: Active constraint polish
    if best_points is not None and time.time() < deadline - 8:
        try:
            pts, s = _active_constraint_polish(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 6: Multi-restart polish from perturbed versions of best
    if best_points is not None and time.time() < deadline - 8:
        rng_p = np.random.RandomState(777)
        for _ in range(3):
            if time.time() >= deadline - 6:
                break
            pert = best_points.copy() + rng_p.randn(N, 2) * 0.01
            try:
                pts, s = _polish_true_objective(pert, deadline)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Phase 7: Local search refinement
    if best_points is not None:
        if time.time() < deadline - 5:
            try:
                pts, s = _local_search(best_points, deadline)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Phase 8: Final Nelder-Mead polish
    if best_points is not None and time.time() < deadline - 1:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True},
            )
            pts = res.x.reshape(N, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _seed_regular()

    return best_points
