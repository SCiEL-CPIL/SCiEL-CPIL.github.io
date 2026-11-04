"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

_DEADLINE = time.time() + 130  # safety margin before 150s kill


def _hull_area(pts):
    """Area of convex hull of 2D points."""
    if len(pts) < 3:
        return 0.0
    try:
        return ConvexHull(pts).volume
    except Exception:
        # Fallback: use all points as hull (shoelace on sorted by angle from centroid)
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        p = pts[order]
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _min_tri_area(pts):
    """Minimum area over all triples (twice the geometric area = |cross|)."""
    n = len(pts)
    best = np.inf
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            for k in range(j + 1, n):
                pk = pts[k]
                a = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) -
                              (pj[1] - pi[1]) * (pk[0] - pi[0]))
                if a < best:
                    best = a
    return best


def _score(pts):
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _min_tri_area(pts) / ha


def _soft_min_obj(x, beta):
    """Smooth surrogate: -logsumexp of log(areas) normalized by hull area.
    Returns negative of soft-min so minimization maximizes the soft-min."""
    pts = x.reshape(N, 2)
    # Compute all triple areas (2*area = |cross|)
    areas = []
    for i in range(N):
        pi = pts[i]
        for j in range(i + 1, N):
            pj = pts[j]
            for k in range(j + 1, N):
                pk = pts[k]
                a = abs((pj[0] - pi[0]) * (pk[1] - pi[1]) -
                        (pj[1] - pi[1]) * (pk[0] - pi[0]))
                areas.append(a)
    areas = np.array(areas)
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 1e10
    log_areas = np.log(areas + 1e-15) - np.log(ha)
    m = log_areas.max()
    lse = m + np.log(np.sum(np.exp(log_areas - m)))
    return -lse / beta


def _soft_min_grad(x, beta):
    """Gradient of the soft-min objective (negative, so we maximize soft-min)."""
    pts = x.reshape(N, 2)
    n = N
    # We'll compute gradient numerically-ish but analytically for speed
    # Actually, let's use finite differences for gradient since analytic is complex
    # with hull area derivative. Use central differences.
    grad = np.zeros_like(x)
    h = 1e-5
    f0 = _soft_min_obj(x, beta)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        grad[i] = (_soft_min_obj(xp, beta) - _soft_min_obj(xm, beta)) / (2 * h)
    return grad


def _local_refine(pts, max_iters=500, tol=1e-10):
    """Coordinate descent on individual points to increase min triangle area."""
    pts = pts.copy()
    best_score = _score(pts)
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for i in range(N):
            # Try small perturbations
            for dx in [-0.05, -0.02, -0.005, 0.005, 0.02, 0.05]:
                for dy in [-0.05, -0.02, -0.005, 0.005, 0.02, 0.05]:
                    if dx == 0 and dy == 0:
                        continue
                    p2 = pts.copy()
                    p2[i] = pts[i] + np.array([dx, dy])
                    s = _score(p2)
                    if s > best_score + tol:
                        pts = p2
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            # Try smaller steps
            for i in range(N):
                for dx in [-0.01, -0.003, 0.003, 0.01]:
                    for dy in [-0.01, -0.003, 0.003, 0.01]:
                        if dx == 0 and dy == 0:
                            continue
                        p2 = pts.copy()
                        p2[i] = pts[i] + np.array([dx, dy])
                        s = _score(p2)
                        if s > best_score + tol:
                            pts = p2
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    return pts, best_score


def _softmin_optimize(pts, beta_start=20.0, beta_end=100.0, n_steps=5):
    """Multi-scale soft-min optimization using L-BFGS-B."""
    x0 = pts.flatten().copy()
    best_x = x0.copy()
    best_s = _score(pts)
    for step in range(n_steps):
        if time.time() > _DEADLINE - 10:
            break
        beta = beta_start + (beta_end - beta_start) * step / max(n_steps - 1, 1)
        try:
            res = minimize(_soft_min_obj, x0, args=(beta,),
                           method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < 1e9:
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best_x = res.x.copy()
                    x0 = res.x.copy()
        except Exception:
            pass
    return best_x.reshape(N, 2), best_s


def _normalize_unit_area(pts):
    """Scale points so convex hull area = 1."""
    ha = _hull_area(pts)
    if ha > 1e-12:
        scale = 1.0 / np.sqrt(ha)
        return pts * scale
    return pts


def _initial_grid():
    """Generate a structured initial point set on a circle/grid."""
    # Start with points on a circle (gives good spread)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_random(seed):
    """Random initial points."""
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2) * 2 - 1  # in [-1,1]^2
    return _normalize_unit_area(pts)


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 130

    best_pts = None
    best_score = 0.0

    # Phase 1: Multiple random restarts with local refinement
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 30:
            break
        if r == 0:
            pts = _initial_grid()
        else:
            pts = _initial_random(42 + r * 7)

        # Soft-min optimization
        pts, s = _softmin_optimize(pts, beta_start=15.0, beta_end=80.0, n_steps=4)

        # Local refinement
        pts, s = _local_refine(pts, max_iters=300)

        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 2: Polish the best result with tighter optimization
    if best_pts is not None and time.time() < _DEADLINE - 15:
        # High-precision soft-min polish
        x0 = best_pts.flatten().copy()
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > _DEADLINE - 5:
                break
            try:
                res = minimize(_soft_min_obj, x0, args=(beta,),
                               method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < 1e9:
                    cand = res.x.reshape(N, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand.copy()
                        x0 = cand.flatten().copy()
            except Exception:
                pass

        # Final local refinement
        if time.time() < _DEADLINE - 5:
            best_pts, best_score = _local_refine(best_pts, max_iters=500, tol=1e-12)

    # Ensure we return a valid array
    if best_pts is None:
        best_pts = _initial_grid()

    return best_pts
# EVOLVE-BLOCK-END
