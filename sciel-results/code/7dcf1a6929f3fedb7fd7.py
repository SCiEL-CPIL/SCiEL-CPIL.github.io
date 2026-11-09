# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 20
_T0 = time.time()
_DEADLINE = _T0 + 580.0  # safety margin before the 630 s wall-clock kill


def _normalize(pts):
    """Subtract centroid and divide by mean distance to centroid."""
    c = pts.mean(axis=0)
    p = pts - c
    md = np.linalg.norm(p, axis=1).mean()
    if md <= 1e-12:
        return p, md
    return p / md, md


def _turn_signs(p):
    """Cross products of consecutive edges (cyclic)."""
    e1 = np.roll(p, -1, axis=0) - p
    e2 = np.roll(p, -2, axis=0) - np.roll(p, -1, axis=0)
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def _is_valid(p):
    if not np.all(np.isfinite(p)):
        return False
    if p.shape != (_N, 2):
        return False
    # no repeated points
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6:
        return False
    cr = _turn_signs(p)
    if np.any(np.abs(cr) < 1e-9):
        return False
    if not (np.all(cr > 0) or np.all(cr < 0)):
        return False
    return True


def _vertex_scores(p):
    """Per-vertex equidistance scores in [0,1] (0 if any hard failure)."""
    n = p.shape[0]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    scores = np.zeros(n)
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        ds = np.sort(row)
        if ds[0] < 0.05:
            continue
        diffs = np.diff(ds)
        k = int(np.argmin(diffs))
        D = 0.5 * (ds[k] + ds[k + 1])
        if D < 0.1:
            continue
        if k == 0:
            sel = ds[0:4]
        elif k == n - 3:  # k == 17 for n=20
            sel = ds[n - 5:n - 1]
        else:
            sel = ds[k - 1:k + 3]
        raw = np.mean(np.maximum(sel / D, D / sel))
        scores[i] = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return scores


def _fourier_angular_polygon(coeffs):
    """
    coeffs: array of shape (6,) = [a1, b1, a2, b2, a3, b3]
    Returns (20, 2) array of points on unit circle with angular deviations.
    angle_i = 2*pi*i/N + 2*pi * sum_{k=1}^{3} (a_k*cos(2*pi*k*i/N) + b_k*sin(2*pi*k*i/N)) / N
    This ensures the total angle sum is 2*pi (cyclic closure).
    """
    n = _N
    i = np.arange(n)
    base_angles = 2.0 * np.pi * i / n
    # Compute deviation term: sum of Fourier modes
    dev = np.zeros(n)
    for k in range(1, 4):
        dev += coeffs[2*(k-1)] * np.cos(2.0 * np.pi * k * i / n)
        dev += coeffs[2*(k-1)+1] * np.sin(2.0 * np.pi * k * i / n)
    # Scale deviation so that sum of all angles = 2*pi
    # sum(base_angles) = 2*pi*(n-1)/2 = pi*(n-1) = 19*pi for n=20... wait
    # Actually base_angles = 2*pi*i/n, sum = 2*pi/n * n(n-1)/2 = pi*(n-1) = 19*pi
    # We need sum(angles) = 2*pi (mod 2*pi) for closure, but since we take cos/sin,
    # the absolute offset doesn't matter, only relative.
    # The key is that the polygon closes. Let's just use:
    # angles = base_angles + 2*pi * dev / n
    # This way sum(dev) might not be zero, but let's check.
    # Actually, for a closed polygon on the circle, we just need the angles to be in [0, 2*pi) sorted.
    # A simpler approach: angles = base_angles + 2*pi * dev
    # But we need to ensure they are sorted and cover 2*pi.
    # Let's use: angles = base_angles + 2*pi * dev / n
    # This keeps the mean angle the same.
    angles = base_angles + 2.0 * np.pi * dev / n
    # Ensure angles are in [0, 2*pi)
    angles = angles % (2.0 * np.pi)
    # Sort angles to ensure cyclic order (though they should already be sorted if dev is small)
    # For safety, let's sort them
    angles = np.sort(angles)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))


def _get_points_from_weights(params):
    """
    params: 6 values (Fourier coefficients). Returns (20, 2) array of points on unit circle.
    """
    return _fourier_angular_polygon(params)


def _soft_min(scores, beta=30.0):
    """Smooth minimum using log-sum-exp."""
    scores = np.asarray(scores, dtype=float)
    m = np.min(scores)
    return m - np.log(np.sum(np.exp(-beta * (scores - m)))) / beta


def _objective_params(params, beta=30.0):
    """Smooth surrogate in Fourier-parameter space (minimize)."""
    pts = _get_points_from_weights(params)
    p, md = _normalize(pts)
    if md <= 0.01:
        return 1e6
    sc = _vertex_scores(p)
    sm = _soft_min(sc, beta=beta)
    # add a tiny convexity regularizer
    cr = _turn_signs(p)
    reg = 1e-4 * np.sum(np.log1p(np.abs(cr)))
    return -sm + reg


def _objective_flat(flat, beta=30.0):
    """Smooth surrogate: -softmin of per-vertex scores in Cartesian space (minimize)."""
    p = flat.reshape(_N, 2)
    p, md = _normalize(p)
    if md <= 0.01:
        return 1e6
    sc = _vertex_scores(p)
    sm = _soft_min(sc, beta=beta)
    cr = _turn_signs(p)
    reg = 1e-4 * np.sum(np.log1p(np.abs(cr)))
    return -sm + reg


def _best_valid_from(starts, base):
    best_pts, best_metric = base, -1.0
    for s in starts:
        p, _ = _normalize(s)
        if not _is_valid(p):
            continue
        sc = _vertex_scores(p)
        m = sc.min()
        if m > best_metric:
            best_metric = m
            best_pts = p
    return best_pts, best_metric


def equidistant_points_polygon20() -> np.ndarray:
    from scipy.optimize import minimize, differential_evolution

    # ---- deterministic regular polygon baseline (score 1.0) ----
    angles = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False)
    base = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    base, _ = _normalize(base)
    best_pts, best_metric = base, float(_vertex_scores(base).min())

    # ---- Phase 1: Differential Evolution in Fourier-parameter space ----
    if time.time() < _DEADLINE - 250.0:
        try:
            res_de = differential_evolution(
                _objective_params,
                [(-0.5, 0.5)] * 6,
                maxiter=300,
                popsize=20,
                tol=1e-8,
                update_interval=1,
                seed=42,
            )
            pts = _get_points_from_weights(res_de.x)
            p, _ = _normalize(pts)
            if _is_valid(p):
                sc = _vertex_scores(p)
                m = sc.min()
                if m > best_metric:
                    best_metric = m
                    best_pts = p
        except Exception:
            pass

    # ---- Phase 2: Local refinement in Fourier-parameter space (L-BFGS-B) ----
    rng = np.random.default_rng(42)
    restarts = [np.zeros(6)]
    for i in range(30):
        if time.time() > _DEADLINE - 80.0:
            break
        x_rand = rng.uniform(-0.3, 0.3, size=6)
        restarts.append(x_rand)
        x_pert = rng.normal(0, 0.05, size=6)
        restarts.append(x_pert)

    for x_init in restarts:
        if time.time() > _DEADLINE - 50.0:
            break
        try:
            res_lm = minimize(
                _objective_params,
                x_init,
                method="L-BFGS-B",
                options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-10},
            )
            pts = _get_points_from_weights(res_lm.x)
            p, _ = _normalize(pts)
            if _is_valid(p):
                sc = _vertex_scores(p)
                m = sc.min()
                if m > best_metric:
                    best_metric = m
                    best_pts = p
        except Exception:
            pass

    # ---- Phase 3: Cartesian-space multi-restart L-BFGS-B on the smooth surrogate ----
    # Build diverse initial points
    starts = [best_pts.copy()]
    # radial perturbation: r(theta) = 1 + a1*cos + b1*sin + a2*cos2 + b2*sin2
    rng2 = np.random.default_rng(12345)
    for _ in range(14):
        a1 = rng2.uniform(-0.25, 0.25)
        b1 = rng2.uniform(-0.25, 0.25)
        a2 = rng2.uniform(-0.15, 0.15)
        b2 = rng2.uniform(-0.15, 0.15)
        th = angles
        r = 1.0 + a1 * np.cos(th) + b1 * np.sin(th) + a2 * np.cos(2 * th) + b2 * np.sin(2 * th)
        r = np.clip(r, 0.6, 1.4)
        pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        # ensure strictly convex ordering by sorting around centroid
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        pts = pts[order]
        starts.append(pts)
    # a couple of pure-noise starts
    for _ in range(2):
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1) + 0.05 * rng2.standard_normal((_N, 2))
        starts.append(pts)

    for s in starts:
        if time.time() > _DEADLINE - 40.0:
            break
        p0, _ = _normalize(s)
        try:
            res = minimize(
                _objective_flat,
                p0.ravel(),
                method="L-BFGS-B",
                options={"maxiter": 800, "maxfun": 20000, "gtol": 1e-8},
            )
        except Exception:
            continue
        p, _ = _normalize(res.x.reshape(_N, 2))
        if not _is_valid(p):
            continue
        sc = _vertex_scores(p)
        m = sc.min()
        if m > best_metric:
            best_metric = m
            best_pts = p

    # ---- Phase 4: Nelder-Mead polish on exact objective in Fourier space ----
    if time.time() < _DEADLINE - 15.0:
        nm_starts = [np.zeros(6)]
        for i in range(10):
            if time.time() > _DEADLINE - 10.0:
                break
            x_pert = rng.normal(0, 0.05, size=6)
            nm_starts.append(x_pert)

        for x_init in nm_starts:
            if time.time() > _DEADLINE - 5.0:
                break
            try:
                res_nm = minimize(
                    _objective_params,
                    x_init,
                    method="Nelder-Mead",
                    options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12},
                )
                pts = _get_points_from_weights(res_nm.x)
                p, _ = _normalize(pts)
                if _is_valid(p):
                    sc = _vertex_scores(p)
                    m = sc.min()
                    if m > best_metric:
                        best_metric = m
                        best_pts = p
            except Exception:
                pass

    # ---- high-precision polish on the best found in Cartesian space ----
    if time.time() < _DEADLINE - 10.0:
        try:
            res = minimize(
                _objective_flat,
                best_pts.ravel(),
                method="L-BFGS-B",
                options={"maxiter": 3000, "maxfun": 200000, "gtol": 1e-12},
            )
            p, _ = _normalize(res.x.reshape(_N, 2))
            if _is_valid(p):
                sc = _vertex_scores(p)
                m = sc.min()
                if m > best_metric:
                    best_metric = m
                    best_pts = p
        except Exception:
            pass

    # ---- Phase 5: Beta progression in Fourier space ----
    if time.time() < _DEADLINE - 5.0:
        for beta in [60.0, 120.0]:
            if time.time() > _DEADLINE - 2.0:
                break
            try:
                res_beta = minimize(
                    lambda x: _objective_params(x, beta=beta),
                    np.zeros(6),
                    method="L-BFGS-B",
                    options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
                )
                pts = _get_points_from_weights(res_beta.x)
                p, _ = _normalize(pts)
                if _is_valid(p):
                    sc = _vertex_scores(p)
                    m = sc.min()
                    if m > best_metric:
                        best_metric = m
                        best_pts = p
            except Exception:
                pass

    # final validity guard; fall back to the regular polygon
    if not _is_valid(best_pts):
        best_pts = base
    return best_pts


# EVOLVE-BLOCK-END
