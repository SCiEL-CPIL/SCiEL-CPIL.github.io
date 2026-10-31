import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
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
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _build_two_ring(n_out, r_inner, angle_offset, radial_pert=None, angular_pert=None):
    """
    Two-ring configuration:
      Outer ring: n_out points at radius 1.0, angles 0, 2pi/n_out, ...
      Inner ring: n_in points at radius r_inner, angles angle_offset, angle_offset + 2pi/n_in, ...
    Optional per-point radial/angular perturbations (arrays of length n_out / n_in).
    """
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)

    r_out = np.ones(n_out)
    r_in = np.full(n_in, r_inner)

    if radial_pert is not None:
        rp_out, rp_in = radial_pert[:n_out], radial_pert[n_out:]
        r_out = r_out * (1.0 + rp_out)
        r_in = r_in * (1.0 + rp_in)
    if angular_pert is not None:
        ap_out, ap_in = angular_pert[:n_out], angular_pert[n_out:]
        a_out = a_out + ap_out
        a_in = a_in + ap_in

    outer = np.column_stack([r_out * np.cos(a_out), r_out * np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([outer, inner])


def _build_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _build_regular_perturbed(base, radial_pert, angular_pert):
    """Base regular N-gon with per-point radial and angular perturbations."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + angular_pert
    radii = 1.0 + radial_pert
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _generate_grid_points():
    """
    Generate a 4x4 grid of 16 points, remove one corner to get 13 points.
    Center the grid at the origin.
    The grid spacing is 1.
    Points: (-1.5, -1.5), ..., (1.5, 1.5)
    Remove (1.5, 1.5) for example.
    """
    coords = np.array([
        [-1.5, -1.5], [-0.5, -1.5], [0.5, -1.5], [1.5, -1.5],
        [-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, -0.5],
        [-1.5, 0.5],  [-0.5, 0.5],  [0.5, 0.5],  [1.5, 0.5],
        [-1.5, 1.5],  [-0.5, 1.5],  [0.5, 1.5],  [1.5, 1.5]
    ])
    # Remove the last point (1.5, 1.5)
    coords = coords[:-1]
    return coords


def _apply_affine_transform(points, a11, a12, a21, a22, tx, ty):
    """
    Apply affine transformation:
    x' = a11*x + a12*y + tx
    y' = a21*x + a22*y + ty
    """
    A = np.array([[a11, a12], [a21, a22]])
    t = np.array([tx, ty])
    return points @ A.T + t


def _optimize_affine_grid(deadline):
    """
    Optimize the affine transformation of the 4x4-minus-1 grid.
    Parameters: [a11, a12, a21, a22, tx, ty]
    """
    base_points = _generate_grid_points()
    
    def obj(p):
        a11, a12, a21, a22, tx, ty = p
        pts = _apply_affine_transform(base_points, a11, a12, a21, a22, tx, ty)
        return _neg_score(pts)

    # Initial guess: Identity (a11=1, a22=1, others 0)
    # We can scale it to unit area or just let the optimizer handle it.
    # The score is invariant to scaling and translation, so we fix scale/translation?
    # No, the score is min_area / hull_area.
    # Scaling both numerator and denominator by k^2 cancels out.
    # Translation doesn't change areas or hull area.
    # So we only need to optimize the shape (a11, a12, a21, a22) up to a scalar.
    # To avoid scale ambiguity, we can fix a11=1 (if non-zero) or fix the norm of A.
    # However, the grid might need to be rotated/sheared.
    # Let's optimize all 4 shape parameters.
    x0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 10:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 5:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    pts = _apply_affine_transform(base_points, *best_p)
    return pts, best_s


def _optimize_from_points(points, deadline, max_iters_powell=400):
    """Polish a point configuration using Powell then Nelder-Mead on the true objective."""
    best = points.copy()
    best_s = _score(best)

    x0 = best.flatten()

    # Powell on the true (non-smooth) objective
    if time.time() < deadline - 6:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                x0,
                method='Powell',
                options={'maxiter': max_iters_powell, 'xatol': 1e-8, 'fatol': 1e-8},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    # Nelder-Mead fine polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    return best, best_s


def _optimize_two_ring(n_out, r_inner, angle_offset, deadline):
    """Optimize a two-ring model by fitting the free params (radius + per-point perturbations)."""
    n_in = N - n_out
    # Parameters: [r_inner, angle_offset, radial_pert (N), angular_pert (N)]
    p0 = np.zeros(2 + 2 * N)
    p0[0] = r_inner
    p0[1] = angle_offset

    def obj(p):
        r_in = float(p[0])
        a_off = float(p[1])
        rp = p[2:2 + N]
        ap = p[2 + N:2 + 2 * N]
        pts = _build_two_ring(n_out, r_in, a_off, radial_pert=rp, angular_pert=ap)
        return _neg_score(pts)

    x0 = p0
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 6:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    pts = _build_two_ring(n_out, float(best_p[0]), float(best_p[1]),
                          radial_pert=best_p[2:2 + N], angular_pert=best_p[2 + N:2 + 2 * N])
    return pts, best_s


def _optimize_three_ring(n1, n2, n3, r2, r3, off2, off3, deadline):
    """Optimize a three-ring model."""
    def obj(p):
        rr2, rr3, o2, o3 = p[0], p[1], p[2], p[3]
        pts = _build_three_ring(n1, n2, n3, rr2, rr3, o2, o3)
        return _neg_score(pts)

    x0 = np.array([r2, r3, off2, off3], dtype=float)
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 6:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 250, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    pts = _build_three_ring(n1, n2, n3, float(best_p[0]), float(best_p[1]),
                            float(best_p[2]), float(best_p[3]))
    return pts, best_s


def _optimize_regular_perturbed(deadline):
    """Regular 13-gon with per-point radial/angular perturbations, optimized."""
    def obj(p):
        rp = p[:N]
        ap = p[N:2 * N]
        pts = _build_regular_perturbed(None, rp, ap)
        return _neg_score(pts)

    x0 = np.zeros(2 * N)
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 6:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    pts = _build_regular_perturbed(None, best_p[:N], best_p[N:2 * N])
    return pts, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill

    best_points = None
    best_score = 0.0

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # ---------- Seed candidates (no optimization) ----------
    # Regular 13-gon
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    _update(np.column_stack([np.cos(ang), np.sin(ang)]))
    
    # Grid seed
    grid_pts = _generate_grid_points()
    _update(grid_pts)

    # ---------- NEW: Affine Grid Optimization ----------
    if time.time() < deadline - 15:
        try:
            pts, s = _optimize_affine_grid(deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Two-ring models: fit parameters ----------
    two_ring_configs = []
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for a_off in [0.0, np.pi / n_out, np.pi / (2.0 * n_out), np.pi / n_in]:
                two_ring_configs.append((n_out, r_in, a_off))

    for (n_out, r_in, a_off) in two_ring_configs:
        if time.time() > deadline - 10:
            break
        try:
            pts, s = _optimize_two_ring(n_out, r_in, a_off, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Three-ring models ----------
    three_ring_configs = [
        (5, 4, 4, 0.6, 0.3, 0.0, 0.0),
        (6, 4, 3, 0.6, 0.3, 0.0, 0.0),
        (7, 4, 2, 0.6, 0.3, 0.0, 0.0),
        (6, 5, 2, 0.6, 0.3, 0.0, 0.0),
        (7, 3, 3, 0.6, 0.3, 0.0, 0.0),
        (5, 5, 3, 0.6, 0.3, 0.0, 0.0),
        (6, 4, 3, 0.5, 0.25, np.pi / 6, 0.0),
        (7, 4, 2, 0.5, 0.25, np.pi / 7, 0.0),
    ]
    for (n1, n2, n3, r2, r3, o2, o3) in three_ring_configs:
        if time.time() > deadline - 8:
            break
        try:
            pts, s = _optimize_three_ring(n1, n2, n3, r2, r3, o2, o3, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Regular 13-gon with perturbations ----------
    if time.time() < deadline - 10:
        try:
            pts, s = _optimize_regular_perturbed(deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Full-coordinate polish from the best found so far ----------
    if best_points is not None and time.time() < deadline - 4:
        try:
            pts, s = _optimize_from_points(best_points, deadline, max_iters_powell=300)
            _update(pts)
        except Exception:
            pass

    # ---------- Final ultra-fine Nelder-Mead on best ----------
    if best_points is not None and time.time() < deadline - 1.5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(ang), np.sin(ang)])

    return best_points
