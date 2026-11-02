"""Heilbronn triangles: maximize the smallest triangle area among 11 points.

Structural approach:
- Barycentric coordinates (a, b, c) with a+b+c=1, all >= 0.  Store (b, c).
- Phase A: low-dimensional smooth mode parameterization.  The 11 points are
  placed at fixed "anchor" positions and each anchor is displaced by a linear
  combination of the first K Laplacian eigenmodes (smooth functions).  We
  optimize the mode coefficients with Nelder-Mead on a soft-min objective.
- Phase B: full 22-dim polish with coordinate descent and Nelder-Mead on the
  hard min objective.
"""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the unit-side equilateral triangle

# Vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

DEADLINE = 145.0  # safety margin before 150 s kill


# ---------------------------------------------------------------------------
# Barycentric helpers
# ---------------------------------------------------------------------------
def _cart_to_bary(pts):
    """pts: (N,2) Cartesian -> (N,2) barycentric (b, c) where a=1-b-c."""
    b = pts[:, 0] - pts[:, 1] / S3
    c = pts[:, 1] / S3H
    return np.column_stack([b, c])


def _bary_to_cart(bc):
    """bc: (N,2) (b,c) -> (N,2) Cartesian."""
    b = bc[:, 0]
    c = bc[:, 1]
    a = 1.0 - b - c
    x = a * V0[0] + b * V1[0] + c * V2[0]
    y = a * V0[1] + b * V1[1] + c * V2[1]
    return np.column_stack([x, y])


def _clip_bary(bc):
    """Project barycentric coords to the valid simplex (a,b,c >= 0)."""
    b = np.clip(bc[:, 0], 0.0, None)
    c = np.clip(bc[:, 1], 0.0, None)
    # ensure b + c <= 1
    s = b + c
    mask = s > 1.0
    if np.any(mask):
        scale = 1.0 / s[mask]
        b[mask] *= scale
        c[mask] *= scale
    return np.column_stack([b, c])


# ---------------------------------------------------------------------------
# Area computation (vectorized, works on Cartesian points)
# ---------------------------------------------------------------------------
def _min_triangle_area(pts):
    """pts: (N,2) Cartesian. Return min triangle area."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_triangle_areas(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _soft_min_area(pts, beta=50.0):
    """Smooth approximation of min area via log-sum-exp of -areas."""
    areas = _all_triangle_areas(pts)
    # Avoid log(0)
    areas = np.maximum(areas, 1e-15)
    neg = -beta * areas
    max_neg = np.max(neg)
    logsumexp = max_neg + np.log(np.sum(np.exp(neg - max_neg)))
    return -logsumexp / beta


# ---------------------------------------------------------------------------
# Laplacian eigenmodes on the equilateral triangle (smooth basis)
# ---------------------------------------------------------------------------
# The eigenfunctions of the Laplacian on an equilateral triangle with
# Dirichlet BC are of the form sin(pi * (m + n/2) * x') * sin(...) etc.
# For our purpose we use a simpler smooth basis:
#   phi_k(b, c) = sin(pi * (k+1) * b) * sin(pi * (k+1) * c) * sin(pi * (k+1) * a)
# These vanish on the boundary and are smooth in the interior.
# We use the first K modes (k=0,1,...,K-1) in each of 2 displacement directions.

def _eigenmode_values(bc, k):
    """Evaluate the k-th eigenmode at barycentric points bc (N,2).
    Returns (N,) values in [0,1]."""
    b = bc[:, 0]
    c = bc[:, 1]
    a = 1.0 - b - c
    # Clip to [0,1] for safety
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    c = np.clip(c, 0.0, 1.0)
    freq = k + 1
    val = np.sin(np.pi * freq * a) * np.sin(np.pi * freq * b) * np.sin(np.pi * freq * c)
    return val


def _mode_displacement(bc, coeffs, K=4):
    """Given barycentric points bc (N,2) and mode coefficients coeffs (2K,),
    return displaced Cartesian points.
    coeffs[0:K] scale the x-displacement modes, coeffs[K:2K] scale y-modes."""
    disp_x = np.zeros(N)
    disp_y = np.zeros(N)
    for k in range(K):
        v = _eigenmode_values(bc, k)
        disp_x += coeffs[k] * v
        disp_y += coeffs[K + k] * v
    # Convert base bary to Cartesian, add small displacement
    cart = _bary_to_cart(bc)
    cart[:, 0] += disp_x
    cart[:, 1] += disp_y
    # Project back into triangle
    cart = _project_to_triangle(cart)
    return cart


def _project_to_triangle(pts):
    """Vectorized projection into equilateral triangle (Cartesian)."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Anchor positions: symmetric "ideal" placement of 11 points
# ---------------------------------------------------------------------------
def _make_anchors():
    """Return (N,2) barycentric anchor positions (symmetric, well-spread)."""
    # We want 11 points.  Use a structured pattern:
    # 3 corners, 3 edge midpoints, 5 interior points in a pentagon-like ring
    anchors_bary = []
    # Corners (a=1,b=0,c=0), (a=0,b=1,c=0), (a=0,b=0,c=1)
    anchors_bary.append([0.0, 0.0])   # V0
    anchors_bary.append([1.0, 0.0])   # V1
    anchors_bary.append([0.0, 1.0])   # V2
    # Edge midpoints
    anchors_bary.append([0.5, 0.0])   # mid V0-V1
    anchors_bary.append([0.0, 0.5])   # mid V0-V2
    anchors_bary.append([0.5, 0.5])   # mid V1-V2
    # Interior: 5 points in a regular pentagon around center (1/3,1/3)
    center = np.array([1.0/3, 1.0/3])
    r = 0.15  # radius in barycentric space
    for i in range(5):
        angle = 2 * np.pi * i / 5 - np.pi / 2  # start from top
        # In barycentric, "circular" is tricky; use a simple mapping
        bx = center[0] + r * np.cos(angle)
        by = center[1] + r * np.sin(angle)
        anchors_bary.append([bx, by])
    anchors_bary = np.array(anchors_bary, dtype=float)
    anchors_bary = _clip_bary(anchors_bary)
    return anchors_bary


# ---------------------------------------------------------------------------
# Coordinate descent in Cartesian space
# ---------------------------------------------------------------------------
def _coordinate_descent(pts, max_iter=150, tol=1e-12):
    """Full 22-dim coordinate descent: for each point, try small perturbations."""
    current_min = _min_triangle_area(pts)
    step_sizes = [0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]

    for it in range(max_iter):
        improved = False
        for idx in range(N):
            best_local = current_min
            best_pos = pts[idx].copy()
            for step in step_sizes:
                for dx in [-step, 0.0, step]:
                    for dy in [-step, 0.0, step]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        new_pts = pts.copy()
                        new_pts[idx] = best_pos + np.array([dx, dy])
                        new_pts = _project_to_triangle(new_pts)
                        new_min = _min_triangle_area(new_pts)
                        if new_min > best_local + tol:
                            best_local = new_min
                            best_pos = new_pts[idx].copy()
            if np.linalg.norm(best_pos - pts[idx]) > 1e-13:
                pts[idx] = best_pos
                current_min = _min_triangle_area(pts)
                improved = True
        if not improved:
            break
    return pts


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin=5.0):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ------------------------------------------------------------------
    # Phase A: Low-dimensional smooth mode optimization
    # ------------------------------------------------------------------
    anchors = _make_anchors()
    K = 4  # number of eigenmodes per direction -> 2K = 8 parameters

    def _mode_objective(coeff_flat):
        """Objective: negative soft-min area for mode-displaced points."""
        coeffs = coeff_flat
        cart = _mode_displacement(anchors, coeffs, K)
        return -_soft_min_area(cart, beta=40.0)

    # Multi-start in coefficient space
    n_mode_starts = 12
    for s in range(n_mode_starts):
        if not _elapsed_ok(40):
            break
        if s == 0:
            coeff0 = np.zeros(2 * K)
        else:
            coeff0 = rng.normal(0, 0.05, size=2 * K)

        def nm_obj_mode(x):
            return _mode_objective(x)

        try:
            res = minimize(nm_obj_mode, coeff0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            cart = _mode_displacement(anchors, res.x, K)
            _update_best(cart)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase B: Full 22-dim Nelder-Mead polish from best candidates
    # ------------------------------------------------------------------
    if best_pts is not None and _elapsed_ok(30):
        bc_best = best_pts.copy()

        # Several scales of perturbation + NM polish
        for scale in [0.04, 0.02, 0.01, 0.005]:
            if not _elapsed_ok(20):
                break
            n_pert = 6
            for _ in range(n_pert):
                if not _elapsed_ok(15):
                    break
                pert = bc_best + rng.normal(0, scale, size=(N, 2))
                pert = _project_to_triangle(pert)

                def nm_obj_full(x):
                    px = _project_to_triangle(x.reshape(-1, 2))
                    return -_min_triangle_area(px)

                try:
                    res = minimize(nm_obj_full, pert.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
                    p = _project_to_triangle(res.x.reshape(-1, 2))
                    _update_best(p)
                    if _min_triangle_area(p) > _min_triangle_area(bc_best):
                        bc_best = p.copy()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Phase C: Coordinate descent polish
    # ------------------------------------------------------------------
    if best_pts is not None and _elapsed_ok(15):
        p = _coordinate_descent(best_pts, max_iter=150)
        _update_best(p)

    # ------------------------------------------------------------------
    # Phase D: Final ultra-fine NM + CD
    # ------------------------------------------------------------------
    if best_pts is not None and _elapsed_ok(8):
        flat = best_pts.flatten()

        def final_nm(x):
            px = _project_to_triangle(x.reshape(-1, 2))
            return -_min_triangle_area(px)

        try:
            res = minimize(final_nm, flat, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-16})
            p = _project_to_triangle(res.x.reshape(-1, 2))
            _update_best(p)
        except Exception:
            pass

        # Final coordinate descent
        try:
            p = _coordinate_descent(best_pts, max_iter=200)
            _update_best(p)
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        return pts

    return _project_to_triangle(best_pts)


def _random_point_in_triangle(rng):
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    x = u + 0.5 * v
    y = v * S3H
    return np.array([x, y])
