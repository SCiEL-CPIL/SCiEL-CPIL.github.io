"""Circle packing in perimeter-4 rectangle:
Fixed-rectangle formulation + symplectic rigid-body collision relaxation
+ vectorized SLSQP joint optimization + expansion/aspect-ratio polish.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 132.0   # safety margin before 150s wall-clock kill


# ---------------------------------------------------------------------------
# Validation / value
# ---------------------------------------------------------------------------
def _is_valid(circles, tol=1e-7):
    """circles: (N,3) [x,y,r]. Valid iff inside a perimeter-4 box, no overlaps."""
    if circles.shape[0] != N:
        return False
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r <= 0):
        return False
    min_x, max_x = np.min(x - r), np.max(x + r)
    min_y, max_y = np.min(y - r), np.max(y + r)
    w, h = max_x - min_x, max_y - min_y
    if 2.0 * (w + h) > 4.0 + tol:
        return False
    # fit inside the actual box
    if np.any(x - r < min_x - tol) or np.any(x + r > max_x + tol):
        return False
    if np.any(y - r < min_y - tol) or np.any(y + r > max_y + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    rs = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d2[mask] < (rs[mask] - tol) ** 2):
        return False
    return True


def _value(circles):
    return float(np.sum(circles[:, 2]))


def _box_of(circles):
    """Return (min_x, min_y, w, h) of the bounding box of circles."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    min_x, max_x = np.min(x - r), np.max(x + r)
    min_y, max_y = np.min(y - r), np.max(y + r)
    return min_x, min_y, max_x - min_x, max_y - min_y


# ---------------------------------------------------------------------------
# Grow-to-contact: max radius per circle given fixed centers + box
# ---------------------------------------------------------------------------
def _grow_to_contact(x, y, box_w, box_h):
    """Max radius for each circle given fixed centers and box [0,box_w]x[0,box_h]."""
    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    # distance to box edges
    dedge = np.minimum(np.minimum(x, box_w - x), np.minimum(y, box_h - y))
    max_r = np.full(n, np.inf)
    for i in range(n):
        max_r[i] = min(d[i].min(), dedge[i])
    max_r = np.maximum(max_r, 1e-6)
    return max_r


# ---------------------------------------------------------------------------
# Symplectic rigid-body relaxation (uniform corner-bias field + collisions)
# ---------------------------------------------------------------------------
def _symplectic_relax(x, y, box_w, box_h, dt=0.004, n_iters=1200,
                      corner_k=0.55, wall_k=60.0, damp=0.90, grow_every=4):
    """
    Rigid-body simulation: circles in box [0,box_w]x[0,box_h], uniform field
    pushing toward the four corners (favors dense corner packing), wall springs,
    pairwise collision resolution via semi-implicit Euler (symplectic) with
    velocity damping. Radii regrown to contact periodically.
    Returns (x, y, r).
    """
    x = x.astype(float).copy()
    y = y.astype(float).copy()
    vx = np.zeros(N)
    vy = np.zeros(N)
    r = _grow_to_contact(x, y, box_w, box_h)

    for it in range(n_iters):
        if time.time() > DEADLINE - 12:
            break
        if it % grow_every == 0:
            r = _grow_to_contact(x, y, box_w, box_h)

        fx = np.zeros(N)
        fy = np.zeros(N)

        # uniform corner-bias field: pull each circle to nearest corner
        corners = np.array([[0.0, 0.0], [box_w, 0.0], [0.0, box_h], [box_w, box_h]])
        dc = np.sqrt((x[:, None] - corners[None, :, 0]) ** 2 +
                     (y[:, None] - corners[None, :, 1]) ** 2)
        nc = np.argmin(dc, axis=1)
        fx += corner_k * (corners[nc, 0] - x)
        fy += corner_k * (corners[nc, 1] - y)

        # wall springs (containment)
        fx += np.where(x - r < 0, wall_k * (r - x), 0.0)
        fx += np.where(x + r > box_w, wall_k * (box_w - x - r), 0.0)
        fy += np.where(y - r < 0, wall_k * (r - y), 0.0)
        fy += np.where(y + r > box_h, wall_k * (box_h - y - r), 0.0)

        # pairwise collisions (impulse separation + repulsion)
        for i in range(N):
            for j in range(i + 1, N):
                ddx = x[i] - x[j]
                ddy = y[i] - y[j]
                d = np.sqrt(ddx * ddx + ddy * ddy)
                if d < 1e-10:
                    ddx, ddy, d = 1e-6, 1e-6, 1e-6
                overlap = (r[i] + r[j]) - d
                if overlap > 0:
                    nx, ny = ddx / d, ddy / d
                    # hard positional separation
                    sep = overlap * 0.5
                    x[i] += sep * nx
                    y[i] += sep * ny
                    x[j] -= sep * nx
                    y[j] -= sep * ny
                    # velocity impulse (symplectic repulsion)
                    f = overlap * 40.0 + 0.05
                    fx[i] += f * nx
                    fy[i] += f * ny
                    fx[j] -= f * nx
                    fy[j] -= f * ny

        # semi-implicit Euler (symplectic) + damping
        vx = (vx + dt * fx) * damp
        vy = (vy + dt * fy) * damp
        # clamp velocities for stability
        vmax = 0.05
        vx = np.clip(vx, -vmax, vmax)
        vy = np.clip(vy, -vmax, vmax)
        x += dt * vx
        y += dt * vy

    r = _grow_to_contact(x, y, box_w, box_h)
    return x, y, r


# ---------------------------------------------------------------------------
# Vectorized SLSQP objective over (x,y,r,w)  [h = 2 - w]
# ---------------------------------------------------------------------------
def _vectorized_objective(z, penalty_weight=1000.0):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    if w < 0.25 or h < 0.25 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(np.maximum(dx * dx + dy * dy, 0.0))
    rs = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    overlap = np.maximum(rs[iu] - d[iu], 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # boundary penalty (circles inside box [0,w]x[0,h])
    lv = np.maximum(r - x, 0.0)
    rv = np.maximum(x + r - w, 0.0)
    bv = np.maximum(r - y, 0.0)
    tv = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(lv * lv) + np.sum(rv * rv) +
                             np.sum(bv * bv) + np.sum(tv * tv))

    # keep aspect ratio sane
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2
    return obj


def _check_z(z, tol=1e-7):
    """Return (valid, value) for z = (x,y,r...,w)."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    if w <= 0.05 or h <= 0.05:
        return False, 0.0
    if np.any(r <= 0):
        return False, 0.0
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False, 0.0
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False, 0.0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    rs = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d2[mask] < (rs[mask] - tol) ** 2):
        return False, 0.0
    return True, float(np.sum(r))


def _z_to_circles(z):
    """Convert z=(x,y,r,w) to (N,3) circles placed in box [0,w]x[0,2-w]."""
    x = z[0::3][:N].copy()
    y = z[1::3][:N].copy()
    r = z[2::3][:N].copy()
    return np.column_stack((x, y, r))


# ---------------------------------------------------------------------------
# SLSQP joint optimizer
# ---------------------------------------------------------------------------
def _slsqp_refine(z0, max_iter=200):
    def objective(z):
        return _vectorized_objective(z, 1000.0)

    bounds = [(0, 2)] * (3 * N) + [(0.35, 1.65)]
    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'maxls': 80})
        return res.x
    except Exception:
        return z0


# ---------------------------------------------------------------------------
# Expansion / aspect-ratio polish on a (N,3) result
# ---------------------------------------------------------------------------
def _expand_radii(circles, box_w, box_h, n_iters=40):
    """Grow radii to contact within fixed box, iterating."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    for _ in range(n_iters):
        if time.time() > DEADLINE - 12:
            break
        r_new = _grow_to_contact(x, y, box_w, box_h)
        r_new = np.maximum(r_new, r * 0.5)
        if np.allclose(r_new, r, atol=1e-12):
            break
        r = r_new
    return np.column_stack((x, y, r))


def _aspect_polish(circles, n_aspect=9):
    """Try several aspect ratios: rescale positions, grow, relax, keep best."""
    best = circles.copy()
    best_val = _value(best) if _is_valid(best) else 0.0
    min_x, min_y, w0, h0 = _box_of(best)
    for w in np.linspace(0.6, 1.4, n_aspect):
        if time.time() > DEADLINE - 12:
            break
        h = 2.0 - w
        # rescale positions into [0,w]x[0,h]
        x = (circles[:, 0] - min_x) / w0 * w
        y = (circles[:, 1] - min_y) / h0 * h
        r = _grow_to_contact(x, y, w, h)
        # light symplectic relax to resolve overlaps
        x2, y2, r2 = _symplectic_relax(x, y, w, h, dt=0.004, n_iters=400,
                                       corner_k=0.4, wall_k=60.0, damp=0.90)
        cand = np.column_stack((x2, y2, r2))
        if _is_valid(cand):
            v = _value(cand)
            if v > best_val:
                best_val = v
                best = cand
    return best


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------
def _grid_init(w, h, side=None):
    if side is None:
        side = int(np.ceil(np.sqrt(N)))
    x, y, r = [], [], []
    for i in range(N):
        row, col = divmod(i, side)
        cx = (col + 0.5) / side * w
        cy = (row + 0.5) / side * h
        r0 = min(w / side, h / side) * 0.42
        x.append(cx)
        y.append(cy)
        r.append(r0)
    return np.array(x), np.array(y)


def _random_init(w, h, rng):
    r0 = rng.uniform(0.05, 0.11, N)
    x = rng.uniform(r0, w - r0, N)
    y = rng.uniform(r0, h - r0, N)
    return x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    best_circles = None
    best_value = 0.0
    rng = np.random.default_rng(12345)

    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8]

    def consider(cand):
        nonlocal best_circles, best_value
        if cand is not None and _is_valid(cand):
            v = _value(cand)
            if v > best_value:
                best_value = v
                best_circles = cand.copy()

    # ---- Phase 1: symplectic physical relaxation from diverse starts ----
    for k, w0 in enumerate(aspect_ratios):
        if time.time() > DEADLINE - 35:
            break
        h0 = 2.0 - w0
        if h0 < 0.35:
            continue
        if k < 6:
            x, y = _grid_init(w0, h0)
        else:
            x, y = _random_init(w0, h0, rng)
        x2, y2, r2 = _symplectic_relax(x, y, w0, h0, dt=0.004, n_iters=1000,
                                       corner_k=0.55, wall_k=60.0, damp=0.90)
        cand = np.column_stack((x2, y2, r2))
        consider(cand)

        # SLSQP joint refinement (adjust aspect + radii)
        if time.time() < DEADLINE - 30:
            z = np.concatenate([x2, y2, r2, [w0]])
            zopt = _slsqp_refine(z, max_iter=150)
            consider(_z_to_circles(zopt))

    # ---- Phase 2: aspect-ratio + expansion polish on the best ----
    if best_circles is not None and time.time() < DEADLINE - 20:
        consider(_aspect_polish(best_circles, n_aspect=9))

    # ---- Phase 3: final SLSQP + tiny shrink safety on the best ----
    if best_circles is not None and time.time() < DEADLINE - 10:
        min_x, min_y, w0, h0 = _box_of(best_circles)
        x = best_circles[:, 0] - min_x
        y = best_circles[:, 1] - min_y
        r = best_circles[:, 2]
        z = np.concatenate([x, y, r, [w0]])
        zopt = _slsqp_refine(z, max_iter=120)
        consider(_z_to_circles(zopt))

    # ---- Fallback ----
    if best_circles is None:
        x, y = _grid_init(1.0, 1.0)
        r = _grow_to_contact(x, y, 1.0, 1.0)
        best_circles = np.column_stack((x, y, r))

    # ---- Final safety: shrink radii until strictly valid ----
    for _ in range(20):
        if _is_valid(best_circles):
            break
        best_circles = best_circles.copy()
        best_circles[:, 2] *= 0.98

    return best_circles
