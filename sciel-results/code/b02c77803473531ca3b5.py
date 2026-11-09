# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _check_validity(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                return False
    return True

def _objective(centers, sides):
    if not _check_validity(centers, sides):
        return -np.inf
    return float(np.sum(sides))

def _can_place(centers, sides, i, s_try, cx_try, cy_try, tol=1e-7):
    n = len(sides)
    if cx_try - s_try/2 < -tol or cx_try + s_try/2 > 1 + tol:
        return False
    if cy_try - s_try/2 < -tol or cy_try + s_try/2 > 1 + tol:
        return False
    for j in range(n):
        if j == i:
            continue
        t = (s_try + sides[j])/2 - tol
        if abs(cx_try - centers[j,0]) < t and abs(cy_try - centers[j,1]) < t:
            return False
    return True

def _max_side_for_square(centers, sides, i, max_s=1.0, tol=1e-7, n_pos_tries=200):
    """Bisection on side length for square i, searching for a valid position."""
    n = len(sides)
    lo = sides[i]
    hi = min(max_s, 2*centers[i,0]+2*tol, 2*(1-centers[i,0])+2*tol,
             2*centers[i,1]+2*tol, 2*(1-centers[i,1])+2*tol)
    if hi <= lo + 1e-13:
        return sides[i], centers[i].copy()
    best_s = lo
    best_c = centers[i].copy()
    rng = np.random.default_rng(i*1000 + 7)
    for _ in range(50):
        mid = (lo + hi)/2
        if mid <= best_s + 1e-13:
            break
        found = False
        # try current position
        if _can_place(centers, sides, i, mid, centers[i,0], centers[i,1], tol):
            found = True
            best_s = mid
            best_c = centers[i].copy()
            lo = mid
            continue
        # structured candidate positions: corners, edges, grid
        half = mid / 2
        candidates = []
        # Corners of the bounding box
        for cx in [half, 1-half]:
            for cy in [half, 1-half]:
                candidates.append((cx, cy))
        # Midpoints of edges
        for cx in [0.5, half, 1-half]:
            candidates.append((cx, half))
            candidates.append((cx, 1-half))
        for cy in [0.5, half, 1-half]:
            candidates.append((half, cy))
            candidates.append((1-half, cy))
        # Center
        candidates.append((0.5, 0.5))
        # Check structured candidates
        for cx, cy in candidates:
            if _can_place(centers, sides, i, mid, cx, cy, tol):
                found = True
                best_s = mid
                best_c = np.array([cx, cy])
                lo = mid
                break
        if found:
            continue
        # random positions
        for _ in range(n_pos_tries):
            cx = rng.uniform(half, 1-half)
            cy = rng.uniform(half, 1-half)
            if _can_place(centers, sides, i, mid, cx, cy, tol):
                found = True
                best_s = mid
                best_c = np.array([cx, cy])
                lo = mid
                break
        if not found:
            hi = mid
    return best_s, best_c

def _coordinate_ascent(centers, sides, max_iter=2000, tol=1e-10):
    n = len(sides)
    centers = centers.copy()
    sides = sides.copy()
    for iteration in range(max_iter):
        improved = False
        order = np.random.permutation(n)
        for i in order:
            new_s, new_c = _max_side_for_square(centers, sides, i, tol=tol, n_pos_tries=150)
            if new_s > sides[i] + tol:
                sides[i] = new_s
                centers[i] = new_c
                improved = True
        if not improved:
            break
    return centers, sides

def _flatten(c, s):
    return np.concatenate([c.flatten(), s])

def _unflatten(x):
    c = x[:20].reshape(10, 2)
    s = x[20:]
    return c, s

def _penalty(x, weight=1e5):
    c, s = _unflatten(x)
    pen = 0.0
    tol = 1e-7
    for i in range(10):
        cx, cy = c[i]
        si = s[i]
        if si <= 0:
            pen += weight * (0.01 - si)
        if cx - si/2 < -tol:
            pen += weight * (-tol - (cx - si/2))
        if cx + si/2 > 1 + tol:
            pen += weight * ((cx + si/2) - (1 + tol))
        if cy - si/2 < -tol:
            pen += weight * (-tol - (cy - si/2))
        if cy + si/2 > 1 + tol:
            pen += weight * ((cy + si/2) - (1 + tol))
    for i in range(10):
        for j in range(i+1, 10):
            t = (s[i] + s[j])/2 - tol
            dx = c[i,0] - c[j,0]
            dy = c[i,1] - c[j,1]
            if abs(dx) < t and abs(dy) < t:
                pen += weight * ((t - abs(dx)) + (t - abs(dy)))
    return pen

def _combined(x, weight=1e5):
    c, s = _unflatten(x)
    return -np.sum(s) + _penalty(x, weight)

def _nelder_mead(x0, time_limit=25):
    start = time.time()
    def obj(x):
        if time.time() - start > time_limit:
            return 1e10
        return _combined(x)
    try:
        result = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
        return result.x
    except Exception:
        return x0

def _lbfgsb(x0, time_limit=25):
    start = time.time()
    bounds = []
    for i in range(10):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.01, 1.0))
    def obj(x):
        if time.time() - start > time_limit:
            return 1e10
        return _combined(x)
    try:
        result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
        return result.x
    except Exception:
        return x0

def _local_search(centers, sides, time_limit=30):
    n = 10
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _objective(best_centers, best_sides)
    start_time = time.time()
    rng = np.random.default_rng(42)
    # Start with small perturbations, gradually increase
    scale = 0.01
    while time.time() - start_time < time_limit:
        centers_p = best_centers.copy()
        sides_p = best_sides.copy()
        for i in range(n):
            centers_p[i, 0] += rng.normal(0, scale)
            centers_p[i, 1] += rng.normal(0, scale)
            sides_p[i] *= (1 + rng.normal(0, scale * 0.5))
        for i in range(n):
            sides_p[i] = max(sides_p[i], 0.01)
            centers_p[i, 0] = np.clip(centers_p[i, 0], sides_p[i]/2, 1 - sides_p[i]/2)
            centers_p[i, 1] = np.clip(centers_p[i, 1], sides_p[i]/2, 1 - sides_p[i]/2)
        score = _objective(centers_p, sides_p)
        if score > best_score + 1e-10:
            best_centers = centers_p
            best_sides = sides_p
            best_score = score
            scale = max(0.002, scale * 0.7)  # shrink on success
        else:
            scale = min(0.05, scale * 1.15)  # grow on failure
        # Periodic coordinate ascent polish
        if int((time.time() - start_time) * 2) % 3 == 0:
            ca_centers, ca_sides = _coordinate_ascent(best_centers, best_sides, max_iter=200)
            ca_score = _objective(ca_centers, ca_sides)
            if ca_score > best_score + 1e-10:
                best_centers = ca_centers
                best_sides = ca_sides
                best_score = ca_score
    return best_centers, best_sides

def _make_2big_8small():
    centers = np.array([
        [0.25, 0.25], [0.75, 0.25],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
        [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
    ], dtype=np.float64)
    sides = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    return centers, sides

def _make_grid():
    k = int(np.ceil(np.sqrt(10)))
    side = 1.0 / k
    idx = np.arange(10)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(10, side, dtype=np.float64)
    return centers, sides

def squares_in_square10() -> tuple:
    deadline = time.time() + 620  # safety margin before 630s kill

    # Build candidate seeds
    seeds = []
    c, s = _make_2big_8small()
    if _check_validity(c, s):
        seeds.append((c, s))
    c, s = _make_grid()
    if _check_validity(c, s):
        seeds.append((c, s))

    best_centers, best_sides = seeds[0][0], seeds[0][1]
    best_score = _objective(best_centers, best_sides)

    for c, s in seeds:
        score = _objective(c, s)
        if score > best_score:
            best_centers, best_sides = c.copy(), s.copy()
            best_score = score

    # Coordinate ascent from best seed
    if time.time() < deadline - 60:
        ca_c, ca_s = _coordinate_ascent(best_centers, best_sides, max_iter=3000)
        ca_score = _objective(ca_c, ca_s)
        if ca_score > best_score:
            best_centers, best_sides = ca_c, ca_s
            best_score = ca_score

    # Nelder-Mead from best
    if time.time() < deadline - 40:
        x0 = _flatten(best_centers, best_sides)
        x_nm = _nelder_mead(x0, time_limit=25)
        c_nm, s_nm = _unflatten(x_nm)
        nm_score = _objective(c_nm, s_nm)
        if nm_score > best_score:
            best_centers, best_sides = c_nm, s_nm
            best_score = nm_score

    # L-BFGS-B from best
    if time.time() < deadline - 30:
        x0 = _flatten(best_centers, best_sides)
        x_lb = _lbfgsb(x0, time_limit=25)
        c_lb, s_lb = _unflatten(x_lb)
        lb_score = _objective(c_lb, s_lb)
        if lb_score > best_score:
            best_centers, best_sides = c_lb, s_lb
            best_score = lb_score

    # Local search with multiple phases
    remaining = deadline - time.time() - 20
    if remaining > 15:
        ls_c, ls_s = _local_search(best_centers, best_sides, time_limit=min(remaining, 45))
        ls_score = _objective(ls_c, ls_s)
        if ls_score > best_score:
            best_centers, best_sides = ls_c, ls_s
            best_score = ls_score

    # Second local search phase with different seed if time permits
    remaining2 = deadline - time.time() - 10
    if remaining2 > 10:
        rng2 = np.random.default_rng(123)
        c2 = best_centers.copy()
        s2 = best_sides.copy()
        for i in range(10):
            c2[i, 0] += rng2.normal(0, 0.03)
            c2[i, 1] += rng2.normal(0, 0.03)
            s2[i] *= (1 + rng2.normal(0, 0.02))
        for i in range(10):
            s2[i] = max(s2[i], 0.01)
            c2[i, 0] = np.clip(c2[i, 0], s2[i]/2, 1 - s2[i]/2)
            c2[i, 1] = np.clip(c2[i, 1], s2[i]/2, 1 - s2[i]/2)
        ls2_c, ls2_s = _local_search(c2, s2, time_limit=min(remaining2, 30))
        ls2_score = _objective(ls2_c, ls2_s)
        if ls2_score > best_score:
            best_centers, best_sides = ls2_c, ls2_s
            best_score = ls2_score

    # Final high-precision coordinate ascent polish
    if time.time() < deadline - 5:
        ca_c, ca_s = _coordinate_ascent(best_centers, best_sides, max_iter=3000)
        ca_score = _objective(ca_c, ca_s)
        if ca_score > best_score:
            best_centers, best_sides = ca_c, ca_s
            best_score = ca_score

    # Ultra-fine final polish: tiny nudges + re-optimize
    if time.time() < deadline - 3:
        rng_final = np.random.default_rng(999)
        cf = best_centers.copy()
        sf = best_sides.copy()
        for i in range(10):
            cf[i, 0] += rng_final.normal(0, 0.001)
            cf[i, 1] += rng_final.normal(0, 0.001)
        for i in range(10):
            cf[i, 0] = np.clip(cf[i, 0], sf[i]/2, 1 - sf[i]/2)
            cf[i, 1] = np.clip(cf[i, 1], sf[i]/2, 1 - sf[i]/2)
        ca_f, ca_sf = _coordinate_ascent(cf, sf, max_iter=500)
        ca_f_score = _objective(ca_f, ca_sf)
        if ca_f_score > best_score:
            best_centers, best_sides = ca_f, ca_sf
            best_score = ca_f_score

    # Ensure validity; fall back to grid
    if not _check_validity(best_centers, best_sides):
        gc, gs = _make_grid()
        return gc, gs

    return best_centers, best_sides

# EVOLVE-BLOCK-END
