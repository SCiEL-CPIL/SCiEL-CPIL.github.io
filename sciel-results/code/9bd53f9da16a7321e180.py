# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _objective(centers, sides):
    return float(np.sum(sides))


def _max_side_at(cx, cy, others_c, others_s, i, tol=1e-7):
    """Max side for square i at (cx,cy) given others (excluding i)."""
    # boundary
    m = min(cx, 1.0 - cx, cy, 1.0 - cy)
    m = max(0.0, 2.0 * m)
    for j, (ojx, ojy, sj) in enumerate(others_c, 0):
        if j == i:
            continue
        dx = abs(cx - ojx)
        dy = abs(cy - ojy)
        # need dx >= (s+sj)/2 OR dy >= (s+sj)/2  -> s <= 2*max(dx,dy) - sj
        cand = 2.0 * max(dx, dy) - sj
        if cand < m:
            m = cand
    return max(0.0, m - tol)


def _grow_and_polish(centers, sides, tol=1e-7, iters=200):
    """Repeatedly grow each square to its max, then nudge to relieve tightness.
    Returns improved (centers, sides)."""
    centers = np.array(centers, dtype=np.float64).copy()
    sides = np.array(sides, dtype=np.float64).copy()
    n = len(sides)
    rng = np.random.default_rng(7)
    best = (centers.copy(), sides.copy())
    best_val = _objective(centers, sides)

    for it in range(iters):
        # grow each square to max
        for i in range(n):
            cx, cy = centers[i]
            others = [(centers[j, 0], centers[j, 1], sides[j]) for j in range(n)]
            smax = _max_side_at(cx, cy, others, sides, i, tol)
            if smax > sides[i]:
                sides[i] = smax
        # check validity; if invalid, shrink slightly the offending pairs
        if not _is_valid(centers, sides, tol):
            # shrink all a touch
            sides *= 0.999
        # random nudge to relieve tight constraints and allow further growth
        if it % 5 == 0:
            for i in range(n):
                centers[i, 0] += rng.normal(0, 0.004)
                centers[i, 1] += rng.normal(0, 0.004)
                # clamp inside
                s = sides[i]
                centers[i, 0] = max(s / 2, min(1 - s / 2, centers[i, 0]))
                centers[i, 1] = max(s / 2, min(1 - s / 2, centers[i, 1]))
        if _is_valid(centers, sides, tol):
            v = _objective(centers, sides)
            if v > best_val:
                best_val = v
                best = (centers.copy(), sides.copy())

    return best


def _smooth_energy(z, n, tol=1e-7, lam=1e4, soft=1e-3):
    """z = (2n,) -> centers (n,2) and sides (n,). Returns (energy, gradient)."""
    centers = z[:2 * n].reshape(n, 2)
    sides = z[2 * n:]

    # objective: maximize sum sides -> minimize -sum sides
    energy = -np.sum(sides)
    grad = np.zeros_like(z)
    grad[2 * n:] = -1.0

    # boundary soft penalty: keep squares inside
    # left: cx - s/2 >= 0 ; right: cx + s/2 <= 1 ; same for y
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # left
        v = cx - s / 2
        if v < 0:
            energy += lam * v * v
            grad[2 * i] += lam * 2 * v
            grad[2 * n + i] += lam * 2 * v * (-0.5)
        # right
        v = 1.0 - (cx + s / 2)
        if v < 0:
            energy += lam * v * v
            grad[2 * i] += lam * 2 * v * (-1)
            grad[2 * n + i] += lam * 2 * v * (-0.5)
        # bottom
        v = cy - s / 2
        if v < 0:
            energy += lam * v * v
            grad[2 * i + 1] += lam * 2 * v
            grad[2 * n + i] += lam * 2 * v * (-0.5)
        # top
        v = 1.0 - (cy + s / 2)
        if v < 0:
            energy += lam * v * v
            grad[2 * i + 1] += lam * 2 * v * (-1)
            grad[2 * n + i] += lam * 2 * v * (-0.5)

    # pairwise smooth overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            # overlap amount along x: (s_i+s_j)/2 - |dx| ; positive = overlapping
            ox = (sides[i] + sides[j]) / 2 - abs(dx)
            oy = (sides[i] + sides[j]) / 2 - abs(dy)
            # both must be <= 0 for no overlap. Penalize when both positive.
            # smooth: use softplus-like on max(ox,0) and max(oy,0) product
            a = max(0.0, ox)
            b = max(0.0, oy)
            if a > 0 and b > 0:
                # penalty grows with both
                p = a * b
                energy += lam * p
                # gradient w.r.t. dx
                g_ox = 0.5 * (1 if sides[i] + sides[j] > 0 else 0)  # d((si+sj)/2 - |dx|)/d dx
                # d ox / d cx_i = -sign(dx)
                sdx = 1.0 if dx > 0 else (-1.0 if dx < 0 else 0.0)
                sdy = 1.0 if dy > 0 else (-1.0 if dy < 0 else 0.0)
                # d ox / d cx_i = -sdx ; d ox / d cx_j = +sdx
                # d oy / d cy_i = -sdy ; d oy / d cy_j = +sdy
                # d ox / d s_i = 0.5 ; d oy / d s_i = 0.5
                grad[2 * i] += lam * b * (-sdx)
                grad[2 * i + 1] += lam * b * (-sdy)
                grad[2 * j] += lam * b * (sdx)
                grad[2 * j + 1] += lam * b * (sdy)
                grad[2 * n + i] += lam * (b * 0.5 + a * 0.5)
                grad[2 * n + j] += lam * (b * 0.5 + a * 0.5)

    return energy, grad


def _smooth_optimize(z0, n, tol=1e-7, maxiter=400):
    z0 = np.array(z0, dtype=np.float64).copy()
    res = minimize(_smooth_energy, z0, jac=True, method='L-BFGS-B',
                   args=(n, tol),
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    z = res.x
    centers = z[:2 * n].reshape(n, 2)
    sides = z[2 * n:]
    # clamp to feasible-ish
    sides = np.clip(sides, 1e-4, 1.0)
    for i in range(n):
        s = sides[i]
        centers[i, 0] = max(s / 2, min(1 - s / 2, centers[i, 0]))
        centers[i, 1] = max(s / 2, min(1 - s / 2, centers[i, 1]))
    return centers, sides


def _init_grid(n):
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1)
    sides = np.full(n, side)
    return centers, sides


def _init_corner(n):
    # 4 corners large + fill center with smaller
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    # 4 corner squares of side 0.35
    cs = 0.35
    corners = [(cs/2, cs/2), (1-cs/2, cs/2), (cs/2, 1-cs/2), (1-cs/2, 1-cs/2)]
    for i in range(4):
        centers[i] = corners[i]
        sides[i] = cs
    # 6 in middle region
    mid = 0.65
    for i in range(4, n):
        k = i - 4
        r = k // 2
        c = k % 2
        centers[i, 0] = 0.5 + (c - 0.5) * 0.3
        centers[i, 1] = 0.5 + (r - 0.5) * 0.3
        sides[i] = 0.15
    return centers, sides


def _init_stripe(n):
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    # two rows of 5
    for i in range(n):
        r = i // 5
        c = i % 5
        centers[i, 0] = (c + 0.5) * 0.2
        centers[i, 1] = 0.25 if r == 0 else 0.75
        sides[i] = 0.18
    return centers, sides


def _init_random(n, seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.15, 0.85, size=(n, 2))
    sides = rng.uniform(0.08, 0.2, size=n)
    return centers, sides


def _to_z(centers, sides):
    n = len(sides)
    z = np.concatenate([centers.ravel(), sides])
    return z


def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 600

    inits = [
        _init_grid(n),
        _init_corner(n),
        _init_stripe(n),
        _init_random(n, 1),
        _init_random(n, 2),
        _init_random(n, 3),
    ]

    best_centers = None
    best_sides = None
    best_val = -np.inf

    def consider(centers, sides):
        nonlocal best_centers, best_sides, best_val
        if not _is_valid(centers, sides, tol):
            return
        v = _objective(centers, sides)
        if v > best_val:
            best_val = v
            best_centers = centers.copy()
            best_sides = sides.copy()

    for ci, (centers, sides) in enumerate(inits):
        if time.time() > deadline - 60:
            break
        # smooth optimize
        z0 = _to_z(centers, sides)
        try:
            c2, s2 = _smooth_optimize(z0, n, tol, maxiter=300)
            consider(c2, s2)
        except Exception:
            pass
        # grow-and-polish from the optimized (and from the init)
        for (c_init, s_init) in [(c2, s2), (centers, sides)]:
            if time.time() > deadline - 60:
                break
            try:
                c3, s3 = _grow_and_polish(c_init, s_init, tol, iters=120)
                consider(c3, s3)
            except Exception:
                pass

    # final polish on best
    if best_centers is not None and time.time() < deadline - 30:
        try:
            c4, s4 = _grow_and_polish(best_centers, best_sides, tol, iters=150)
            consider(c4, s4)
        except Exception:
            pass

    if best_centers is None:
        # fallback to a valid grid
        centers, sides = _init_grid(n)
        c4, s4 = _grow_and_polish(centers, sides, tol, iters=50)
        best_centers, best_sides = c4, s4

    return best_centers, best_sides


# EVOLVE-BLOCK-END
