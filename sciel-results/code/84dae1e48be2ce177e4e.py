"""Circle packing in a perimeter-4 rectangle.

Structural change from the seed:
  * Seed: a fixed square grid of 21 cells (5x5 minus 4), no optimisation.
  * New: a physics-style "repulsion" packing (soft Coulomb + hard wall
    constraints) that finds a dense arrangement of 21 equal circles, then
    we scale the whole layout to the largest rectangle of perimeter 4 that
    still contains it, and finally a short local refinement pass.
"""

N = 21
import time
import numpy as np

_DEADLINE = 150.0          # wall-clock limit (s); evaluator kills at ~120-150s
_SAFETY   = 25.0           # return this many seconds before the deadline
_RNG_SEED = 12345


def _elapsed():
    return time.time() - _T0


def _valid(arr):
    """Check disjoint interiors + inside a perimeter-4 rectangle.

    Returns (ok, perimeter). We infer the rectangle from the circles'
    bounding box; if the bounding-box perimeter <= 4 and all circles are
    disjoint and fully inside the box, it is valid.
    """
    if arr is None or arr.shape != (N, 3):
        return False, 4.0
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0) or np.any(~np.isfinite(arr)):
        return False, 4.0
    # bounding box
    xmin, xmax = x.min() - r.min(), x.max() + r.max()
    ymin, ymax = y.min() - r.min(), y.max() + r.max()
    W, H = xmax - xmin, ymax - ymin
    if W <= 0 or H <= 0:
        return False, 4.0
    perim = 2.0 * (W + H)
    if perim > 4.0 + 1e-6:
        return False, perim
    # circles fully inside the box
    if (x.min() - r.min() < xmin - 1e-7) or (x.max() + r.max() > xmax + 1e-7):
        return False, perim
    if (y.min() - r.min() < ymin - 1e-7) or (y.max() + r.max() > ymax + 1e-7):
        return False, perim
    # pairwise disjoint interiors
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    iu = np.triu_indices(N, 1)
    if np.any(dist[iu] < need[iu] - 1e-9):
        return False, perim
    return True, perim


def _objective(arr):
    """Sum of radii, penalised heavily if invalid."""
    ok, perim = _valid(arr)
    if not ok:
        return -1e9
    return float(np.sum(arr[:, 2]))


def _scale_to_perimeter(arr, target_perim=4.0, margin=1e-7):
    """Scale positions and radii so the bounding-box perimeter == target.

    Returns (scaled_arr, scale_factor).
    """
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    xmin, xmax = x.min() - r.min(), x.max() + r.max()
    ymin, ymax = y.min() - r.min(), y.max() + r.max()
    W, H = xmax - xmin, ymax - ymin
    perim = 2.0 * (W + H)
    if perim <= 0:
        return arr.copy(), 1.0
    s = (target_perim - 2.0 * margin) / perim
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    xs = (x - cx) * s
    ys = (y - cy) * s
    rs = r * s
    return np.column_stack([xs, ys, rs]), s


def _repulsion_pack(seed, time_budget):
    """Run a repulsion-based packing for a given random seed.

    Returns the best (by objective) valid arrangement found.
    """
    rng = np.random.default_rng(seed)

    # Start from a random jittered grid inside a unit square.
    side = int(np.ceil(np.sqrt(N)))
    base = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                     for i in range(side) for j in range(side)][:N])
    x = base[:, 0] + 0.05 * (rng.random(N) - 0.5)
    y = base[:, 1] + 0.05 * (rng.random(N) - 0.5)
    # equal initial radius
    r = np.full(N, 0.10)

    # We work in a "design" space where the rectangle is [0,1] x [0,1]
    # (unit square, perimeter 4). We push circles apart until they are
    # packed, then scale.
    W_box = 1.0
    H_box = 1.0

    t_end = _T0 + time_budget
    step = 0.01
    best = None
    best_val = -1e9

    it = 0
    while _elapsed() < t_end - 1.0:
        it += 1
        # Repulsion forces (soft, inverse-square-ish)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist2 = dx * dx + dy * dy
        dist2 = np.maximum(dist2, 1e-12)
        dist = np.sqrt(dist2)
        # pairwise repulsion
        fx = np.zeros(N)
        fy = np.zeros(N)
        for i in range(N):
            for j in range(i + 1, N):
                d = dist[i, j]
                if d < 1e-9:
                    continue
                # strong when close, gentle when far
                f = 1.0 / (d * d + 0.001)
                fx[i] += f * dx[i, j] / d
                fy[i] += f * dy[i, j] / d
                fx[j] -= f * dx[i, j] / d
                fy[j] -= f * dy[i, j] / d

        # wall constraints: keep circles inside [0,1] box
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += 5.0 * (r[i] - x[i])
            if x[i] > W_box - r[i]:
                fx[i] += 5.0 * (W_box - r[i] - x[i])
            if y[i] < r[i]:
                fy[i] += 5.0 * (r[i] - y[i])
            if y[i] > H_box - r[i]:
                fy[i] += 5.0 * (H_box - r[i] - y[i])

        # integrate
        x += step * fx
        y += step * fy
        # hard clamp inside box
        x = np.clip(x, r, W_box - r)
        y = np.clip(y, r, H_box - r)

        # occasionally try to grow radii (anneal)
        if it % 200 == 0:
            # check current validity in design space
            arr = np.column_stack([x, y, r])
            ok, _ = _valid(arr)
            if ok:
                # try to increase radius a bit
                r = r * 1.002
            else:
                # shrink a bit to recover
                r = r * 0.998

        # decay step
        step *= 0.9995

        # evaluate
        arr = np.column_stack([x, y, r])
        ok, perim = _valid(arr)
        if ok:
            # scale to perimeter 4
            sarr, s = _scale_to_perimeter(arr)
            val = float(np.sum(sarr[:, 2]))
            if val > best_val:
                best_val = val
                best = sarr.copy()

    if best is None:
        # fallback: return the last scaled arrangement even if not fully valid
        arr = np.column_stack([x, y, r])
        best, _ = _scale_to_perimeter(arr)
    return best


def construct():
    global _T0
    _T0 = time.time()

    # Time budget for the main packing search
    budget = _DEADLINE - _SAFETY - 5.0  # leave 5s for refinement + return

    # Try a few seeds, allocating time proportionally
    seeds = [12345, 67890, 24680, 13579, 97531]
    per_seed = budget / len(seeds)

    best = None
    best_val = -1e9

    for s in seeds:
        if _elapsed() > _DEADLINE - _SAFETY:
            break
        t_budget = min(per_seed, _DEADLINE - _SAFETY - 2.0 - _elapsed())
        if t_budget < 0.5:
            break
        arr = _repulsion_pack(s, t_budget)
        if arr is None:
            continue
        val = _objective(arr)
        if val > best_val:
            best_val = val
            best = arr

    # Local refinement: small random perturbations + re-pack
    if best is not None and _elapsed() < _DEADLINE - _SAFETY - 2.0:
        rng = np.random.default_rng(999)
        refine_budget = min(15.0, _DEADLINE - _SAFETY - 2.0 - _elapsed())
        t_start = time.time()
        while time.time() - t_start < refine_budget and _elapsed() < _DEADLINE - _SAFETY:
            # perturb
            pert = best.copy()
            eps = 0.005
            pert[:, 0] += eps * (rng.random(N) - 0.5)
            pert[:, 1] += eps * (rng.random(N) - 0.5)
            # short repulsion fix
            x, y, r = pert[:, 0], pert[:, 1], pert[:, 2]
            for _ in range(50):
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = np.sqrt(dx * dx + dy * dy)
                fx = np.zeros(N)
                fy = np.zeros(N)
                for i in range(N):
                    for j in range(i + 1, N):
                        d = dist[i, j]
                        if d < 1e-9:
                            continue
                        f = 1.0 / (d * d + 0.001)
                        fx[i] += f * dx[i, j] / d
                        fy[i] += f * dy[i, j] / d
                        fx[j] -= f * dx[i, j] / d
                        fy[j] -= f * dy[i, j] / d
                x += 0.005 * fx
                y += 0.005 * fy
                # keep inside bounding box (recompute)
                xmin = x.min() - r.min()
                xmax = x.max() + r.max()
                ymin = y.min() - r.min()
                ymax = y.max() + r.max()
                # clamp to current box
                x = np.clip(x, xmin + r, xmax - r)
                y = np.clip(y, ymin + r, ymax - r)
            cand = np.column_stack([x, y, r])
            cv = _objective(cand)
            if cv > best_val:
                best_val = cv
                best = cand

    # Final: ensure validity; if not, fall back to a safe grid
    if best is None or not _valid(best)[0]:
        # safe fallback: 5x5 grid scaled to perimeter 4
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)][:N]
        best = np.array(cells)
        # scale to perimeter 4
        best, _ = _scale_to_perimeter(best)

    # Final validity check; if still invalid, use the seed grid
    if not _valid(best)[0]:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)][:N]
        best = np.array(cells)
        best, _ = _scale_to_perimeter(best)

    return best


_T0 = time.time()
