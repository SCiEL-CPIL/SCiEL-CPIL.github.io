"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0
_TIME_MARGIN = 5.0  # seconds of safety margin before the kill limit


def _time_left():
    return _DEADLINE - _TIME_MARGIN - time.time()


def _max_radius(x, y, radii, positions, i, n):
    """Largest radius for circle i at (x,y) that stays in [0,1]^2 and is disjoint from others."""
    r = min(x, y, 1.0 - x, 1.0 - y)
    if r <= 0:
        return 0.0
    for j in range(n):
        if j == i:
            continue
        d = np.hypot(x - positions[j, 0], y - positions[j, 1])
        cand = d - radii[j]
        if cand < r:
            r = cand
    return max(r, 0.0)


def _resolve_overlaps(positions, radii, n, max_iters=200):
    """Hard-constraint push: repeatedly push apart any overlapping pair.
    Returns True if a feasible (non-overlapping) configuration was reached."""
    for _ in range(max_iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                d = np.hypot(dx, dy)
                need = radii[i] + radii[j]
                if d < need - 1e-9:
                    # push them apart along the separation direction
                    if d < 1e-12:
                        ang = (i + 0.37 * j) * 0.618
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx / d, dy / d
                    # split the deficit between the two circles (equal mass)
                    push = (need - d) * 0.5 + 1e-9
                    positions[i, 0] -= ux * push
                    positions[i, 1] -= uy * push
                    positions[j, 0] += ux * push
                    positions[j, 1] += uy * push
                    moved = True
        if not moved:
            return True
    # last attempt: check feasibility
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(positions[j, 0] - positions[i, 0], positions[j, 1] - positions[i, 1])
            if d < radii[i] + radii[j] - 1e-8:
                return False
    return True


def _clamp_to_square(positions, radii, n):
    """Nudge positions so each circle lies inside [0,1]^2."""
    for i in range(n):
        r = radii[i]
        positions[i, 0] = min(max(positions[i, 0], r), 1.0 - r)
        positions[i, 1] = min(max(positions[i, 1], r), 1.0 - r)


def _is_valid(positions, radii, n):
    for i in range(n):
        r = radii[i]
        if r < 0:
            return False
        if positions[i, 0] < -1e-7 or positions[i, 1] < -1e-7:
            return False
        if positions[i, 0] + r > 1.0 + 1e-7 or positions[i, 1] + r > 1.0 + 1e-7:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(positions[j, 0] - positions[i, 0], positions[j, 1] - positions[i, 1])
            if d < radii[i] + radii[j] - 1e-7:
                return False
    return True


def _grow_and_push(positions, radii, n, max_rounds=400):
    """Repeatedly grow the most constrained circle to its max feasible radius,
    then hard-resolve any resulting overlaps. Returns (positions, radii)."""
    for _ in range(max_rounds):
        # find the circle with the smallest possible growth
        best_i = -1
        best_gain = -1.0
        for i in range(n):
            rmax = _max_radius(positions[i, 0], positions[i, 1], radii, positions, i, n)
            gain = rmax - radii[i]
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_gain <= 1e-7:
            break
        # grow that circle
        radii[best_i] = radii[best_i] + best_gain
        # hard-resolve overlaps caused by the growth
        _resolve_overlaps(positions, radii, n)
        # clamp to square (pushing may have pushed a circle out)
        _clamp_to_square(positions, radii, n)
        # after clamping, re-resolve (clamping can re-create overlaps)
        _resolve_overlaps(positions, radii, n)
        _clamp_to_square(positions, radii, n)
    return positions, radii


def _initial_grid(n, rng):
    """Grid of equal small circles (jittered), all feasible."""
    side = int(np.ceil(np.sqrt(n)))
    positions = np.zeros((n, 2))
    radii = np.zeros(n)
    base_r = 0.5 / side - 1e-4
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            positions[idx, 0] = (i + 0.5) / side + (rng.random() - 0.5) * 0.02
            positions[idx, 1] = (j + 0.5) / side + (rng.random() - 0.5) * 0.02
            radii[idx] = base_r
            idx += 1
    _clamp_to_square(positions, radii, n)
    _resolve_overlaps(positions, radii, n)
    _clamp_to_square(positions, radii, n)
    return positions, radii


def _initial_random(n, rng):
    """Random placement of small circles, feasible after resolution."""
    positions = np.zeros((n, 2))
    radii = np.zeros(n)
    base_r = 0.05
    for _ in range(50):
        positions[:, 0] = rng.uniform(0.1, 0.9, size=n)
        positions[:, 1] = rng.uniform(0.1, 0.9, size=n)
        radii[:] = base_r
        if _resolve_overlaps(positions, radii, n):
            _clamp_to_square(positions, radii, n)
            if _is_valid(positions, radii, n):
                return positions, radii
    # fallback to grid
    return _initial_grid(n, rng)


def construct():
    start = time.time()
    best_positions = None
    best_radii = None
    best_sum = -1.0

    n = N
    seed = 0
    while _time_left() > 3.0:
        rng = np.random.default_rng(seed)
        seed += 1

        # choose initial placement: alternate grid / random
        if seed % 3 == 0:
            positions, radii = _initial_grid(n, rng)
        else:
            positions, radii = _initial_random(n, rng)

        # grow-and-push
        positions, radii = _grow_and_push(positions, radii, n, max_rounds=300)

        # final polish: a few more rounds with tighter resolution
        for _ in range(3):
            positions, radii = _grow_and_push(positions, radii, n, max_rounds=60)

        # validate; if invalid, shrink slightly and re-resolve
        if not _is_valid(positions, radii, n):
            # shrink all radii by a small factor until valid
            scale = 1.0
            for _ in range(40):
                scale *= 0.98
                radii2 = radii * scale
                if _resolve_overlaps(positions, radii2, n) and _is_valid(positions, radii2, n):
                    radii = radii2
                    break
            if not _is_valid(positions, radii, n):
                # give up this restart
                continue

        cur_sum = float(np.sum(radii))
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_positions = positions.copy()
            best_radii = radii.copy()

        if _time_left() < 3.0:
            break

    if best_positions is None:
        # ultimate fallback: trivial grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:n])

    result = np.column_stack([best_positions, best_radii])
    return result
# EVOLVE-BLOCK-END
