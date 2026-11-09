# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(centers, sides, tol=1e-7):
    """Vectorized feasibility check. Returns True if all constraints hold."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside unit square
    if np.any(centers[:, 0] - sides / 2 < -tol):
        return False
    if np.any(centers[:, 0] + sides / 2 > 1.0 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol):
        return False
    if np.any(centers[:, 1] + sides / 2 > 1.0 + tol):
        return False
    # No overlaps (vectorized over all pairs)
    i = np.arange(n)
    j = np.arange(n)
    I, J = np.meshgrid(i, j, indexing='ij')
    mask = I < J
    ii = I[mask]
    jj = J[mask]
    t = (sides[ii] + sides[jj]) / 2.0 - tol
    dx = np.abs(centers[ii, 0] - centers[jj, 0])
    dy = np.abs(centers[ii, 1] - centers[jj, 1])
    if np.any((dx < t) & (dy < t)):
        return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _repair(centers, sides, tol=1e-7):
    """Deterministically fix out-of-bounds and reduce overlaps (best-effort)."""
    centers = centers.copy()
    sides = sides.copy()
    n = len(sides)
    # Pull back into the unit square
    for i in range(n):
        s = sides[i]
        lo = s / 2.0
        hi = 1.0 - s / 2.0
        centers[i, 0] = min(max(centers[i, 0], lo), hi)
        centers[i, 1] = min(max(centers[i, 1], lo), hi)
    # Push overlapping pairs apart
    for _ in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                if abs(dx) < t and abs(dy) < t:
                    # separate along the smaller axis
                    if abs(dx) < abs(dy):
                        push = (t - abs(dx)) + 1e-6
                        if dx >= 0:
                            centers[i, 0] += push / 2.0
                            centers[j, 0] -= push / 2.0
                        else:
                            centers[i, 0] -= push / 2.0
                            centers[j, 0] += push / 2.0
                    else:
                        push = (t - abs(dy)) + 1e-6
                        if dy >= 0:
                            centers[i, 1] += push / 2.0
                            centers[j, 1] -= push / 2.0
                        else:
                            centers[i, 1] -= push / 2.0
                            centers[j, 1] += push / 2.0
        # re-clamp
        for i in range(n):
            s = sides[i]
            lo = s / 2.0
            hi = 1.0 - s / 2.0
            centers[i, 0] = min(max(centers[i, 0], lo), hi)
            centers[i, 1] = min(max(centers[i, 1], lo), hi)
    return centers, sides


def _sa(centers, sides, rng, deadline, best_val, best_centers, best_sides):
    """Simulated annealing from a given feasible start; returns best found."""
    n = len(sides)
    tol = 1e-7
    current_val = _objective(centers, sides)
    best_val = max(best_val, current_val)
    best_centers = centers.copy() if current_val >= best_val else best_centers
    best_sides = sides.copy() if current_val >= best_val else best_sides

    T_start = 0.05
    T_end = 1e-7
    max_iters = 400000
    base_step = 0.03

    for it in range(max_iters):
        if time.time() > deadline - 25.0:
            break
        progress = it / max_iters
        T = T_start * (T_end / T_start) ** progress
        step = base_step * (1.0 - 0.85 * progress)  # shrink step over time

        i = rng.integers(n)
        new_centers = centers.copy()
        new_sides = sides.copy()

        new_centers[i, 0] += rng.normal(0.0, step)
        new_centers[i, 1] += rng.normal(0.0, step)
        new_sides[i] += rng.normal(0.0, step * 0.6)

        # Clamp to valid range
        ns = new_sides[i]
        if ns <= 0.0:
            ns = 0.005
        if ns > 1.0:
            ns = 1.0
        new_sides[i] = ns
        lo = ns / 2.0
        hi = 1.0 - ns / 2.0
        new_centers[i, 0] = min(max(new_centers[i, 0], lo), hi)
        new_centers[i, 1] = min(max(new_centers[i, 1], lo), hi)

        if _is_valid(new_centers, new_sides, tol):
            new_val = _objective(new_centers, new_sides)
            delta = new_val - current_val
            if delta >= 0 or rng.random() < np.exp(-delta / T):
                centers = new_centers
                sides = new_sides
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_centers = centers.copy()
                    best_sides = sides.copy()
        else:
            # Try a light repair; if it becomes feasible, consider it
            rc, rs = _repair(new_centers, new_sides, tol)
            if _is_valid(rc, rs, tol):
                new_val = _objective(rc, rs)
                delta = new_val - current_val
                if delta >= 0 or rng.random() < np.exp(-delta / T):
                    centers = rc
                    sides = rs
                    current_val = new_val
                    if current_val > best_val:
                        best_val = current_val
                        best_centers = centers.copy()
                        best_sides = sides.copy()
    return best_val, best_centers, best_sides


def _seed_grid3():
    """3x3 grid (9 squares of side 1/3) plus a 10th small square in a corner gap."""
    k = 3
    side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(9, side, dtype=np.float64)
    centers = np.vstack([centers, np.array([side / 2.0, side / 2.0])])
    sides = np.concatenate([sides, np.array([0.33333333])])
    return centers, sides


def _seed_large_strip():
    """One large square plus a strip of smaller squares along the edges."""
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    # Large square in the corner
    s0 = 0.5
    centers[0] = [s0 / 2.0, s0 / 2.0]
    sides[0] = s0
    # Fill the remaining L-shaped region with 9 small squares
    s = 1.0 / 6.0
    positions = [
        (0.5 + s / 2, s / 2), (0.5 + s / 2, 0.5 + s / 2), (0.5 + s / 2, 1.0 - s / 2),
        (1.0 - s / 2, s / 2), (1.0 - s / 2, 0.5 + s / 2), (1.0 - s / 2, 1.0 - s / 2),
        (s / 2, 1.0 - s / 2), (0.5 + s / 2, 1.0 - s / 2 - 0.0), (s / 2, 0.5 + s / 2),
    ]
    for k in range(9):
        centers[1 + k] = positions[k]
        sides[1 + k] = s
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place 10 axis-aligned squares with
    pairwise disjoint interiors inside the unit square [0,1]^2 so as to
    MAXIMIZE the sum of their side lengths.

    Strategy: multi-start simulated annealing. We build several diverse,
    feasible initial packings (a 3x3 grid + a small square, and a
    large-square + strip layout), then run a fast vectorized simulated
    annealing local search from each, with an adaptive step size and a
    deterministic push-apart repair to recover near-feasible states. The
    best feasible packing found is returned.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin before the 630s kill

    best_val = 0.0
    best_centers = None
    best_sides = None

    seeds = [_seed_grid3(), _seed_large_strip()]

    # Ensure each seed is feasible; repair if needed
    feasible_seeds = []
    for c, s in seeds:
        if not _is_valid(c, s, tol):
            c, s = _repair(c, s, tol)
        if _is_valid(c, s, tol):
            feasible_seeds.append((c, s))

    # Fallback to the plain grid if repair failed
    if not feasible_seeds:
        k = 3
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        feasible_seeds.append((centers, sides))

    for idx, (c, s) in enumerate(feasible_seeds):
        if time.time() > deadline - 60.0:
            break
        rng = np.random.default_rng(1000 + idx)
        best_val, best_centers, best_sides = _sa(
            c, s, rng, deadline, best_val, best_centers, best_sides
        )

    # Final safety: if somehow nothing was found, return a valid grid
    if best_centers is None:
        k = 3
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers, best_sides


# EVOLVE-BLOCK-END
