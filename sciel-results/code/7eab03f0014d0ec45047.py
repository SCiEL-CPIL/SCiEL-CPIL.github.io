# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 10
_TOL = 1e-7
_DEADLINE = 590.0  # wall-clock safety margin (evaluator kills at 630s)


def _initial_solution(seed: int):
    """Deterministic, verified-feasible layout: 2 large (0.45) + 8 small (0.25)."""
    rng = np.random.default_rng(seed)
    centers = np.empty((_N, 2), dtype=np.float64)
    sides = np.empty(_N, dtype=np.float64)

    # Two large squares in the top corners.
    centers[0] = (0.225, 0.775)
    centers[1] = (0.775, 0.775)
    sides[0] = 0.45
    sides[1] = 0.45

    # Eight small squares filling the rest.
    small = np.array([
        (0.125, 0.125), (0.375, 0.125), (0.625, 0.125), (0.875, 0.125),
        (0.125, 0.375), (0.375, 0.375), (0.625, 0.375), (0.875, 0.375),
    ], dtype=np.float64)
    centers[2:] = small
    sides[2:] = 0.25

    # Tiny perturbation for variety across restarts (keeps feasibility margin).
    centers += rng.normal(0.0, 0.001, size=centers.shape)
    sides += rng.normal(0.0, 0.0005, size=_N)
    return centers, sides


def _evaluate(centers, sides):
    """Return (objective, max_violation). Higher objective + low violation is better."""
    cx = centers[:, 0]
    cy = centers[:, 1]
    obj = float(np.sum(sides))

    # Boundary violations.
    lo_x = cx - sides * 0.5
    hi_x = cx + sides * 0.5
    lo_y = cy - sides * 0.5
    hi_y = cy + sides * 0.5
    bv = np.maximum(0.0, np.maximum(np.maximum(-lo_x, hi_x - 1.0),
                                    np.maximum(-lo_y, hi_y - 1.0)))
    max_bv = float(np.max(bv))

    # Pairwise overlap violations.
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    t = 0.5 * (sides[:, None] + sides[None, :]) - _TOL
    ov = np.maximum(0.0, t - np.abs(dx)) * np.maximum(0.0, t - np.abs(dy))
    ov = np.triu(ov, 1)
    max_ov = float(np.max(ov))

    return obj, max(max_bv, max_ov)


def _local_search(centers, sides, start_time, deadline, rng, step0=0.05):
    """Adaptive multi-variable local search with a light feasibility penalty."""
    obj, viol = _evaluate(centers, sides)
    best_obj = obj
    best_centers = centers.copy()
    best_sides = sides.copy()
    step = step0
    n = _N
    dim = 2 * n
    penalty = 1e3

    while time.time() < deadline:
        # Random coordinate perturbation.
        idx = rng.integers(0, dim)
        if idx < n:
            delta = rng.normal(0.0, step)
            new_centers = centers.copy()
            new_centers[idx, 0] += delta
            new_sides = sides
        elif idx < 2 * n:
            delta = rng.normal(0.0, step)
            new_centers = centers
            new_sides = sides.copy()
            new_sides[idx - n] += delta
        else:
            # Perturb both x and y of a random center.
            i = idx - 2 * n
            new_centers = centers.copy()
            new_centers[i, 0] += rng.normal(0.0, step)
            new_centers[i, 1] += rng.normal(0.0, step)
            new_sides = sides

        nobj, nviol = _evaluate(new_centers, new_sides)
        score = nobj - penalty * nviol
        if score > obj - penalty * viol:
            centers = new_centers
            sides = new_sides
            obj = nobj
            viol = nviol
            step *= 1.05
            if viol < 1e-9 and obj > best_obj:
                best_obj = obj
                best_centers = centers.copy()
                best_sides = sides.copy()
        else:
            step *= 0.95
            if step < 1e-6:
                step = 1e-6

    # Return best feasible solution found.
    bobj, bviol = _evaluate(best_centers, best_sides)
    if bviol > 1e-7:
        # Fallback to current if best is infeasible.
        if viol <= 1e-7:
            return centers, sides, obj
        # Otherwise return best anyway (shouldn't happen often).
    return best_centers, best_sides, bobj


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square to maximize the sum of
    side lengths, with pairwise disjoint interiors (touching allowed).

    Returns:
        centers: np.ndarray of shape (10, 2), (x, y) center of each square.
        sides:   np.ndarray of shape (10,), side length of each square.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Start from the verified baseline.
    base_centers, base_sides = _initial_solution(0)
    base_obj, base_viol = _evaluate(base_centers, base_sides)
    best_centers = base_centers.copy()
    best_sides = base_sides.copy()
    best_obj = base_obj

    # Multi-start local search.
    restart = 0
    while time.time() < deadline - 30.0:  # keep 30s margin
        rng = np.random.default_rng(1000 + restart)
        c, s = _initial_solution(restart)
        # Occasionally start from the current best with a jitter.
        if restart % 3 == 0 and restart > 0:
            c = best_centers + rng.normal(0.0, 0.02, size=best_centers.shape)
            s = np.clip(best_sides + rng.normal(0.0, 0.01, size=_N), 1e-4, None)

        c, s, obj = _local_search(c, s, time.time(), deadline, rng)
        if obj > best_obj:
            best_obj = obj
            best_centers = c
            best_sides = s
        restart += 1

    # Final validation and clamping to ensure strict feasibility.
    # Clamp centers/sides to keep squares inside [0,1]^2 with margin.
    sides = np.clip(best_sides, 1e-5, None)
    cx = np.clip(best_centers[:, 0], sides * 0.5, 1.0 - sides * 0.5)
    cy = np.clip(best_centers[:, 1], sides * 0.5, 1.0 - sides * 0.5)
    centers = np.stack([cx, cy], axis=1)

    # Verify feasibility; if any violation, fall back to the pure baseline.
    obj, viol = _evaluate(centers, sides)
    if viol > 1e-7:
        # Use the clean baseline (guaranteed feasible).
        bc, bs = _initial_solution(0)
        # Clean baseline without perturbation.
        centers = np.array([
            (0.225, 0.775), (0.775, 0.775),
            (0.125, 0.125), (0.375, 0.125), (0.625, 0.125), (0.875, 0.125),
            (0.125, 0.375), (0.375, 0.375), (0.625, 0.375), (0.875, 0.375),
        ], dtype=np.float64)
        sides = np.array([0.45, 0.45, 0.25, 0.25, 0.25, 0.25,
                          0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    return centers, sides

# EVOLVE-BLOCK-END
