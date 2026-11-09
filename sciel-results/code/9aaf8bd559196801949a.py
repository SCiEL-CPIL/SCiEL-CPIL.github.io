# EVOLVE-BLOCK-START
import numpy as np
import time

_N = 10
_TOL = 1e-7
_DEADLINE = 590.0  # Wall-clock safety margin (evaluator kills at 630s)


def _initial_solution(seed: int):
    """
    Deterministic, verified-feasible layout: 2 large (0.45) + 8 small (0.25).
    This configuration yields a sum of side lengths of 2*0.45 + 8*0.25 = 0.9 + 2.0 = 2.9.
    The grid baseline (10 squares of 0.333) yields ~3.33, but the grid is often suboptimal 
    for maximization if it forces small squares. Wait, 1/3 * 10 = 3.33.
    Let's re-evaluate the baseline.
    Grid: k=4 (ceil(sqrt(10))). side=0.25. Sum = 10 * 0.25 = 2.5.
    Inspiration 1 baseline: 2*0.45 + 8*0.25 = 0.9 + 2.0 = 2.9.
    So Inspiration 1's baseline is better than the simple grid.
    
    We can potentially do even better with a different configuration, but this is a solid start.
    """
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
    """
    Return (objective, max_violation).
    Objective: sum of side lengths.
    Violation: max of boundary violations and overlap violations.
    """
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
    # Overlap if |dx| < t AND |dy| < t.
    # Violation magnitude is how much they overlap in the "forbidden" region.
    # We use a product of the violations in each axis to create a smooth penalty.
    ov_x = np.maximum(0.0, t - np.abs(dx))
    ov_y = np.maximum(0.0, t - np.abs(dy))
    ov = ov_x * ov_y
    ov = np.triu(ov, 1)
    max_ov = float(np.max(ov))

    return obj, max(max_bv, max_ov)


def _local_search(centers, sides, start_time, deadline, rng, step0=0.05):
    """
    Adaptive multi-variable local search with a light feasibility penalty.
    """
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
            # Perturb x-coordinate of square idx
            delta = rng.normal(0.0, step)
            new_centers = centers.copy()
            new_centers[idx, 0] += delta
            new_sides = sides
        elif idx < 2 * n:
            # Perturb y-coordinate of square (idx - n)
            delta = rng.normal(0.0, step)
            new_centers = centers
            new_sides = sides.copy()
            new_sides[idx - n] += delta
        else:
            # Perturb both x and y of a random center
            i = idx - 2 * n
            new_centers = centers.copy()
            new_centers[i, 0] += rng.normal(0.0, step)
            new_centers[i, 1] += rng.normal(0.0, step)
            new_sides = sides

        nobj, nviol = _evaluate(new_centers, new_sides)
        score = nobj - penalty * nviol
        current_score = obj - penalty * viol
        
        if score > current_score:
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


def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid according to the strict task contract."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


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
    
    # Ensure baseline is valid before using it as the "best"
    if _is_valid(base_centers, base_sides):
        best_centers = base_centers.copy()
        best_sides = base_sides.copy()
        best_obj = base_obj
    else:
        # Fallback to a simple grid if baseline somehow fails (shouldn't happen)
        k = 4
        side = 0.25
        idx = np.arange(_N)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(_N, side, dtype=np.float64)
        best_obj = np.sum(best_sides)

    # Multi-start local search.
    restart = 0
    # We want to try several restarts. Each restart runs local search until deadline.
    # To allow multiple restarts, we should allocate a portion of the time to each.
    # Let's aim for ~10 restarts, so ~50s each? Or just run until deadline?
    # Inspiration 1 runs _local_search until deadline. This means only 1 restart effectively 
    # unless _local_search terminates early. It doesn't terminate early on success, only on time.
    # So we need to modify the loop to allow multiple attempts.
    
    # Strategy: Run local search for a fixed duration, or until no improvement for a while.
    # Or simply run multiple iterations of local search with a time slice.
    
    max_restarts = 20
    time_per_restart = (_DEADLINE - 10.0) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        rng = np.random.default_rng(1000 + restart)
        
        # Initialize current solution for this restart
        if restart == 0:
            c, s = base_centers, base_sides
        else:
            # Occasionally start from the current best with a jitter
            if restart % 3 == 0:
                c = best_centers + rng.normal(0.0, 0.02, size=best_centers.shape)
                s = np.clip(best_sides + rng.normal(0.0, 0.01, size=_N), 1e-4, None)
            else:
                c, s = _initial_solution(restart)
        
        # Run local search for a specific time slice
        local_deadline = min(time.time() + time_per_restart, deadline - 5.0)
        c, s, obj = _local_search(c, s, time.time(), local_deadline, rng)
        
        if obj > best_obj:
            # Verify before updating best
            if _is_valid(c, s):
                best_obj = obj
                best_centers = c.copy()
                best_sides = s.copy()
            else:
                # If invalid, try to repair or just skip
                pass

    # Final validation and clamping to ensure strict feasibility.
    # Clamp centers/sides to keep squares inside [0,1]^2 with margin.
    sides = np.clip(best_sides, 1e-5, None)
    cx = np.clip(best_centers[:, 0], sides * 0.5, 1.0 - sides * 0.5)
    cy = np.clip(best_centers[:, 1], sides * 0.5, 1.0 - sides * 0.5)
    centers = np.stack([cx, cy], axis=1)

    # Verify feasibility; if any violation, fall back to the pure baseline.
    if not _is_valid(centers, sides):
        # Use the clean baseline (guaranteed feasible).
        centers = np.array([
            (0.225, 0.775), (0.775, 0.775),
            (0.125, 0.125), (0.375, 0.125), (0.625, 0.125), (0.875, 0.125),
            (0.125, 0.375), (0.375, 0.375), (0.625, 0.375), (0.875, 0.375),
        ], dtype=np.float64)
        sides = np.array([0.45, 0.45, 0.25, 0.25, 0.25, 0.25,
                          0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        
        # Double check fallback
        if not _is_valid(centers, sides):
             # Absolute last resort: tiny squares in corners
             centers = np.zeros((10,2))
             sides = np.full(10, 0.01)
             # Place them far apart
             for i in range(10):
                 centers[i,0] = (i % 5) * 0.2
                 centers[i,1] = (i // 5) * 0.2

    return centers, sides


# EVOLVE-BLOCK-END
