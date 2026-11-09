# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from scipy.optimize import linprog


def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in the unit square to maximize the sum of side
    lengths.

    Strategy:
      1. Run a deterministic multi-start local search (gradient-free) over the
         continuous variables (centers, sides) with a smooth soft-penalty for
         overlaps.  This produces a set of candidate topologies (sign patterns
         of horizontal/vertical separation between pairs).
      2. For each candidate topology, build an exact linear program:
             maximize  sum_i s_i
             subject to  s_i > 0
                         cx_i - s_i/2 >= -tol
                         cx_i + s_i/2 <= 1 + tol
                         cy_i - s_i/2 >= -tol
                         cy_i + s_i/2 <= 1 + tol
                         For each pair (i,j):
                             sign_x[i,j] * (cx_i - cx_j) >= (s_i + s_j)/2 - tol
                             sign_y[i,j] * (cy_i - cy_j) >= (s_i + s_j)/2 - tol
         where sign_x, sign_y are fixed from the candidate topology.  The LP
         has a unique optimum for a fixed feasible topology, so it escapes the
         local minima that plague the heuristic search.
      3. Return the best feasible solution found.

    A deadline guard returns the best result before the 630 s wall-clock limit.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 600.0
    safety = 30.0  # return at least this many seconds before the hard limit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def check_valid(centers, sides, tol=1e-7):
        """Return True if (centers, sides) satisfies the public contract."""
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
            return False
        if np.any(sides <= 0):
            return False
        # inside the unit square (with tolerance)
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
                return False
            if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
                return False
        # pairwise non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and \
                   abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def extract_topology(centers, sides, tol=1e-7):
        """
        Determine the separation sign pattern from a feasible packing.
        For each pair (i,j) we need at least one axis where the squares are
        separated.  We pick the axis with the larger gap and record its sign.
        Returns (sign_x, sign_y, axis) where axis[i,j] in {0,1} tells which
        axis was used; sign is +1 or -1.
        """
        sign_x = np.zeros((n, n), dtype=np.int8)
        sign_y = np.zeros((n, n), dtype=np.int8)
        axis = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                gx = centers[i, 0] - centers[j, 0]
                gy = centers[i, 1] - centers[j, 1]
                # Pick the axis where separation is largest in magnitude.
                if abs(gx) >= abs(gy):
                    # separate in x
                    if gx >= 0:
                        sign_x[i, j] = 1
                        sign_x[j, i] = -1
                    else:
                        sign_x[i, j] = -1
                        sign_x[j, i] = 1
                    axis[i, j] = axis[j, i] = 0
                else:
                    # separate in y
                    if gy >= 0:
                        sign_y[i, j] = 1
                        sign_y[j, i] = -1
                    else:
                        sign_y[i, j] = -1
                        sign_y[j, i] = 1
                    axis[i, j] = axis[j, i] = 1
        return sign_x, sign_y, axis

    def solve_lp(sign_x, sign_y, axis, tol=1e-7, max_iter=None):
        """
        Solve the exact LP for a fixed topology.
        Variables: x = [cx_0..cx_{n-1}, cy_0..cy_{n-1}, s_0..s_{n-1}]  (3n vars)
        Objective: maximize sum s_i  ->  minimize -sum s_i
        Returns (centers, sides, status) or (None, None, status).
        """
        m = 3 * n
        # objective
        c = np.zeros(m)
        c[2 * n:3 * n] = -1.0

        # A_ub x <= b_ub
        A_rows = []
        b_rows = []
        # s_i > 0  ->  -s_i <= 0
        for i in range(n):
            row = np.zeros(m)
            row[2 * n + i] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)

        # inside unit square: cx_i - s_i/2 >= -tol  ->  -cx_i + s_i/2 <= tol
        for i in range(n):
            row = np.zeros(m)
            row[i] = -1.0
            row[2 * n + i] = 0.5
            A_rows.append(row)
            b_rows.append(tol)
            row = np.zeros(m)
            row[i] = 1.0
            row[2 * n + i] = 0.5
            A_rows.append(row)
            b_rows.append(1 + tol)
            row = np.zeros(m)
            row[n + i] = -1.0
            row[2 * n + i] = 0.5
            A_rows.append(row)
            b_rows.append(tol)
            row = np.zeros(m)
            row[n + i] = 1.0
            row[2 * n + i] = 0.5
            A_rows.append(row)
            b_rows.append(1 + tol)

        # pairwise separation constraints
        for i in range(n):
            for j in range(i + 1, n):
                if axis[i, j] == 0:
                    # separate in x: sign_x[i,j]*(cx_i - cx_j) >= (s_i+s_j)/2 - tol
                    #  -> -sign_x[i,j]*cx_i + sign_x[i,j]*cx_j + 0.5*s_i + 0.5*s_j <= tol
                    row = np.zeros(m)
                    row[i] = -sign_x[i, j]
                    row[j] = sign_x[i, j]
                    row[2 * n + i] = 0.5
                    row[2 * n + j] = 0.5
                    A_rows.append(row)
                    b_rows.append(tol)
                else:
                    row = np.zeros(m)
                    row[n + i] = -sign_y[i, j]
                    row[n + j] = sign_y[i, j]
                    row[2 * n + i] = 0.5
                    row[2 * n + j] = 0.5
                    A_rows.append(row)
                    b_rows.append(tol)

        A_ub = np.array(A_rows, dtype=np.float64)
        b_ub = np.array(b_rows, dtype=np.float64)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method="highs")
        if res.status != 0:
            return None, None, res.status
        x = res.x
        centers = np.column_stack([x[:n], x[n:2 * n]])
        sides = x[2 * n:3 * n]
        # sanity: all sides positive
        if np.any(sides <= 0):
            return None, None, -1
        return centers, sides, 0

    # ------------------------------------------------------------------
    # Phase 0: seed packings (deterministic)
    # ------------------------------------------------------------------
    seeds = []

    # (a) 4x3 grid (12 cells, use 10)
    side = 0.25
    cols = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    rows = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    cx = (np.array(cols) + 0.5) * side
    cy = (np.array(rows) + 0.5) * side
    seeds.append((np.column_stack([cx, cy]), np.full(n, side)))

    # (b) 3x4 grid
    side = 1.0 / 3
    cols = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    rows = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    cx = (np.array(cols) + 0.5) * side
    cy = (np.array(rows) + 0.5) * side
    seeds.append((np.column_stack([cx, cy]), np.full(n, side)))

    # (c) Two large squares in corners + 8 smaller
    # Big squares of side 0.5 at (0.25,0.25) and (0.75,0.75)
    # Remaining 8 squares fill the gaps with side ~ 0.25
    centers_c = np.array([
        [0.25, 0.25], [0.75, 0.75],
        [0.25, 0.75], [0.75, 0.25],
        [0.125, 0.5], [0.375, 0.5],
        [0.5, 0.125], [0.5, 0.375],
        [0.625, 0.5], [0.875, 0.5],
    ])
    sides_c = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    seeds.append((centers_c, sides_c))

    # (d) One big square (side 0.5) at bottom-left, 9 smaller (side 1/6 ~ 0.1667)
    # Actually: big square side 0.5 at (0.25,0.25). Top strip height 0.5: 5 squares
    # of side 0.5? No, only 5*0.5=2.5 > 1. Let's do side 0.2 in top: 5 squares.
    # Right strip width 0.5, height 0.5: 2 squares of side 0.25? 
    # Let's just make a reasonable seed:
    centers_d = np.array([
        [0.25, 0.25],   # big
        [0.2, 0.75], [0.4, 0.75], [0.6, 0.75], [0.8, 0.75],  # top row
        [0.75, 0.375], [0.75, 0.625],  # right column
        [0.125, 0.5], [0.375, 0.5], [0.625, 0.5],  # middle
    ])
    sides_d = np.array([0.5, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.25, 0.25])
    seeds.append((centers_d, sides_d))

    # (e) Random-ish deterministic perturbation of the 4x3 grid
    rng = np.random.RandomState(42)
    base_c, base_s = seeds[0]
    for trial in range(5):
        c = base_c + (rng.rand(n, 2) - 0.5) * 0.05
        s = base_s * (1 + (rng.rand(n) - 0.5) * 0.1)
        # clamp to keep inside
        for i in range(n):
            s[i] = min(s[i], 0.3)
            c[i, 0] = np.clip(c[i, 0], s[i] / 2, 1 - s[i] / 2)
            c[i, 1] = np.clip(c[i, 1], s[i] / 2, 1 - s[i] / 2)
        seeds.append((c, s))

    # ------------------------------------------------------------------
    # Phase 1: local search (smooth soft-penalty)
    # ------------------------------------------------------------------
    def objective_and_grad(params):
        """
        params: (3n,) = [cx, cy, s]
        Returns (score, penalty) where score = -sum(s) + penalty.
        penalty is a smooth approximation of overlap + boundary violations.
        """
        cx = params[:n]
        cy = params[n:2 * n]
        s = params[2 * n:3 * n]
        # boundary penalty
        pen = 0.0
        for i in range(n):
            # left
            v = s[i] / 2 - cx[i]
            if v > 0:
                pen += v * v
            # right
            v = cx[i] + s[i] / 2 - 1
            if v > 0:
                pen += v * v
            # bottom
            v = s[i] / 2 - cy[i]
            if v > 0:
                pen += v * v
            # top
            v = cy[i] + s[i] / 2 - 1
            if v > 0:
                pen += v * v
        # overlap penalty (smooth): for each pair, if both axes overlap, penalize
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                # overlap in x if |dx| < t, in y if |dy| < t
                # smooth: use softplus-like
                ox = t - abs(dx)
                oy = t - abs(dy)
                if ox > 0 and oy > 0:
                    # both overlap -> penalty
                    pen += (ox * oy)
                elif ox > 0:
                    # only x overlap but y separated: small penalty to encourage separation
                    pen += 0.1 * ox * ox
                elif oy > 0:
                    pen += 0.1 * oy * oy
        score = -np.sum(s) + 100.0 * pen
        return score, pen

    def local_search(centers0, sides0, max_iters=2000, step0=0.01):
        """Simple coordinate-descent / gradient-free local search."""
        params = np.concatenate([centers0.ravel(), sides0])
        best_score, _ = objective_and_grad(params)
        best_params = params.copy()
        step = step0
        for it in range(max_iters):
            improved = False
            for _ in range(3):  # 3 random perturbations per iteration
                trial = params + (np.random.rand(3 * n) - 0.5) * 2 * step
                # clamp sides to positive
                trial[2 * n:] = np.maximum(trial[2 * n:], 1e-4)
                # clamp centers
                for i in range(n):
                    si = trial[2 * n + i]
                    trial[i] = np.clip(trial[i], si / 2 - 1e-3, 1 - si / 2 + 1e-3)
                    trial[n + i] = np.clip(trial[n + i], si / 2 - 1e-3, 1 - si / 2 + 1e-3)
                sc, pen = objective_and_grad(trial)
                if sc < best_score:
                    best_score = sc
                    best_params = trial.copy()
                    improved = True
            if improved:
                step *= 0.95
            else:
                step *= 0.8
                if step < 1e-6:
                    break
        return best_params, best_score

    # Run local search from each seed
    candidates = []
    for (c0, s0) in seeds:
        if time.time() > deadline - safety:
            break
        # Only run local search if seed is roughly feasible
        if check_valid(c0, s0):
            params, score = local_search(c0, s0, max_iters=1000, step0=0.02)
            centers = np.column_stack([params[:n], params[n:2 * n]])
            sides = params[2 * n:3 * n]
            if check_valid(centers, sides):
                candidates.append((centers, sides, float(np.sum(sides))))

    # ------------------------------------------------------------------
    # Phase 2: LP polish for each candidate topology
    # ------------------------------------------------------------------
    best_value = 0.0
    best_result = None

    # Also include the seeds themselves as candidates for LP
    all_candidates = list(candidates)
    for (c0, s0) in seeds:
        if check_valid(c0, s0):
            all_candidates.append((c0, s0, float(np.sum(s0))))

    # Deduplicate topologies
    seen_topologies = set()
    for (centers, sides, val) in all_candidates:
        if time.time() > deadline - safety:
            break
        try:
            sign_x, sign_y, axis = extract_topology(centers, sides)
        except Exception:
            continue
        # Create a hashable key
        key = (tuple(sign_x.ravel()), tuple(sign_y.ravel()), tuple(axis.ravel()))
        if key in seen_topologies:
            continue
        seen_topologies.add(key)

        try:
            c_lp, s_lp, status = solve_lp(sign_x, sign_y, axis)
        except Exception:
            continue
        if c_lp is None:
            continue
        if not check_valid(c_lp, s_lp):
            continue
        val = float(np.sum(s_lp))
        if val > best_value:
            best_value = val
            best_result = (c_lp, s_lp)

    # ------------------------------------------------------------------
    # Phase 3: If we have time, try more LP topologies from perturbed candidates
    # ------------------------------------------------------------------
    if best_result is not None and time.time() < deadline - safety:
        # Perturb the best result and re-extract topology
        rng2 = np.random.RandomState(123)
        for trial in range(20):
            if time.time() > deadline - safety:
                break
            c0 = best_result[0] + (rng2.rand(n, 2) - 0.5) * 0.02
            s0 = best_result[1] * (1 + (rng2.rand(n) - 0.5) * 0.05)
            for i in range(n):
                s0[i] = max(s0[i], 1e-4)
                c0[i, 0] = np.clip(c0[i, 0], s0[i] / 2, 1 - s0[i] / 2)
                c0[i, 1] = np.clip(c0[i, 1], s0[i] / 2, 1 - s0[i] / 2)
            if not check_valid(c0, s0):
                continue
            try:
                sign_x, sign_y, axis = extract_topology(c0, s0)
            except Exception:
                continue
            key = (tuple(sign_x.ravel()), tuple(sign_y.ravel()), tuple(axis.ravel()))
            if key in seen_topologies:
                continue
            seen_topologies.add(key)
            try:
                c_lp, s_lp, status = solve_lp(sign_x, sign_y, axis)
            except Exception:
                continue
            if c_lp is None or not check_valid(c_lp, s_lp):
                continue
            val = float(np.sum(s_lp))
            if val > best_value:
                best_value = val
                best_result = (c_lp, s_lp)

    # ------------------------------------------------------------------
    # Final fallback: return the best found, or a known valid grid
    # ------------------------------------------------------------------
    if best_result is None:
        # Fallback: 4x3 grid (known valid)
        side = 0.25
        cols = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        rows = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
        cx = (np.array(cols) + 0.5) * side
        cy = (np.array(rows) + 0.5) * side
        centers = np.column_stack([cx, cy]).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    centers, sides = best_result
    centers = centers.astype(np.float64)
    sides = sides.astype(np.float64)

    # Final validity check
    if not check_valid(centers, sides):
        # Fallback to grid
        side = 0.25
        cols = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        rows = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
        cx = (np.array(cols) + 0.5) * side
        cy = (np.array(rows) + 0.5) * side
        centers = np.column_stack([cx, cy]).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)

    return centers, sides


# EVOLVE-BLOCK-END
