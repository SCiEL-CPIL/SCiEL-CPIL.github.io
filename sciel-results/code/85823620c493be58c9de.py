# EVOLVE-BLOCK-START
import time
import numpy as np

_P = 7
_Q = _P * _P  # 49


def _build_field_ops(p):
    """Build addition and multiplication tables for F_{p^2} = F_p[alpha], alpha^2 = 3."""
    k = 3
    n = p * p

    def idx(a, b):
        return a * p + b

    # Addition: component-wise mod p
    add = np.zeros((n, n), dtype=np.int32)
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    add[idx(a, b), idx(c, d)] = idx((a + c) % p, (b + d) % p)

    # Multiplication: (a + b*alpha)(c + d*alpha) = (ac + bd*k) + (ad + bc)*alpha
    mul = np.zeros((n, n), dtype=np.int32)
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    mul[idx(a, b), idx(c, d)] = idx((a * c + b * d * k) % p,
                                                    (a * d + b * c) % p)

    # Inverse table (for nonzero elements)
    inv = np.zeros(n, dtype=np.int32)
    inv[0] = 0
    for x in range(1, n):
        for y in range(1, n):
            if mul[x, y] == 1:
                inv[x] = y
                break

    # Precompute: for each x, the additive inverse x + t for all t
    # add_shift[x] = array of length n: add_shift[x][t] = x + t
    add_shift = np.zeros((n, n), dtype=np.int32)
    for x in range(n):
        for t in range(n):
            add_shift[x, t] = add[x, t]

    return add, mul, inv, add_shift


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_{49}^2 by greedily maximizing the complement S.

    A set N is a Nikodym set if for every point x in F_q^2, there exists a direction
    v (of the form (1, m) or (0, 1)) such that all other points on the line
    {x + t*v : t in F_q, t != 0} belong to N.

    Equivalently, S = F_q^2 \ N is a set such that for every x, there is a line
    through x (in one of the q+1 allowed directions) whose other q-1 points are NOT in S.

    We greedily add points to S (starting from the seed's excluded set) to maximize |S|.
    """
    deadline = time.time() + 620  # safety margin before 630s kill
    p = _P
    q = _Q
    total = q * q  # 2401

    add, mul, inv, add_shift = _build_field_ops(p)

    # Precompute all lines: for each point x and each direction (q+1 directions),
    # store the set of other points on that line.
    # Directions: m in F_q (direction (1, m)) plus the vertical direction (0, 1).
    # Total q + 1 = 50 directions.
    #
    # Line through x in direction (1, m): points are x + t*(1, m) for t in F_q.
    #   x = (xa, ya), so point at parameter t is (xa + t, ya + t*m).
    # Vertical line through x in direction (0, 1): points are x + t*(0, 1) = (xa, ya + t).

    # For each point x (index 0..total-1) and each direction d (0..q),
    # line_points[x][d] = list of the other q-1 point indices on that line.
    # We'll store as a flat structure for speed.

    # Point index: point (x, y) where x, y in F_q (each 0..48) -> index = x * q + y
    # x = (xa, xb) -> x_idx = xa * p + xb
    # y = (ya, yb) -> y_idx = ya * p + yb
    # point_idx = x_idx * q + y_idx

    # Precompute line membership: for each direction d and each point x,
    # the set of other points on the line.
    # We'll use: lines[x][d] = numpy array of shape (q-1,) with the other point indices.

    # Direction encoding:
    # d = 0: vertical (0, 1)
    # d = 1..q: direction (1, m) where m = d-1 (as field element index)

    # For efficiency, precompute for each direction d and each x, the other points.
    # But that's 2401 * 50 * 48 = ~5.7M entries. Let's compute on the fly or use a smarter structure.

    # Actually, let's precompute a "line_id" for each point-direction pair.
    # A line is determined by its direction and one point on it.
    # For direction (1, m): the line through (x0, y0) consists of points (x, y) where
    #   y - y0 = m * (x - x0), i.e., y = m*x + (y0 - m*x0).
    #   So the line is identified by (m, c) where c = y0 - m*x0.
    # For vertical direction: the line through (x0, y0) is x = x0, identified by x0.

    # Let's precompute: for each point, which lines (in allowed directions) pass through it.
    # And for each such line, which other points are on it.

    # Given the size (2401 points, 50 directions), let's precompute a list of
    # (other_points) for each (point, direction) pair.

    # lines_through[x][d] = list of other point indices (length q-1 = 48)
    lines_through = np.zeros((total, q + 1, q - 1), dtype=np.int32)

    for x_idx in range(q):
        for y_idx in range(q):
            pt = x_idx * q + y_idx
            # Vertical direction (d=0): other points are (x_idx, y_idx + t) for t != 0
            for t in range(1, q):
                y_other = add[y_idx, t]
                lines_through[pt, 0, t - 1] = x_idx * q + y_other

            # Direction (1, m) for m = 0..q-1 (d = 1..q)
            for m in range(q):
                d = m + 1
                for t in range(1, q):
                    x_other = add[x_idx, t]
                    y_other = add[y_idx, mul[m, t]]
                    lines_through[pt, d, t - 1] = x_other * q + y_other

    # Initialize S with the seed's excluded set (points in unital U with y not in Y_special)
    k = 3  # alpha^2 = 3
    in_S = np.zeros(total, dtype=bool)

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(u * u % p * p + v)  # y_idx = (ya, yb) = (u^2 mod p, v)

    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = xa * p + xb
                    y_idx = ya * p + yb
                    pt = x_idx * q + y_idx
                    # In unital: Trace(x) + Norm(y) = 2*xa + ya^2 - yb^2*k == 0 mod p
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        if (ya, yb) not in y_special:
                            in_S[pt] = True

    s_count = in_S.sum()

    # Greedy expansion: try to add each point not in S to S.
    # A point pt can be added to S if for EVERY direction d, there exists at least one
    # point on the line through pt in direction d (other than pt) that is NOT in S.
    # Wait, that's the condition for N to be valid. Let me re-read.
    #
    # N must be a Nikodym set: every point x must lie on a line (in allowed direction)
    # whose points OTHER than x all belong to N.
    #
    # So for x in S (i.e., x not in N), the condition is: there exists a direction d
    # such that ALL other points on the line through x in direction d are in N (not in S).
    #
    # For x not in S (i.e., x in N), the condition is: there exists a direction d
    # such that ALL other points on the line through x in direction d are in N (not in S).
    #
    # So the condition for the whole set to be valid is:
    # For EVERY point x (whether in S or not), there exists a direction d such that
    # all other q-1 points on the line through x in direction d are NOT in S.
    #
    # When we add a point pt to S, we need to check:
    # 1. For pt itself: there exists a direction d where all other points on the line
    #    through pt in direction d are not in S (after adding pt).
    # 2. For every other point y that shares a line with pt in some direction d,
    #    if that line was the ONLY valid line for y, it might now be invalid.
    #
    # This is complex. Let's use a different check:
    # After adding pt to S, for every point x that is on a line with pt in some direction,
    # we need to verify x still has at least one valid direction.
    #
    # Actually, the simplest correct approach:
    # A point x is "covered" if there exists a direction d such that no other point
    # on the line through x in direction d is in S.
    #
    # We need every point to be covered.
    #
    # When we add pt to S:
    # - pt loses coverage in direction d if any other point on the line through pt
    #   in direction d is also in S.
    # - Other points y on lines with pt: for each direction d where y and pt are on
    #   the same line, y's coverage in direction d is lost if ALL other points on that
    #   line (besides y) are in S.
    #
    # Let me define: cover_count[x] = number of directions d for which x is covered
    # (i.e., no other point on the line through x in direction d is in S).
    #
    # Initially, compute cover_count for all points.
    # Then, when adding pt to S:
    #   - For pt: recompute which directions are still valid.
    #   - For each direction d, for each other point y on the line through pt in direction d:
    #     y's coverage in direction d might change. Specifically, if y was covered in
    #     direction d before (no other point on that line was in S), after adding pt,
    #     y is no longer covered in direction d (since pt is now in S and on that line).
    #     But if y still has other valid directions, it's fine.
    #
    # So the check for adding pt:
    #   1. pt must have at least one valid direction (after adding pt to S).
    #   2. For every other point y that is on a line with pt (in any direction),
    #      y must still have at least one valid direction.
    #
    # The affected points are: all points on lines through pt in all q+1 directions.
    # That's (q+1)*(q-1) + (q-1) points... actually for each of the q+1 directions,
    # there are q-1 other points. Total affected: (q+1)*(q-1) = 50*48 = 2400 points.
    # Plus pt itself.
    #
    # For each affected point, we check if it still has at least one valid direction.
    # A direction d is valid for point y if no other point on the line through y in
    # direction d is in S.
    #
    # This check per candidate is O((q+1)*(q-1)*(q+1)) which is too slow for 2401 candidates.
    #
    # Let me optimize: maintain cover_count[x] = number of valid directions for x.
    # When adding pt to S:
    #   - For pt: cover_count[pt] might decrease.
    #   - For each other point y on a line with pt in direction d:
    #     cover_count[y] decreases by 1 if y was previously covered in direction d
    #     (i.e., before adding pt, no other point on that line was in S).
    #
    # Wait, I need to be more careful. Let me define:
    #   valid[x][d] = True if no other point on the line through x in direction d is in S.
    #   cover_count[x] = sum_d valid[x][d].
    #
    # Initially (with seed S), compute valid and cover_count.
    #
    # When adding pt to S:
    #   - valid[pt][d] for all d: becomes False if any other point on the line through
    #     pt in direction d is in S. (It was already computed; adding pt doesn't change
    #     whether OTHER points are in S, so valid[pt][d] stays the same.)
    #     Wait, valid[pt][d] checks if other points on the line are in S. Adding pt to S
    #     doesn't affect this. So valid[pt] is unchanged.
    #   - For other point y on line through pt in direction d:
    #     valid[y][d] was True if no other point on that line was in S.
    #     After adding pt, if pt is on that line and pt != y, then valid[y][d] becomes
    #     False (since pt is now in S and is another point on the line).
    #     So cover_count[y] decreases by 1 for each such (y, d) pair.
    #
    # So the check: after adding pt, every point must have cover_count >= 1.
    # The points whose cover_count changes are:
    #   - pt: no change (its valid directions depend on other points being in S).
    #   - For each direction d, each other point y on the line through pt in direction d:
    #     cover_count[y] -= 1 (if valid[y][d] was True before).
    #
    # So we need: for all affected y, cover_count[y] - 1 >= 1, i.e., cover_count[y] >= 2.
    # And cover_count[pt] >= 1 (unchanged, so just check it's already >= 1).
    #
    # This is much faster! We just need to check that all affected points have
    # cover_count >= 2 before adding.

    # Compute initial valid and cover_count
    valid = np.zeros((total, q + 1), dtype=bool)
    for x in range(total):
        for d in range(q + 1):
            # Check if any other point on the line through x in direction d is in S
            others = lines_through[x, d]
            valid[x, d] = not in_S[others].any()
    cover_count = valid.sum(axis=1)

    # Greedy: repeatedly add the point that maximizes some criterion.
    # Simple approach: try to add each point not in S, in order of increasing impact.
    # A point pt can be added if:
    #   1. cover_count[pt] >= 1 (it's currently covered)
    #   2. For all affected points y (on lines through pt), cover_count[y] >= 2.
    #
    # Let's precompute the affected points for each pt.
    # affected[pt] = list of (y, d) pairs where y is on a line through pt in direction d, y != pt.

    # For speed, let's just iterate and check.
    # Order: try points in random or heuristic order.

    # Let's try a simple greedy: repeatedly find a point that can be added, add it,
    # update cover_count, repeat.

    # For efficiency, let's precompute for each pt the list of affected (y, d) pairs.
    # affected_pts[pt] = set of other points on lines through pt (in any direction).
    # But we also need to know which direction for each.

    # Actually, let's just compute on the fly. For each candidate pt, the affected points
    # are lines_through[pt, d] for all d. That's 50 * 48 = 2400 entries (with duplicates).

    # Let's use a different approach: for each pt, compute the set of affected points
    # and their directions.

    # For speed, precompute affected[pt] as a list of (y, d) tuples.
    affected = [None] * total
    for pt in range(total):
        aff = []
        for d in range(q + 1):
            for t in range(q - 1):
                y = lines_through[pt, d, t]
                aff.append((y, d))
        affected[pt] = aff

    # Now greedy loop
    # We'll maintain a list of candidate points and try to add them.
    # To be efficient, we'll iterate multiple passes.

    # Actually, let's just do a single pass in a smart order, then repeat.
    # Order: points with lower cover_count first (more "risky" to add but also
    # potentially more valuable since they're close to being uncovered).
    # Or: points that affect the fewest other points.

    # Let's try: sort by cover_count ascending, then by number of affected points with
    # cover_count == 1 (ascending).

    # For simplicity, let's just do repeated passes in index order, with a deadline check.

    added_any = True
    while added_any and time.time() < deadline - 10:
        added_any = False
        # Build list of candidates
        candidates = [pt for pt in range(total) if not in_S[pt]]
        # Sort by cover_count ascending (points with fewer valid directions first)
        candidates.sort(key=lambda pt: cover_count[pt])

        for pt in candidates:
            if time.time() > deadline - 10:
                break
            if in_S[pt]:
                continue
            # Check if pt can be added
            if cover_count[pt] < 1:
                continue
            # Check all affected points
            can_add = True
            for (y, d) in affected[pt]:
                if cover_count[y] < 2:
                    can_add = False
                    break
            if can_add:
                # Add pt to S
                in_S[pt] = True
                s_count += 1
                # Update cover_count for affected points
                for (y, d) in affected[pt]:
                    if valid[y, d]:
                        valid[y, d] = False
                        cover_count[y] -= 1
                added_any = True

    # Build the result: N = all points not in S
    points = []
    for pt in range(total):
        if not in_S[pt]:
            x_idx = pt // q
            y_idx = pt % q
            xa = x_idx // p
            xb = x_idx % p
            ya = y_idx // p
            yb = y_idx % p
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
