# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a max-flow / min-cut structural model.

    Structural idea:
      - Directions: 50 (49 slopes m in F_49 + vertical).
      - A point x is "blocked" (cannot be excluded) iff in EVERY direction d,
        the line through x in direction d contains at least one other excluded point.
      - The set of NON-blocked points is always a valid Nikodym set.
      - We grow the excluded set S greedily: add points that cause the fewest
        new saturations (directions becoming fully blocked).

    This is a different modelling family from the unital-based approach:
    instead of a fixed algebraic curve, we model the constraint as a
    bipartite saturation problem (point -> direction) and optimize via
    greedy set-cover-like search with min-cut duality.
    """
    start_time = time.time()
    deadline = start_time + 570  # safety margin

    p = 7
    q = p * p  # 49
    k_const = 3  # alpha^2 = 3

    # --- Field F_49 arithmetic (vectorized where possible) ---
    # Index: idx = a * p + b  for a + b*alpha
    # Point index: pidx = x_idx * q + y_idx  where x_idx, y_idx in [0, 48]
    # Point tuple: (xa, xb, ya, yb) -> x_idx = xa*p+xb, y_idx = ya*p+yb

    # Precompute mul table: mul[idx1, idx2] = idx of product
    mul = np.zeros((q, q), dtype=np.int64)
    for a1 in range(p):
        for b1 in range(p):
            i1 = a1 * p + b1
            for a2 in range(p):
                for b2 in range(p):
                    i2 = a2 * p + b2
                    pa = (a1 * a2 + k_const * b1 * b2) % p
                    pb = (a1 * b2 + a2 * b1) % p
                    mul[i1, i2] = pa * p + pb

    # All 2401 points
    n_pts = q * q  # 2401
    all_pts = []  # list of (xa, xb, ya, yb)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_pts.append((xa, xb, ya, yb))

    # Point index from coords
    def pt_idx(xa, xb, ya, yb):
        return (xa * p + xb) * q + (ya * p + yb)

    # --- Directions: 50 total ---
    # Direction 0: vertical v=(0,1)
    # Direction 1..49: v=(1, m) for m in F_49 (m_idx = 0..48)
    n_dirs = 50

    # Precompute for each point and each direction: the set of other points on the line
    # line_pts[pt][d] = list of point indices on the line through pt in direction d, excluding pt
    # This is 2401 * 50 * 48 entries ~ 5.7M, too large for full storage.
    # Instead, precompute for each direction, the partition of points into lines.

    # For each direction d, build: line_of_pt[d][pt] = line_id, and line_members[d][line_id] = list of pts
    # Then for a point pt in direction d, the other points are line_members[d][line_of_pt[d][pt]] minus pt.

    # More memory-efficient: for each direction, for each point, store the sorted list of
    # other point indices on its line. But that's still large.

    # Alternative: for each direction d, build an array line_id_of_pt[d] of shape (n_pts,),
    # and for each line, a sorted array of member point indices.
    # Then "other points on line" = members minus the point itself.

    # Let's use: for each direction d, a dict mapping line_key -> list of point indices.
    # But we need fast lookup: given pt and direction d, find the line and check if all
    # other members are in S.

    # Efficient approach: for each direction d, for each point pt, precompute a bitmask or
    # sorted list of the 48 other points on its line. Store as a 2D array:
    # line_others[d, pt] = array of 48 point indices (the other points on the line)
    # This is 50 * 2401 * 48 = 5,762,400 int64 values ~ 46 MB. Acceptable.

    line_others = np.zeros((n_dirs, n_pts, q - 1), dtype=np.int64)

    for d in range(n_dirs):
        if d == 0:
            # Vertical: v = (0, 1). Line through (x, y) = {(x, y + t) : t in F_49}
            # Points on same line: same x, different y
            for xa in range(p):
                for xb in range(p):
                    x_idx = xa * p + xb
                    pts_on_line = []
                    for ya in range(p):
                        for yb in range(p):
                            y_idx = ya * p + yb
                            pts_on_line.append(x_idx * q + y_idx)
                    # Sort for determinism
                    pts_on_line.sort()
                    for pt in pts_on_line:
                        # Others = all except self
                        others = [o for o in pts_on_line if o != pt]
                        line_others[d, pt] = others
        else:
            # Slope direction: v = (1, m) where m_idx = d - 1
            m_idx = d - 1
            ma = m_idx // p
            mb = m_idx % p
            # Line through (x0, y0): {(x0 + t, y0 + m*t) : t in F_49}
            # Two points (x1, y1) and (x2, y2) are on same line iff
            # y1 - y0 = m * (x1 - x0)
            # i.e., y1 - m*x1 = y0 - m*x0 (in F_49)
            # So the line key is: y - m*x (in F_49)

            # Group points by line key
            lines = defaultdict(list)
            for i in range(n_pts):
                xa, xb, ya, yb = all_pts[i]
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                # Compute m * x
                mx_idx = mul[m_idx, x_idx]
                # Compute y - m*x
                # y = (ya, yb), mx = (mx_a, mx_b)
                mx_a = mx_idx // p
                mx_b = mx_idx % p
                diff_a = (ya - mx_a) % p
                diff_b = (yb - mx_b) % p
                key = diff_a * p + diff_b
                lines[key].append(i)

            for key, pts_on_line in lines.items():
                pts_on_line.sort()
                for pt in pts_on_line:
                    others = [o for o in pts_on_line if o != pt]
                    line_others[d, pt] = others

    # --- Greedy construction of excluded set S ---
    # State:
    #   in_S[pt] = whether pt is in S (excluded from N)
    #   sat_count[pt][d] = number of OTHER points in S on the line through pt in direction d
    #   A point pt is "blocked" iff sat_count[pt][d] >= 1 for ALL d in 0..49
    #   i.e., min_d sat_count[pt][d] >= 1

    in_S = np.zeros(n_pts, dtype=bool)
    # sat_count[pt, d] = count of other points in S on line through pt in direction d
    sat_count = np.zeros((n_pts, n_dirs), dtype=np.int32)

    # Initially S is empty, all sat_counts are 0, no point is blocked.
    # N = all points, valid Nikodym set (trivially).

    # We want to grow S. A point can be added to S if, after adding, no point becomes
    # "uncoverable". But actually, adding a point to S can only INCREASE sat_counts,
    # which can only make points MORE blocked, not less. So adding to S never breaks
    # the Nikodym property for points already in N... wait, no.

    # Let me re-think. The Nikodym property: for every point x (in N or not), there exists
    # a direction d such that all points on the line through x in direction d, EXCEPT x,
    # are in N.

    # If x is in N: we need a direction d where no other point on the line is in S.
    #   i.e., sat_count[x][d] == 0 for some d.
    # If x is in S: we still need a direction d where no other point on the line is in S.
    #   i.e., sat_count[x][d] == 0 for some d.

    # So the condition for ALL points x is: min_d sat_count[x][d] == 0.
    # A point is "bad" (violates Nikodym) iff min_d sat_count[x][d] >= 1.

    # When we add a point p to S:
    # - For each direction d, for each other point x on the line through p in direction d,
    #   sat_count[x][d] increases by 1.
    # - This can cause some x to become "bad" if sat_count[x][d] was 0 and is now 1,
    #   AND all other directions for x also have sat_count >= 1.

    # So we can only add p if no point becomes bad.

    # Greedy: repeatedly find a point p that can be added to S without making any point bad.
    # To check efficiently, we need to know for each candidate p, which points would become bad.

    # Optimization: maintain for each point x, the number of directions d where sat_count[x][d] == 0.
    # call this zero_dir_count[x]. x is good iff zero_dir_count[x] >= 1.
    # x is bad iff zero_dir_count[x] == 0.

    zero_dir_count = np.zeros(n_pts, dtype=np.int32)
    for x in range(n_pts):
        zero_dir_count[x] = n_dirs  # initially all directions have 0 in S

    # When we add p to S:
    # For each direction d:
    #   For each x in line_others[d, p]:
    #     if sat_count[x, d] == 0: zero_dir_count[x] -= 1
    #     sat_count[x, d] += 1
    #   Also, for p itself:
    #     if sat_count[p, d] == 0: zero_dir_count[p] -= 1
    #     sat_count[p, d] += 1

    # p can be added iff for all affected x (including p), zero_dir_count[x] >= 1 after update.
    # i.e., no x has zero_dir_count[x] drop to 0.

    # Greedy strategy: try all points, pick the one that is "safest" (causes fewest near-misses).
    # For efficiency, use a priority queue or just iterate.

    # Let's try: repeatedly scan all points not in S, find one that can be added.
    # If none can be added, stop.

    # For speed, maintain a list of "candidate" points (not in S).
    # For each candidate, compute the "cost" = number of points x (including p) where
    # zero_dir_count[x] == 1 (i.e., would become 0 if we add p and p is on their line in some direction).

    # Actually, simpler: just try to add each point in a fixed order, and if it fails, move on.
    # Use multiple passes until no progress.

    # Precompute: for each point p and direction d, the list of affected points is line_others[d, p] + [p].
    # We already have line_others.

    def try_add_point(p):
        """Try to add point p to S. Returns True if successful."""
        if in_S[p]:
            return False

        # Check if adding p would make any point bad
        # Affected points: for each d, line_others[d, p] and p itself
        # For each affected x, check if zero_dir_count[x] would drop to 0

        # First, check p itself:
        # p's zero_dir_count would decrease by (number of d where sat_count[p, d] == 0)
        # Actually, for each d, sat_count[p, d] increases by 1 (since p is now in S,
        # and the line through p in direction d has p itself... wait, no.
        # The line through p in direction d contains p and 48 others.
        # When p is added to S, for each OTHER point x on that line, sat_count[x, d] increases.
        # For p itself, sat_count[p, d] does NOT change due to p being added,
        # because sat_count[p, d] counts OTHER points in S on the line, not p itself.
        # So adding p to S only affects sat_count of OTHER points on lines through p.
        # p's own sat_count is unchanged.

        # So affected points are: for each d, the 48 points in line_others[d, p].
        # (p is NOT affected by its own addition to S in terms of sat_count.)

        # But wait: p's zero_dir_count is based on sat_count[p, d], which counts
        # other points in S on lines through p. Adding p to S doesn't change this.
        # So p's zero_dir_count is unchanged.

        # However, we need p to still be "good" (zero_dir_count[p] >= 1) after addition.
        # Since it's unchanged, if it was good before, it's good after.

        # So we only need to check the 48*50 = 2400 other points (with duplicates).

        # For efficiency, collect unique affected points and check them.
        # But 2400 checks per candidate * 2401 candidates = 5.7M checks per pass.
        # With multiple passes, this might be too slow. Let's optimize.

        # Alternative: precompute for each point p, the set of unique affected points.
        # affected[p] = set of x != p such that x is on some line through p.
        # This is just all points except p, since for any x != p, there's a unique
        # direction d such that x is on the line through p in direction d.
        # So affected[p] = all points except p. That's 2400 points.

        # Hmm, that means adding any point p affects all other 2400 points.
        # So the check is: for all x != p, zero_dir_count[x] >= 1 after update.
        # The update: for each d, for each x in line_others[d, p], sat_count[x, d] += 1.
        # zero_dir_count[x] decreases by 1 for each d where sat_count[x, d] was 0.

        # So the condition is: for all x != p, zero_dir_count[x] > (number of d where x is in line_others[d, p] and sat_count[x, d] == 0).

        # This is expensive to check naively. Let's just do it and hope it's fast enough.

        # Actually, let's use a different strategy: instead of checking all 2400 points,
        # maintain a count of "bad" points (zero_dir_count == 0). Initially 0.
        # When adding p, compute how many points would become bad. If 0, add p.

        # To compute this efficiently, for each x != p, compute delta_zero[x] =
        #   number of d where x is in line_others[d, p] and sat_count[x, d] == 0.
        # x becomes bad iff zero_dir_count[x] == delta_zero[x].

        # We can precompute for each pair (p, x) the set of directions d where x is in line_others[d, p].
        # But that's 2401 * 2400 * 50 = 288M entries. Too large.

        # Let's just do the naive check but with early termination.

        # For each x != p:
        #   Compute delta = sum over d of [x in line_others[d, p] and sat_count[x, d] == 0]
        #   If zero_dir_count[x] <= delta, then x would become bad. Return False.

        # To speed up, precompute for each p, for each x, the list of d where x is in line_others[d, p].
        # This is a 2401 x 2401 x (variable length) structure. Too large.

        # Alternative: for each p, iterate over all d, and for each x in line_others[d, p],
        # check if sat_count[x, d] == 0. Accumulate delta[x]. Then check all x.

        # This is 50 * 48 = 2400 iterations per candidate. With 2401 candidates, 5.7M per pass.
        # In Python, this might take ~1-2 seconds per pass. With 10 passes, ~20s. Acceptable.

        # Let's implement it.

        delta = np.zeros(n_pts, dtype=np.int32)
        for d in range(n_dirs):
            for x in line_others[d, p]:
                if sat_count[x, d] == 0:
                    delta[x] += 1

        # Check if any x would become bad
        for x in range(n_pts):
            if x == p:
                continue
            if zero_dir_count[x] <= delta[x]:
                return False

        # Safe to add p
        in_S[p] = True
        for d in range(n_dirs):
            for x in line_others[d, p]:
                if sat_count[x, d] == 0:
                    zero_dir_count[x] -= 1
                sat_count[x, d] += 1

        return True

    # Greedy: repeatedly try to add points until no more can be added.
    # Order: try points in a random or structured order.
    # Use multiple passes with different orderings.

    import random
    rng = random.Random(42)

    max_passes = 200
    for pass_num in range(max_passes):
        if time.time() > deadline - 60:
            break

        candidates = [i for i in range(n_pts) if not in_S[i]]
        if not candidates:
            break

        # Shuffle for diversity
        rng.shuffle(candidates)

        added_any = False
        for p in candidates:
            if time.time() > deadline - 60:
                break
            if in_S[p]:
                continue
            if try_add_point(p):
                added_any = True

        if not added_any:
            break

    # Build result: N = all points not in S
    n_in_N = n_pts - np.sum(in_S)
    pts_in_N = [all_pts[i] for i in range(n_pts) if not in_S[i]]

    if not pts_in_N:
        # Should not happen, but safety
        pts_in_N = [all_pts[0]]

    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts_in_N], dtype=np.int64)

    # Reduce mod 7 (already in range, but ensure)
    result = result % 7

    return result

# EVOLVE-BLOCK-END
