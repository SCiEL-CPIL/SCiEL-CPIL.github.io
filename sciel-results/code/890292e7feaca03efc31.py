# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.

    Strategy:
    1. Start with a known valid Nikodym set based on a Hermitian unital
       construction (size ~ p^4 - p^2*(p+1)/2).
    2. Iteratively remove points from N that are not essential for the
       Nikodym property. A point can be removed if every point in F_q^2
       still has a valid line (in direction v=(1,m) for some m in F_q,
       or v=(0,1)) whose other points are all in N.
    3. The result is a minimal (or near-minimal) Nikodym set.
    """
    deadline = time.time() + 620  # safety margin before 630s limit

    p = 7
    q = p * p  # 49

    # Field F_q = F_p[alpha], alpha^2 = k
    # Find smallest quadratic non-residue mod p
    k_field = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_field = c
            break
    if k_field is None:
        k_field = 3  # fallback

    # Represent elements of F_q as integers 0..48: element (a, b) -> a + b*p
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Field arithmetic in F_q
    def f_add(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        return to_int((xa + ya) % p, (xb + yb) % p)

    def f_sub(x, y):
        return f_add(x, f_neg(y))

    def f_neg(x):
        xa, xb = to_pair(x)
        return to_int((-xa) % p, (-xb) % p)

    def f_mul(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        # (xa + xb*alpha)(ya + yb*alpha) = xa*ya + (xa*yb + xb*ya)*alpha + xb*yb*alpha^2
        # = xa*ya + xb*yb*k_field + (xa*yb + xb*ya)*alpha
        real = (xa * ya + xb * yb * k_field) % p
        imag = (xa * yb + xb * ya) % p
        return to_int(real, imag)

    def f_inv(x):
        if x == 0:
            return 0
        xa, xb = to_pair(x)
        # Inverse of (xa + xb*alpha) = (xa - xb*alpha)/(xa^2 - xb^2*k)
        denom = (xa * xa - xb * xb * k_field) % p
        if denom == 0:
            return 0
        denom_inv = pow(denom, p - 2, p)
        real = (xa * denom_inv) % p
        imag = (-xb * denom_inv) % p
        return to_int(real, imag)

    def f_div(x, y):
        return f_mul(x, f_inv(y))

    # Directions: v = (1, m) for m in F_q, or v = (0, 1)
    # All valid directions
    directions = []
    for m in range(q):
        directions.append((1, m))  # v = (1, m)
    directions.append((0, 1))  # vertical

    # All points in F_q^2 as integer pairs
    all_points = [(x, y) for x in range(q) for y in range(q)]
    point_index = {(x, y): i for i, (x, y) in enumerate(all_points)}

    # Build line for each point in each direction
    # line[point_idx][dir_idx] = list of point indices on the line through point_idx in direction dir_idx
    # Actually, let's be smarter: for each direction and each point, we need the set of other points on the line.
    # But that's q^2 * (q+1) * q which is too much. Instead, let's compute on the fly or use a different approach.

    # Alternative approach:
    # For a Nikodym set N, every point x must have at least one direction v such that
    # all points on the line through x in direction v, except x itself, are in N.
    # 
    # To check if we can remove a point p from N:
    # For every point x in F_q^2, there must exist a direction v such that:
    #   - all points on line(x, v) except x are in N (after removal of p)
    # This means: for every x, at least one direction v satisfies:
    #   line(x, v) \ {x} ⊆ N \ {p} (if p != x) or line(x, v) \ {x} ⊆ N (if p == x, but then x's line must not use p)
    #
    # This is expensive to check naively. Let's think of a different approach.
    #
    # Better approach:
    # 1. Start with the base Nikodym set.
    # 2. For each point in N (in some order), try to remove it.
    # 3. To check if removal is valid: for each point x, check if there exists a direction v
    #    such that all other points on line(x,v) are still in N (after removal).
    #
    # Optimization: maintain for each point x and direction v a count of how many points on
    # line(x,v) \ {x} are NOT in N. If this count is 0, then x is "covered" by direction v.
    # A point x is valid if at least one direction v has count 0.
    #
    # When we remove point p from N:
    # - For every x such that p is on line(x,v) \ {x}, the count for (x,v) increases by 1.
    # - We need to check that no x loses all its valid directions.

    # Let's precompute lines
    # For each direction v = (vx, vy), and each point x = (x1, x2),
    # the line is {x + t*v : t in F_q}
    # The other points are x + t*v for t != 0

    # Precompute: for each direction index d, for each point index i,
    # the list of other points on the line.
    # But that's 50 * 2401 * 48 ≈ 5.8 million entries, each a list of 48 ints. Too much memory?
    # 5.8M * 48 * 4 bytes ≈ 1.1 GB. Too much.
    #
    # Alternative: don't precompute. Compute on the fly but cache smartly.
    # Or: use the count approach but compute the affected (x, v) pairs on the fly.

    # Let's use the count approach with lazy computation.
    # not_in_N_count[x_idx][d_idx] = number of points on line(x, v_d) \ {x} that are not in N

    # Initialize N
    in_N = [True] * (q * q)

    # Build the base Nikodym set
    # Hermitian unital: U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - b^2*k
    # Blocking set B = points of U whose y-coordinate has y_a a square mod p
    # Excluded set S = U \ B
    # N = complement of S

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

    # Determine which points are excluded (not in N)
    excluded = set()
    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * k_field) % p
            if (trace_x + norm_y) % p == 0:
                # In unital
                if y_idx not in y_special:
                    # Excluded
                    excluded.add((x_idx, y_idx))

    # Initialize N: all points except excluded
    for (x_idx, y_idx) in excluded:
        in_N[point_index[(x_idx, y_idx)]] = False

    # Initialize not_in_N_count
    # For each point x and direction v, count points on line(x,v)\{x} not in N
    n_dir = len(directions)
    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)

    # Compute initial counts
    for d_idx, (vx, vy) in enumerate(directions):
        for x_idx in range(q):
            for y_idx in range(q):
                point = (x_idx, y_idx)
                p_idx = point_index[point]
                count = 0
                for t in range(1, q):
                    tx = f_add(x_idx, f_mul(t, vx))
                    ty = f_add(y_idx, f_mul(t, vy))
                    t_point = (tx, ty)
                    if t_point not in point_index:
                        continue
                    t_idx = point_index[t_point]
                    if not in_N[t_idx]:
                        count += 1
                not_in_N_count[p_idx, d_idx] = count

    # A point is "valid" if min over directions of not_in_N_count[point, d] == 0
    # We want to remove points from N. After removal, update counts.
    # Try to remove points in order of "least essential" - points that are on many lines
    # where the line has alternative coverage.

    # For efficiency, track for each point how many directions "cover" it
    cover_count = np.array([np.sum(not_in_N_count[i] == 0) for i in range(q * q)], dtype=np.int32)

    # Try to remove points. Order: points in N with highest cover_count (most redundant)
    # Actually, we want to remove points that don't break coverage.
    # Let's iterate: for each point in N (sorted by some heuristic), try to remove.

    # Get indices of points currently in N
    n_indices = [i for i in range(q * q) if in_N[i]]

    # Sort by: points that are excluded from fewer lines (i.e., points that are on lines with
    # high not_in_N_count for most directions) - these are less critical.
    # Actually, let's try: sort by the number of (x, d) pairs where this point is the
    # only non-N point on line(x, d). Hard to compute.
    # Simpler: just try all points in N in some order.

    # For speed, let's try a greedy approach: repeatedly remove the point that
    # causes the fewest new uncovered points. But that's expensive.
    # Let's just iterate through points and try to remove each one.

    # To make this tractable, we'll use a heuristic: try to remove points in order of
    # increasing "criticality". A point p is critical if removing it would uncover
    # some point x (i.e., x had exactly one covering direction, and that direction's
    # line contains p).

    # For now, let's just iterate through all points in N and try to remove each.
    # If removal is valid, do it.

    # Precompute for each point p, the set of (x, d) pairs where p is on line(x, d)\{x}
    # This is: for each x and d, p is on line(x,d)\{x} iff p = x + t*d for some t != 0
    # i.e., x = p - t*d for some t != 0
    # So for a given p, the (x, d) pairs are: for each d, for each t in 1..q-1,
    #   x = p - t*d

    # Let's precompute this mapping
    # affects[p_idx] = list of (x_idx, d_idx) pairs
    affects = [[] for _ in range(q * q)]
    for p_idx in range(q * q):
        px, py = point_index_key = (None, None)
        # Decode p_idx to (px, py)
        # Actually, we need to invert point_index
        # Let's build reverse mapping
        pass

    # Build reverse mapping
    idx_to_point = {}
    for pt, idx in point_index.items():
        idx_to_point[idx] = pt

    for p_idx in range(q * q):
        px, py = idx_to_point[p_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tx = f_sub(px, f_mul(t, vx))
                ty = f_sub(py, f_mul(t, vy))
                if (tx, ty) in point_index:
                    x_idx = point_index[(tx, ty)]
                    affects[p_idx].append((x_idx, d_idx))

    # Now try to remove points
    # For each point in N, check if removal is valid
    # Removal of p is valid if for all (x, d) in affects[p], 
    #   not_in_N_count[x, d] > 0 after removal (i.e., there's another non-N point on that line)
    # But we also need x to still have at least one covering direction after removal.

    # This is getting complex. Let's simplify: 
    # Try removing points one by one. For each candidate p in N:
    #   1. Tentatively remove p: in_N[p] = False
    #   2. For each (x, d) in affects[p]: not_in_N_count[x, d] += 1
    #   3. Check if any x now has no covering direction (all not_in_N_count[x, d] > 0)
    #   4. If valid, keep removal. Else, revert.

    # For step 3, we need to know for each x how many directions cover it.
    # Maintain cover_count[x] = number of d with not_in_N_count[x, d] == 0

    # Let's implement this with early termination

    removed = 0
    for p_idx in n_indices:
        if not in_N[p_idx]:
            continue
        if time.time() > deadline - 30:
            break

        # Tentatively remove p_idx
        in_N[p_idx] = False
        affected_counts = []
        for (x, d) in affects[p_idx]:
            old_val = not_in_N_count[x, d]
            not_in_N_count[x, d] = old_val + 1
            affected_counts.append((x, d, old_val))

        # Check validity: for all x in affected, cover_count[x] > 0
        # But cover_count needs updating. Let's track.
        invalid = False
        new_cover_counts = {}
        for (x, d, old_val) in affected_counts:
            # cover_count[x] decreases by 1 if old_val == 0
            if old_val == 0:
                if x not in new_cover_counts:
                    new_cover_counts[x] = cover_count[x]
                new_cover_counts[x] -= 1
                if new_cover_counts[x] <= 0:
                    invalid = True
                    break

        if invalid:
            # Revert
            for (x, d, old_val) in affected_counts:
                not_in_N_count[x, d] = old_val
            in_N[p_idx] = True
        else:
            # Keep removal, update cover_count
            for (x, d, old_val) in affected_counts:
                if old_val == 0:
                    cover_count[x] -= 1
            removed += 1

    # Build result
    points = []
    for i in range(q * q):
        if in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback: return the base set
        points = []
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                if (x_idx, y_idx) not in excluded:
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
