import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s kill

    p = 7
    k = 3  # alpha^2 = 3, non-residue mod 7

    # Precompute field operations tables
    # Represent F_49 elements as (a, b) meaning a + b*alpha
    # Precompute addition and multiplication tables indexed by flat index a*7+b
    def flat_idx(a, b):
        return a * p + b

    def unflat(idx):
        return (idx // p, idx % p)

    # Addition table: add_tab[i][j] = i + j in F_49 (flat index)
    add_tab = np.zeros((p*p, p*p), dtype=np.int32)
    for i in range(p*p):
        ia, ib = unflat(i)
        for j in range(p*p):
            ja, jb = unflat(j)
            add_tab[i, j] = flat_idx((ia + ja) % p, (ib + jb) % p)

    # Multiplication table: mul_tab[i][j] = i * j in F_49 (flat index)
    mul_tab = np.zeros((p*p, p*p), dtype=np.int32)
    for i in range(p*p):
        ia, ib = unflat(i)
        for j in range(p*p):
            ja, jb = unflat(j)
            # (ia + ib*alpha)(ja + jb*alpha) = ia*ja + ia*jb*alpha + ib*ja*alpha + ib*jb*alpha^2
            # = (ia*ja + ib*jb*k) + (ia*jb + ib*ja)*alpha
            ra = (ia * ja + ib * jb * k) % p
            rb = (ia * jb + ib * ja) % p
            mul_tab[i, j] = flat_idx(ra, rb)

    # Identity elements
    zero = flat_idx(0, 0)
    one = flat_idx(1, 0)

    # All points in F_49^2: 2401 points, indexed by (xa, xb, ya, yb)
    # Flat point index: ((xa * p + xb) * p + ya) * p + yb
    def point_flat(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def point_unflat(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)

    total_points = p * p * p * p  # 2401

    # Precompute all lines: for each point and each direction, store the set of points on the line
    # Directions: v = (1, m) for m in F_49 (49 directions) + v = (0, 1) (1 direction) = 50 directions
    # For direction (1, m): line through point (x, y) is {(x + t, y + t*m) : t in F_49}
    # For direction (0, 1): line through point (x, y) is {(x, y + t) : t in F_49}

    # Precompute line membership: lines[dir][point] = set of points on that line through that point
    # Actually, we need: for each point, for each direction, which other points are on the same line
    # Let's precompute: for each direction d and each point x, the line L(d, x) = {x + t*v_d : t in F_49}

    # Number of directions
    num_dirs = p * p + 1  # 49 + 1 = 50

    # For each direction, precompute the line through each point
    # line_points[dir_idx][point_idx] = array of 49 point indices on that line
    # This is 50 * 2401 * 49 = ~5.9 million entries, manageable

    # Actually, let's think differently. For the Nikodym property:
    # For each point x, there must exist a direction d such that all other 48 points on L(d,x) are in N.
    # Equivalently: for each point x, for each direction d, check if all other points on L(d,x) are in N.
    # If any direction works, x is satisfied.

    # Let's precompute for each direction and each point, the list of other points on the line.
    # line_other[dir_idx][point_idx] = array of 48 point indices (excluding the point itself)

    # Direction (0,1): v = (0, 1) in F_49^2
    # x + t*v = (x_x, x_y + t*1) = (x_x, x_y + t)
    # So the line is all points with same x-coordinate, varying y.

    # Direction (1, m): v = (1, m)
    # x + t*v = (x_x + t*1, x_y + t*m) = (x_x + t, x_y + t*m)

    # Precompute line_other as a list of arrays
    line_other = []  # line_other[d][x] = numpy array of 48 point indices

    # Direction 0: v = (0, 1)
    dir_01 = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = point_flat(xa, xb, ya, yb)
                    others = []
                    for ta in range(p):
                        for tb in range(p):
                            if ta == 0 and tb == 0:
                                continue
                            # new point: (xa, xb, ya + ta, yb + tb)
                            nya = (ya + ta) % p
                            nyb = (yb + tb) % p
                            others.append(point_flat(xa, xb, nya, nyb))
                    dir_01.append(np.array(others, dtype=np.int32))
    line_other.append(dir_01)

    # Directions 1..49: v = (1, m) for m in F_49
    for ma in range(p):
        for mb in range(p):
            dir_m = []
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            x_idx = point_flat(xa, xb, ya, yb)
                            others = []
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    # new_x = (xa, xb) + (ta, tb) = (xa+ta, xb+tb)
                                    nxa = (xa + ta) % p
                                    nxb = (xb + tb) % p
                                    # new_y = (ya, yb) + (ta, tb) * (ma, mb)
                                    # t*m = (ta + tb*alpha) * (ma + mb*alpha)
                                    # = (ta*ma + tb*mb*k) + (ta*mb + tb*ma)*alpha
                                    tma = (ta * ma + tb * mb * k) % p
                                    tmb = (ta * mb + tb * ma) % p
                                    nya = (ya + tma) % p
                                    nyb = (yb + tmb) % p
                                    others.append(point_flat(nxa, nxb, nya, nyb))
                            dir_m.append(np.array(others, dtype=np.int32))
            line_other.append(dir_m)

    # Now line_other has shape: num_dirs x total_points, each entry is array of 48 indices

    # Vectorized verification:
    # Given a boolean array `in_N` of shape (total_points,), check if Nikodym property holds.
    # For each point x, check if there exists a direction d such that all 48 other points on L(d,x) are in N.
    # i.e., in_N[line_other[d][x]] are all True.

    def verify_nikodym_fast(in_N):
        """
        in_N: boolean array of shape (total_points,)
        Returns True if Nikodym property holds.
        """
        # For each direction d, compute which points x have all other points on L(d,x) in N.
        # For direction d, point x is "covered by direction d" if all in_N[line_other[d][x]] are True.

        # We can vectorize: for each direction, check all 2401 points at once.
        # line_other[d] is a list of 2401 arrays of 48 elements.
        # Stack them: shape (2401, 48)
        # Then in_N[line_other[d]] has shape (2401, 48)
        # Point x is covered if all True along axis 1.

        for d in range(num_dirs):
            if time.time() - start_time > deadline:
                return False
            # Stack the line_other for this direction
            # line_other[d] is a list of 2401 numpy arrays of shape (48,)
            lo_d = np.array(line_other[d], dtype=np.int32)  # shape (2401, 48)
            # in_N[lo_d] has shape (2401, 48)
            all_in = np.all(in_N[lo_d], axis=1)  # shape (2401,)
            # If any point is not covered by this direction, we need to check other directions
            # Actually, we need: for EACH point, AT LEAST ONE direction covers it.
            # So we need the union over all directions.

            # Let's track which points are still uncovered
            # We'll do this iteratively
            pass

        # Better approach: initialize uncovered = all points
        # For each direction, mark points that are covered by this direction
        # At the end, if uncovered is empty, valid.

        uncovered = np.ones(total_points, dtype=bool)
        for d in range(num_dirs):
            if time.time() - start_time > deadline:
                return False
            lo_d = np.array(line_other[d], dtype=np.int32)  # shape (2401, 48)
            all_in = np.all(in_N[lo_d], axis=1)  # shape (2401,)
            uncovered &= ~all_in
            if not np.any(uncovered):
                return True

        return not np.any(uncovered)

    # Start with a known valid construction:
    # Hermitian unital: U = {(xa, xb, ya, yb) : 2*xa + ya^2 - 3*yb^2 = 0 mod 7}
    # Complement S = {points in U where y_a is a square mod 7}
    # N = F_49^2 \ S

    # Compute initial S
    in_S = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        # Check if ya is a square mod 7: squares are {0, 1, 2, 4}
                        if ya in {0, 1, 2, 4}:
                            in_S[point_flat(xa, xb, ya, yb)] = True

    in_N = ~in_S

    # Verify initial construction
    if not verify_nikodym_fast(in_N):
        # Fallback: return a trivial valid set (all points)
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    # Greedy expansion: try to add more points to S (remove from N)
    # A point can be added to S if removing it from N doesn't break the Nikodym property.
    # i.e., for every point x that currently uses this point's line direction,
    # there must be another direction that still works.

    # More precisely: if we remove point p from N (add to S), then for any point x
    # whose only valid direction d has p on L(d,x) (p != x), x would lose that direction.
    # We need x to have at least one other valid direction.

    # Let's precompute: for each point x, which directions currently work.
    def get_valid_directions(in_N_arr):
        """Return a boolean array of shape (total_points, num_dirs) where
        valid[x, d] = True if all other points on L(d,x) are in N."""
        valid = np.zeros((total_points, num_dirs), dtype=bool)
        for d in range(num_dirs):
            if time.time() - start_time > deadline:
                break
            lo_d = np.array(line_other[d], dtype=np.int32)  # shape (2401, 48)
            all_in = np.all(in_N_arr[lo_d], axis=1)  # shape (2401,)
            valid[:, d] = all_in
        return valid

    # Greedy: repeatedly try to add points to S
    # For efficiency, only consider points that are in the unital (or near it)
    # Actually, let's try all points but prioritize those that are "safe"

    # Get initial valid directions
    valid_dirs = get_valid_directions(in_N)

    # Count how many valid directions each point has
    num_valid = np.sum(valid_dirs, axis=1)  # shape (total_points,)

    # Points with exactly 1 valid direction are "critical" - removing a point from N
    # that lies on their unique valid line would break them.

    # Greedy: try to add points to S in some order
    # A point p can be added to S if for every point x that has p on L(d,x) for its
    # unique valid direction d, x has another valid direction.

    # Simplified greedy: try points in random order, check if adding them keeps validity
    # To speed up, only try points that are in the unital (more likely to be addable)

    # Get unital points
    unital_mask = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        unital_mask[point_flat(xa, xb, ya, yb)] = True

    # Candidate points to add to S: unital points not already in S
    candidates = np.where(unital_mask & ~in_S)[0]

    # Shuffle candidates
    np.random.shuffle(candidates)

    # For each candidate, check if we can add it to S
    # This requires checking if any point loses all valid directions

    # Precompute: for each direction d and point p, which points x have p on L(d,x) (p != x)
    # This is the "reverse" mapping.
    # For direction d, point p is on L(d, x) iff x is on L(d, p) (lines are symmetric in this sense? No.)
    # Actually, L(d, x) = {x + t*v_d : t in F_49}. p is on L(d,x) iff p = x + t*v_d for some t != 0.
    # So x = p + t'*v_d for some t' != 0 (where t' = -t).
    # So the set of x such that p is on L(d,x) is L(d, p) \ {p}.

    # So for each direction d and point p, the affected points are line_other[d][p].

    # When we remove point p from N (add to S), for each direction d, points x in line_other[d][p]
    # lose point p from their line. If p was the only point not in N on L(d,x), then x loses direction d.
    # But actually, we're removing p from N, so p is no longer in N. For x to still have direction d valid,
    # all other points on L(d,x) must be in N. Since p is now not in N, if p is on L(d,x) and p != x,
    # then direction d is no longer valid for x (unless p was already not in N, but we're adding p to S).

    # Wait: we're adding p to S, so p leaves N. For any x such that p is on L(d,x) with p != x,
    # direction d becomes invalid for x (since p is no longer in N).

    # So x loses direction d. If x had only direction d valid, x becomes unsatisfied.

    # Check: for candidate p, for each direction d, for each x in line_other[d][p],
    # if x's only valid direction is d, then adding p to S breaks x.

    # This is expensive to check naively. Let's use a heuristic:
    # Only add p if no point has exactly 1 valid direction and p is on that direction's line.

    # Precompute for each point x with num_valid[x] == 1, what its unique valid direction is
    critical_points = np.where(num_valid == 1)[0]
    if len(critical_points) > 0:
        unique_dir = np.argmax(valid_dirs[critical_points], axis=1)

        # For each critical point x, the "forbidden" points are those on L(unique_dir[x], x) \ {x}
        # If we add any of those to S, x breaks.
        forbidden = np.zeros(total_points, dtype=bool)
        for i, x in enumerate(critical_points):
            d = unique_dir[i]
            # Points on L(d, x) other than x
            for p_idx in line_other[d][x]:
                forbidden[p_idx] = True
    else:
        forbidden = np.zeros(total_points, dtype=bool)

    # Now try to add candidates that are not forbidden
    added_count = 0
    for p_idx in candidates:
        if time.time() - start_time > deadline:
            break
        if in_S[p_idx]:
            continue
        if forbidden[p_idx]:
            continue

        # Try adding p_idx to S
        in_N[p_idx] = False
        in_S[p_idx] = True

        # Check validity
        if verify_nikodym_fast(in_N):
            added_count += 1
            # Update valid_dirs and forbidden
            valid_dirs = get_valid_directions(in_N)
            num_valid = np.sum(valid_dirs, axis=1)
            critical_points = np.where(num_valid == 1)[0]
            if len(critical_points) > 0:
                unique_dir = np.argmax(valid_dirs[critical_points], axis=1)
                forbidden = np.zeros(total_points, dtype=bool)
                for i, x in enumerate(critical_points):
                    d = unique_dir[i]
                    for p_idx2 in line_other[d][x]:
                        forbidden[p_idx2] = True
            else:
                forbidden = np.zeros(total_points, dtype=bool)
        else:
            # Revert
            in_N[p_idx] = True
            in_S[p_idx] = False

    # Build output
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if not in_S[point_flat(xa, xb, ya, yb)]:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
