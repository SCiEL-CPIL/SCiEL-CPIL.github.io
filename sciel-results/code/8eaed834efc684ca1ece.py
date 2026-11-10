# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3).

    The current seed uses the Hermitian unital with the 'skip step 2.5' blocking
    construction. Here we refine within that family by searching for a smaller
    excluded set S (larger complement) that still yields a valid Nikodym set.

    Strategy:
      1. Build the baseline excluded set S0 = { (x,y) in U : y not in Y0 }
         where U is the Hermitian unital and Y0 = {(u^2 mod 7, v)}.
      2. Try to enlarge S (exclude more points) by removing points from N
         while maintaining the Nikodym property. We do this via a greedy
         removal: iteratively try to remove a point p from N (i.e., add it to S)
         and check if the Nikodym condition still holds. If yes, keep it removed.
      3. Also try alternative Y sets (not just squares) to get a better starting S.

    The Nikodym condition: for every point x in F_49^2, there exists a direction
    v = (1, m) or v = (0, 1) such that {x + t*v : t != 0} ⊆ N.
    Equivalently, for every x, for at least one direction v, ALL other points
    on the line through x in direction v are in N (not in S).

    We represent S as a boolean array of shape (49, 49) indexed by (x_idx, y_idx)
    where x_idx = xa + 7*xb, y_idx = ya + 7*yb.
    """
    p = 7
    q = p * p  # 49
    k_const = 3  # alpha^2 = 3

    # Precompute field operations
    # Element i = a + b*alpha where i = a + 7*b
    # We need addition, multiplication in F_49

    # Build multiplication table for F_49
    # (a + b*alpha) * (c + d*alpha) = (ac + bd*k) + (ad + bc)*alpha
    mult = np.zeros((q, q), dtype=np.int64)
    add = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        a = i % p
        b = i // p
        for j in range(q):
            c = j % p
            d = j // p
            # Addition
            add[i, j] = ((a + c) % p) + p * ((b + d) % p)
            # Multiplication
            ra = (a * c + b * d * k_const) % p
            rb = (a * d + b * c) % p
            mult[i, j] = ra + p * rb

    # Inverse table (for nonzero elements)
    inv = np.full(q, -1, dtype=np.int64)
    for i in range(1, q):
        for j in range(1, q):
            if mult[i, j] == 1:
                inv[i] = j
                break

    # Total number of points
    total_points = q * q  # 2401

    # Index mapping: point (x, y) where x, y are in F_49 (each 0..48)
    # x_idx = x, y_idx = y, flat index = x * q + y

    # Hermitian unital: U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace(x) = x + x^7 = 2 * Re(x) = 2 * (x % 7)  [since in F_{p^2}, x^p = conjugate]
    # Actually for F_{p^2} = F_p(alpha), the Frobenius x^p = a - b*alpha (conjugate)
    # So Trace(x) = x + x^p = (a + b*alpha) + (a - b*alpha) = 2a
    # Norm(y) = y * y^p = (a + b*alpha)(a - b*alpha) = a^2 - b^2 * k_const

    def trace(x_idx):
        return 2 * (x_idx % p) % p

    def norm(y_idx):
        a = y_idx % p
        b = y_idx // p
        return (a * a - b * b * k_const) % p

    # Build the unital U as a set of (x, y) pairs
    unital = np.zeros((q, q), dtype=bool)
    for x in range(q):
        tr_x = trace(x)
        for y in range(q):
            if (tr_x + norm(y)) % p == 0:
                unital[x, y] = True

    # Y_special for the baseline: y_a must be a square mod p
    squares_mod_p = set((u * u) % p for u in range(p))

    def y_in_special(y_idx, special_set):
        a = y_idx % p
        return a in special_set

    # Build baseline S0
    S0 = np.zeros((q, q), dtype=bool)
    for x in range(q):
        for y in range(q):
            if unital[x, y] and not y_in_special(y, squares_mod_p):
                S0[x, y] = True

    # N0 = complement of S0
    N0 = ~S0
    n_size = np.count_nonzero(N0)
    complement_size = total_points - n_size

    # Precompute all lines in directions (1, m) for m in F_49 and (0, 1)
    # A line in direction (1, m) through point (x0, y0):
    #   {(x0 + t, y0 + t*m) : t in F_49}
    # A line in direction (0, 1) through point (x0, y0):
    #   {(x0, y0 + t) : t in F_49}

    # For each point x, we need to check: exists direction v such that
    # for all t != 0, point x + t*v is in N (not in S).

    # Let's precompute for each direction the lines.
    # Directions: (0,1) and (1, m) for m = 0..48
    # Total directions: 1 + 49 = 50

    # For direction (0, 1): lines are vertical lines x = const
    # For direction (1, m): lines are graphs y = m*(x - x0) + y0

    # To check the Nikodym property efficiently:
    # For each point p, and each direction v, check if all other points on the line are in N.
    # This is expensive if done naively. Let's think of a faster approach.

    # Alternative: For a given S, a point x is "covered" if there exists a direction v
    # such that the line through x in direction v has no other points in S.
    # I.e., for direction v, the line L through x in direction v satisfies:
    #   L \ {x} ∩ S = ∅

    # Let's precompute for each direction v, and for each line in that direction,
    # the set of points on that line. Then for a given S, a point x on line L is
    # covered by direction v if S ∩ (L \ {x}) = ∅.

    # Number of lines per direction: q = 49 lines (since each line has q points,
    # and there are q^2 points total, so q lines per direction).

    # For direction (0,1): lines are {x = const}, 49 lines, each with 49 points.
    # For direction (1, m): lines are {y - m*x = c} for c in F_49. 49 lines each.

    # Let's precompute line membership.
    # For each direction index d (0 for (0,1), 1..49 for (1, m)),
    # line_id for point (x, y):
    #   d=0: line_id = x
    #   d=1+m: line_id = (y - m*x) in F_49

    # We'll store for each direction and line, the list of points on it.
    # But that's 50 * 49 * 49 = 120050 points total, which is fine.

    # Actually, let's just check the Nikodym property directly but efficiently.

    def is_nikodym(S):
        """Check if complement of S is a Nikodym set. S is (q, q) boolean array."""
        # For each point (x, y), check if there exists a direction v such that
        # all other points on the line through (x,y) in direction v are not in S.
        # Directions: (0,1) and (1, m) for m in 0..48

        # Precompute: for each direction, for each line, which points are in S.
        # If for some direction and line, the count of S points on the line is 0 or 1,
        # then every point on that line is covered by that direction (if count is 0,
        # all points are covered; if count is 1, the one S point is not covered by this
        # direction, but all others are).

        # More precisely: for a point x on line L in direction v, x is covered by v
        # iff |S ∩ L| == 0 or (|S ∩ L| == 1 and x not in S).
        # Wait, that's not right. x is covered by v iff S ∩ (L \ {x}) = ∅.
        # This is true iff |S ∩ L| == 0, or |S ∩ L| == 1 and x ∉ S.

        # So for a given direction v, a point x is NOT covered by v iff
        # |S ∩ L_v(x)| >= 1 and (|S ∩ L_v(x)| >= 2 or x ∈ S).
        # I.e., x is not covered by v iff x ∈ S and |S ∩ L_v(x)| >= 1 (trivially true if x in S)
        # OR x ∉ S and |S ∩ L_v(x)| >= 2.

        # A point x is covered (by some direction) iff there exists v such that
        # x is covered by v.

        # Let's just do the direct check but vectorize where possible.

        # For efficiency, let's check: for each direction, compute for each line the
        # set of points in S on that line. Then a point x is covered by direction v
        # iff the number of S points on L_v(x) is 0, or (is 1 and x not in S).

        # If for some direction v, EVERY point is covered by v, we're done.
        # Otherwise, we need the union over all directions.

        # Let's compute a boolean array "covered" of shape (q, q).
        covered = np.zeros((q, q), dtype=bool)

        # Direction (0, 1): lines are x = const
        for x in range(q):
            # Line: {(x, y) : y in 0..48}
            s_count = np.count_nonzero(S[x, :])
            if s_count == 0:
                covered[x, :] = True
            elif s_count == 1:
                # Only one point in S on this line; all other points are covered
                # Find which point is in S
                s_indices = np.where(S[x, :])[0]
                for y in range(q):
                    if y != s_indices[0]:
                        covered[x, y] = True

        # Directions (1, m) for m = 0..48
        for m in range(q):
            # Line through (x0, y0) in direction (1, m): 
            #   {(x0 + t, y0 + t*m) : t in F_49}
            # The line can be identified by the "intercept" c = y0 - m*x0 (in F_49)
            # For a point (x, y), it's on the line with intercept c = y - m*x

            # Group points by their intercept c
            # For each c, the line is {(x, c + m*x) : x in F_49}
            for c in range(q):
                # Points on this line: (x, (c + m*x) % q) for x in 0..48
                # But (c + m*x) is in F_49, so we need to compute it properly.
                # Let's just iterate over x.
                pts_in_s = []
                pts_all = []
                for x in range(q):
                    # y = c + m*x in F_49
                    y = add[c, mult[m, x]]  # c + m*x
                    pts_all.append((x, y))
                    if S[x, y]:
                        pts_in_s.append((x, y))

                s_count = len(pts_in_s)
                if s_count == 0:
                    for (x, y) in pts_all:
                        covered[x, y] = True
                elif s_count == 1:
                    sx, sy = pts_in_s[0]
                    for (x, y) in pts_all:
                        if x != sx or y != sy:
                            covered[x, y] = True

        return np.all(covered)

    # Verify baseline
    assert is_nikodym(S0), "Baseline should be valid"

    # Now try to improve: greedily add points to S (remove from N) while maintaining validity.
    # We want to maximize |S|, i.e., minimize |N|.

    # Greedy approach: try to add each point to S one at a time.
    # But checking is_nikodym after each addition is O(50 * 49 * 49) ≈ 120k ops,
    # and we have up to 2401 points to try. That's 2401 * 120k ≈ 288M ops, too slow.

    # Better: use a smarter approach. Let's try to find a larger S by modifying the Y set.

    # Alternative Y sets:
    # Y0: squares (3 values: 0, 1, 4) -> 3*7 = 21 y-values
    # Y1: all values -> 7*7 = 49 y-values (then S0 is empty, N = everything, complement = 0)
    # Y2: non-squares (2 values: 2, 3, 5, 6) -> 4*7 = 28 y-values
    # Y3: {0} -> 1*7 = 7 y-values
    # Y4: {1} -> 1*7 = 7 y-values
    # etc.

    # For Y with s distinct a-values, |Y| = s * 7.
    # S = {(x,y) in U : y not in Y}
    # |U| = q + 1 = 50 (Hermitian unital has q+1 points... wait, let me check)
    # Actually, the number of points on the Hermitian unital in F_q^2 is q + 1 = 50? 
    # No, for the equation Tr(x) + N(y) = 0, for each x there are exactly p solutions for y
    # (since N(y) = c has p solutions for each c). So |U| = q * p = 49 * 7 = 343.

    # Let me verify: for fixed x, Tr(x) = 2a. We need N(y) = -2a.
    # N(y) = a_y^2 - b_y^2 * k. This is a quadratic form. For each value c,
    # the number of solutions to N(y) = c is... it's a norm equation.
    # In F_{p^2}, the norm map is surjective, and each value has exactly p preimages.
    # So for each x, there are p = 7 values of y. Total |U| = 49 * 7 = 343.

    # So S = U \ (U ∩ {y in Y}).
    # |S| = |U| - |U ∩ {y in Y}|.
    # |U ∩ {y in Y}| = sum over y in Y of (number of x with Tr(x) + N(y) = 0)
    #                 = sum over y in Y of (number of x with Tr(x) = -N(y))
    # For each y, N(y) is some value c. Number of x with Tr(x) = c:
    # Tr(x) = 2a, so 2a = c mod 7. Since 2 is invertible mod 7, a = c/2 mod 7.
    # So there's exactly 1 value of a, and 7 values of b. So 7 x's for each y.
    # Thus |U ∩ {y in Y}| = 7 * |Y|.
    # |S| = 343 - 7 * |Y|.

    # For Y0 (squares): |Y0| = 3 * 7 = 21, |S0| = 343 - 147 = 196.
    # |N0| = 2401 - 196 = 2205. Complement = 196.

    # To maximize complement, we want to maximize |S|.
    # |S| = 343 - 7 * |Y|, so minimize |Y|.
    # Minimum |Y| = 7 (one a-value). Then |S| = 343 - 49 = 294.
    # But we need the Nikodym property to hold!

    # Let's try Y with |Y| = 7 (one a-value) and check if it's valid.

    best_S = S0.copy()
    best_complement = np.count_nonzero(S0)

    # Try all possible single-a Y sets
    for a_val in range(p):
        Y_test = set()
        for v in range(p):
            Y_test.add((a_val, v))

        S_test = np.zeros((q, q), dtype=bool)
        for x in range(q):
            for y in range(q):
                if unital[x, y]:
                    ya = y % p
                    yb = y // p
                    if (ya, yb) not in Y_test:
                        S_test[x, y] = True

        comp = np.count_nonzero(S_test)
        if comp > best_complement:
            if is_nikodym(S_test):
                best_S = S_test.copy()
                best_complement = comp

    # Try two-a-value Y sets
    for a1 in range(p):
        for a2 in range(a1 + 1, p):
            Y_test = set()
            for v in range(p):
                Y_test.add((a1, v))
                Y_test.add((a2, v))

            S_test = np.zeros((q, q), dtype=bool)
            for x in range(q):
                for y in range(q):
                    if unital[x, y]:
                        ya = y % p
                        yb = y // p
                        if (ya, yb) not in Y_test:
                            S_test[x, y] = True

            comp = np.count_nonzero(S_test)
            if comp > best_complement:
                if is_nikodym(S_test):
                    best_S = S_test.copy()
                    best_complement = comp

    # Now try greedy removal from N (adding to S) to see if we can do even better.
    # Start from the best S found above, and try to add points one by one.
    # But only points that are in U (since S ⊆ U in our construction).
    # Actually, we could potentially add points NOT in U to S too, but that would
    # break the structure. Let's stick with S ⊆ U for now.

    # Get list of points in U not currently in S
    u_points = []
    for x in range(q):
        for y in range(q):
            if unital[x, y] and not best_S[x, y]:
                u_points.append((x, y))

    # Try to add each point to S if it maintains the Nikodym property
    # This is O(|u_points| * cost_of_is_nikodym)
    # |u_points| can be up to 343, and cost is ~120k, so ~40M ops. Might be slow but let's try.

    # Actually, let's be smarter. Only try points that are "safe" to remove.
    # A point p in N can be added to S if for every line through p, there's still
    # some direction that covers all other points. This is complex.

    # Let's just do a limited greedy search with a time budget.
    start_time = time.time()
    deadline = 580  # seconds, leaving 50s margin

    # Try adding points in a random order
    import random
    random.seed(42)
    random.shuffle(u_points)

    for (x, y) in u_points:
        if time.time() > deadline:
            break
        # Try adding this point to S
        old_val = best_S[x, y]
        best_S[x, y] = True
        if is_nikodym(best_S):
            best_complement += 1
        else:
            best_S[x, y] = old_val

    # Also try removing points from S (adding back to N) to see if we can rearrange
    # Actually, we want to MAXIMIZE S, so we only add to S.

    # Build the output: N = complement of S
    points = []
    for x in range(q):
        for y in range(q):
            if not best_S[x, y]:
                xa = x % p
                xb = x // p
                ya = y % p
                yb = y // p
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
