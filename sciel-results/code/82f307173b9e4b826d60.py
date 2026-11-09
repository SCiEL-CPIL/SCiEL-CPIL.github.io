# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using a greedy algorithm.
    The goal is to minimize |N|, equivalently maximize the complement (excluded set) E.
    A point x is "covered" if there exists a line direction v = (1, m) or (0, 1)
    such that all other points on that line through x belong to N.
    We greedily add points to E if doing so doesn't "uncover" any point.
    """
    deadline = time.time() + 580  # Safety margin before 600s limit
    p = 7
    # alpha^2 = 3 (3 is a non-residue mod 7: 3^3 = 27 = 6 = -1)
    k = 3

    # Precompute field operations for F_49 = F_7[alpha], alpha^2 = k
    # Represent elements as pairs (a, b) for a + b*alpha
    # Precompute multiplication table
    mul_table = np.zeros((p, p, p, p, p, p), dtype=np.int64)
    for a1 in range(p):
        for b1 in range(p):
            for a2 in range(p):
                for b2 in range(p):
                    # (a1 + b1*alpha) * (a2 + b2*alpha)
                    # = a1*a2 + (a1*b2 + a2*b1)*alpha + b1*b2*alpha^2
                    # = (a1*a2 + b1*b2*k) + (a1*b2 + a2*b1)*alpha
                    res_a = (a1 * a2 + b1 * b2 * k) % p
                    res_b = (a1 * b2 + a2 * b1) % p
                    mul_table[a1, b1, a2, b2] = [res_a, res_b]

    def f_add(a, b, c, d):
        return ((a + c) % p, (b + d) % p)

    def f_neg(a, b):
        return ((p - a) % p, (p - b) % p)

    def f_mul(a, b, c, d):
        ra, rb = mul_table[a, b, c, d]
        return (int(ra), int(rb))

    def f_sub(a, b, c, d):
        return ((a - c) % p, (b - d) % p)

    # Total number of points
    n_pts = p * p * p * p  # 2401

    # Encode point (xa, xb, ya, yb) to index
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def decode(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb

    # Precompute all lines
    # For each point and each direction, compute the set of other points on the line
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # m is represented as (ma, mb)
    
    # Total directions: 49 + 1 = 50
    # For each point and direction, we need to know which other points are on the line
    
    # Strategy: 
    # 1. Start with N = all points (E = empty)
    # 2. A point x is "covered" if there exists a direction v such that all 
    #    other points on the line through x in direction v are in N.
    # 3. We can add x to E (remove from N) if x is still covered by at least one 
    #    direction after removal, OR if x was only covered by directions that still 
    #    work after removing x.
    
    # Actually, let me think more carefully:
    # A point x is covered if there's a line L through x such that L \ {x} ⊆ N.
    # If we remove x from N (add to E), we need to check:
    # - For each line through x: if L \ {x} ⊆ N still holds, x is still covered.
    # - If no line through x has L \ {x} ⊆ N after removal, x becomes uncovered.
    
    # Key insight: When we remove point y from N, it affects the coverage of points
    # that are on lines through y. Specifically, for a point x on a line L through y,
    # if L \ {x} was contained in N and y ∈ L \ {x}, then after removing y, 
    # L \ {x} may no longer be contained in N.
    
    # Let's precompute for each point x, the set of lines through x
    # and for each line, the set of points on it.
    
    # Precompute: for each point, list of (direction_index, other_points_on_line)
    # But this is too much memory. Let's be smarter.
    
    # Alternative approach: 
    # For each point x and each direction v, define "bad(x, v)" = number of points 
    # on the line through x in direction v (excluding x) that are NOT in N.
    # x is covered by direction v iff bad(x, v) == 0.
    # x is covered iff there exists v with bad(x, v) == 0.
    
    # When we remove point y from N:
    # - For each point x and direction v such that y is on the line through x in direction v:
    #   bad(x, v) increases by 1.
    # - This may cause some (x, v) pairs to go from bad == 0 to bad == 1.
    # - If x was only covered by direction v (i.e., v was the only direction with bad == 0),
    #   then x becomes uncovered.
    
    # Let's precompute:
    # 1. For each point y, which (x, v) pairs are affected when y is removed.
    #    y is on the line through x in direction v iff x + t*v = y for some t != 0.
    
    # This is getting complex. Let me use a simpler but potentially slower approach.
    
    # Simpler approach:
    # 1. Start with N = all points.
    # 2. For each point x, compute the set of directions v for which bad(x, v) == 0.
    # 3. Greedily: try to remove points from N. For each candidate point y to remove,
    #    check if removing y causes any point to become uncovered.
    #    If not, remove y.
    
    # But checking "if removing y causes any point to become uncovered" is expensive.
    
    # Let me try a different greedy strategy:
    # 1. Start with E = empty, N = all points.
    # 2. For each point y, compute its "removal cost": the number of points that would
    #    become uncovered if we remove y from N.
    # 3. If any point has cost 0, remove it (add to E).
    # 4. Repeat until no point has cost 0.
    
    # To compute removal cost efficiently:
    # When we remove y, for each x and direction v where y is on the line through x in direction v:
    #   bad(x, v) increases by 1.
    # If bad(x, v) was 0 and becomes 1, and v was the only direction with bad == 0 for x,
    # then x becomes uncovered.
    
    # Let me precompute the incidence: for each point y, which (x, v) pairs have y on 
    # the line through x in direction v.
    
    # Directions: 50 total (49 slopes + vertical)
    # For a given direction v = (vx, vy), the line through x in direction v contains 
    # all points x + t*v for t in F_49.
    # y is on this line iff y - x is a scalar multiple of v, i.e., y - x = t*v for some t.
    # For v = (1, m): y_x - x_x = t * 1, y_y - x_y = t * m.
    #   So t = y_x - x_x, and we need y_y - x_y = (y_x - x_x) * m.
    # For v = (0, 1): y_x = x_x, and t = y_y - x_y.
    
    # Let me precompute for each direction, a mapping from point to the set of points 
    # on the same line. Actually, let me think of it as: for each direction v, the lines
    # partition the plane. For each line, all points on it are mutually "on the same line".
    
    # For direction v = (1, m): two points (x1, y1) and (x2, y2) are on the same line iff
    # (y2 - y1) = m * (x2 - x1).
    # For v = (0, 1): same line iff x1 = x2.
    
    # Let me precompute for each direction, the line assignment for each point.
    
    # Actually, let me just do a direct but hopefully fast enough approach.
    
    # Initialize
    in_N = np.ones(n_pts, dtype=bool)
    in_E = np.zeros(n_pts, dtype=bool)
    
    # bad[x, v] = number of points on line through x in direction v (excl x) not in N
    # Directions: 0 to 48 are (1, m) for m = (ma, mb), 49 is (0, 1)
    # m encoded as ma * p + mb
    
    n_dirs = 49 + 1  # 50
    
    # Precompute direction vectors
    dir_vectors = []
    for ma in range(p):
        for mb in range(p):
            dir_vectors.append((1, ma, mb))  # (vx_a, vy_a, vy_b) where v = (1, ma + mb*alpha)
    dir_vectors.append((0, 0, 1))  # vertical direction (0, 1)
    
    # For each point x and direction v, precompute the set of other points on the line
    # line_points[x, v] = list of other point indices on the line through x in direction v
    
    # This is 2401 * 50 * 48 ≈ 5.7 million entries, might be too much
    # Let me compute on the fly or use a more memory-efficient structure.
    
    # Alternative: for each direction, group points into lines.
    # For direction (1, m): line is determined by (x, y - m*x) = (c1, c2)
    # For direction (0, 1): line is determined by x
    
    # Let me precompute line IDs for each direction and point.
    # line_id[x, v] = unique ID of the line through x in direction v
    
    # For direction (1, ma, mb): 
    #   The line through (xa, xb, ya, yb) in direction (1, ma+mb*alpha) contains points
    #   (xa + t, ya + t*(ma+mb*alpha)) for t in F_49.
    #   Two points are on the same line iff they have the same "intercept".
    #   The intercept can be computed as: (x, y - m*x)
    
    def compute_line_key(xa, xb, ya, yb, ma, mb, vertical):
        if vertical:
            # Line determined by x = (xa, xb)
            return (xa, xb)
        else:
            # y - m*x should be constant
            # m = ma + mb*alpha
            # y - m*x = (ya, yb) - (ma + mb*alpha)*(xa + xb*alpha)
            # = (ya, yb) - (ma*xa + mb*xb*k, ma*xb + mb*xa)
            res_a = (ya - ma * xa - mb * xb * k) % p
            res_b = (yb - ma * xb - mb * xa) % p
            return (res_a, res_b)
    
    # Precompute for each direction, a dict from line_key to list of points
    # But this is 50 * (up to 2401 lines) which is fine.
    
    # Actually, let me just directly compute the bad counts and use a greedy approach.
    
    # Initialize bad[x, v] for all x and v
    # bad[x, v] = number of points on line through x in direction v (excl x) not in N
    # Initially N = all points, so bad[x, v] = 0 for all x, v.
    
    bad = np.zeros((n_pts, n_dirs), dtype=np.int32)
    
    # covered[x] = number of directions v with bad[x, v] == 0
    covered = np.full(n_pts, n_dirs, dtype=np.int32)
    
    # For each direction, precompute which points are on each line
    # dir_lines[v] = list of sets of points on each line
    
    # Let me precompute for each direction v, a mapping from point to line_members
    # But storing all this is memory intensive. Let me compute on the fly.
    
    # For the greedy algorithm, when we remove point y from N:
    # For each direction v:
    #   Find all points x on the line through y in direction v (x != y).
    #   For each such x: bad[x, v] += 1
    #   If bad[x, v] was 0 and becomes 1: covered[x] -= 1
    #   If covered[x] becomes 0: x is now uncovered -> this is a "violation"
    
    # To find points x on the line through y in direction v:
    # For v = (1, ma, mb): x = y + t*(1, ma, mb) for t in F_49, t != 0
    #   x_x = y_x + t, x_y = y_y + t*(ma, mb)
    # For v = (0, 0, 1): x = y + t*(0, 1) for t in F_49, t != 0
    #   x_x = y_x, x_y = y_y + t*(0, 1)
    
    # Let me precompute for each point y and direction v, the list of other points on the line.
    # This is 2401 * 50 * 48 ≈ 5.7M entries. Each entry is an int32. ~23 MB. Acceptable.
    
    print(f"Precomputing line memberships...")
    pre_t0 = time.time()
    
    # line_others[y, v] = array of point indices (excluding y) on the line through y in direction v
    # Use a list of arrays
    line_others = [None] * (n_pts * n_dirs)
    
    for ya in range(p):
        for yb in range(p):
            for xa in range(p):
                for xb in range(p):
                    y = encode(xa, xb, ya, yb)
                    for v in range(n_dirs):
                        others = []
                        if v < 49:
                            # v = (1, ma + mb*alpha)
                            ma = v // p
                            mb = v % p
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    # x = y + t * (1, ma + mb*alpha)
                                    # t = ta + tb*alpha
                                    # t * 1 = t = (ta, tb)
                                    # t * (ma + mb*alpha) = (ta*ma + tb*mb*k, ta*mb + tb*ma)
                                    new_xa = (xa + ta) % p
                                    new_xb = (xb + tb) % p
                                    ty_a = (ta * ma + tb * mb * k) % p
                                    ty_b = (ta * mb + tb * ma) % p
                                    new_ya = (ya + ty_a) % p
                                    new_yb = (yb + ty_b) % p
                                    other = encode(new_xa, new_xb, new_ya, new_yb)
                                    others.append(other)
                        else:
                            # v = (0, 1)
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    # x = y + t * (0, 1)
                                    # t * (0, 1) = (0*ta + 1*tb*k, 0*tb + 1*ta) = (tb*k, ta)
                                    new_xa = xa
                                    new_xb = xb
                                    new_ya = (ya + tb * k) % p
                                    new_yb = (yb + ta) % p
                                    other = encode(new_xa, new_xb, new_ya, new_yb)
                                    others.append(other)
                        line_others[y * n_dirs + v] = np.array(others, dtype=np.int32)
    
    print(f"Precomputation done in {time.time() - pre_t0:.1f}s, using {line_others[0].shape[0]} points per line")
    
    # Now greedy removal
    print(f"Starting greedy removal...")
    greedy_t0 = time.time()
    
    # Track which points are in N
    in_N = np.ones(n_pts, dtype=bool)
    removed_count = 0
    
    # We'll iterate: find a point with zero "removal cost" and remove it.
    # Removal cost of y = number of points x that become uncovered if we remove y.
    # A point x becomes uncovered if after removing y, covered[x] == 0.
    
    # To compute removal cost efficiently, we need to know for each y, which (x, v) pairs
    # would have bad[x, v] go from 0 to 1, and whether that makes covered[x] == 0.
    
    # Let me just do a simple greedy: try each point, compute cost, pick min.
    # But this is O(n_pts^2 * n_dirs * line_size) which might be too slow.
    
    # Alternative: maintain a priority queue or just iterate.
    
    # Let me try: repeatedly find a point with cost 0 and remove it.
    # To compute cost of removing y:
    #   For each v: for each x in line_others[y, v]:
    #     if bad[x, v] == 0 and covered[x] == 1: cost += 1
    #   (because if covered[x] > 1, x remains covered by other directions)
    
    # Actually, covered[x] counts directions with bad == 0. If bad[x, v] goes from 0 to 1,
    # covered[x] decreases by 1. x becomes uncovered iff covered[x] was 1 and becomes 0.
    
    # Let me implement this.
    
    while time.time() < deadline - 30:
        # Find a point to remove with cost 0
        found = False
        for y in range(n_pts):
            if not in_N[y]:
                continue
            # Compute cost of removing y
            cost = 0
            for v in range(n_dirs):
                others = line_others[y * n_dirs + v]
                for x in others:
                    if not in_N[x]:
                        continue
                    if bad[x, v] == 0 and covered[x] == 1:
                        cost += 1
                        if cost > 0:
                            break
                if cost > 0:
                    break
            if cost == 0:
                # Remove y from N
                in_N[y] = False
                removed_count += 1
                for v in range(n_dirs):
                    others = line_others[y * n_dirs + v]
                    for x in others:
                        if not in_N[x]:
                            continue
                        bad[x, v] += 1
                        if bad[x, v] == 1:  # was 0, now 1
                            covered[x] -= 1
                found = True
                break
        
        if not found:
            break
    
    print(f"Greedy removal done in {time.time() - greedy_t0:.1f}s, removed {removed_count} points")
    
    # Build result: points in N
    result_indices = np.where(in_N)[0]
    n_result = len(result_indices)
    
    result = np.zeros((n_result, 2, 2), dtype=np.int64)
    for i, idx in enumerate(result_indices):
        xa, xb, ya, yb = decode(idx)
        result[i, 0, 0] = xa
        result[i, 0, 1] = xb
        result[i, 1, 0] = ya
        result[i, 1, 1] = yb
    
    print(f"Result: |N| = {n_result}, complement = {n_pts - n_result}")
    
    return result


# EVOLVE-BLOCK-END
