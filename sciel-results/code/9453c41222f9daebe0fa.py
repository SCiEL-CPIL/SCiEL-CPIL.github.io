# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = 49 = 7^2. A set N is a Nikodym set
    if, for every point x in F_49^2, there exists a direction v such that the
    punctured line {x + t*v : t in F_49, t != 0} is fully contained in N.
    We want to maximize |F_49^2 \\ N| (the complement S).

    Strategy: Start from the algebraic baseline (Hermitian unital blocking set)
    which gives a valid Nikodym set, then greedily grow S by adding points that
    still allow every point in S to have at least one "free" direction (a direction
    where all other points on that line are in S).

    Field: F_49 = F_7[alpha], alpha^2 = 3. Element (a,b) = a + b*alpha.
    Directions: v = (1, m) for m in F_49 (49 directions) plus v = (0, 1) (1 direction) = 50 total.
    """
    p = 7
    q = p * p  # 49
    K = 3  # alpha^2 = 3 (smallest quadratic non-residue mod 7)

    # Precompute field operations
    # add(a, b) where a, b are pairs (a0, a1), (b0, b1)
    # We'll represent field elements as integers 0..48: elem = a0 + a1 * p
    def elem_to_pair(e):
        return (e % p, e // p)

    def pair_to_elem(a0, a1):
        return a0 + a1 * p

    def field_add(e1, e2):
        a1, b1 = elem_to_pair(e1)
        a2, b2 = elem_to_pair(e2)
        return pair_to_elem((a1 + a2) % p, (b1 + b2) % p)

    def field_neg(e):
        a, b = elem_to_pair(e)
        return pair_to_elem((-a) % p, (-b) % p)

    def field_mul(e1, e2):
        a1, b1 = elem_to_pair(e1)
        a2, b2 = elem_to_pair(e2)
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + (a1*b2 + a2*b1)*alpha + b1*b2*alpha^2
        # alpha^2 = K
        res_a = (a1 * a2 + b1 * b2 * K) % p
        res_b = (a1 * b2 + a2 * b1) % p
        return pair_to_elem(res_a, res_b)

    def field_inv(e):
        if e == 0:
            return 0
        a, b = elem_to_pair(e)
        # Norm = a^2 - K*b^2
        norm = (a * a - K * b * b) % p
        if norm == 0:
            return 0  # shouldn't happen for nonzero in F_49
        inv_norm = pow(norm, p - 2, p)
        # inverse = (a - b*alpha) / norm
        res_a = (a * inv_norm) % p
        res_b = ((-b) * inv_norm) % p
        return pair_to_elem(res_a, res_b)

    def field_sub(e1, e2):
        return field_add(e1, field_neg(e2))

    # Build lookup tables for speed
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            add_table[i, j] = field_add(i, j)
            mul_table[i, j] = field_mul(i, j)

    neg_table = np.zeros(q, dtype=np.int64)
    for i in range(q):
        neg_table[i] = field_neg(i)

    # Directions: index 0..48 -> v = (1, m) where m = 0..48; index 49 -> v = (0, 1)
    # For a point P = (px, py) and direction v, the line is {P + t*v : t in F_49}
    # For v = (1, m): points are (px + t, py + t*m) for t in F_49
    # For v = (0, 1): points are (px, py + t) for t in F_49

    # Represent a point as a single integer: pt = x + y * q where x, y in 0..48
    # x is the field element for the first coordinate, y for the second.

    # Precompute for each direction d and each point p, the set of other points on the line.
    # line_points[d][p] = list of q-1 other points on the line through p in direction d.
    # Actually, for efficiency, let's precompute: for each direction d, partition all q^2 points into q lines of q points each.

    num_dirs = q + 1  # 50
    # line_membership[d][line_id][t] = point index
    # Or better: for each direction d, create an array of shape (q, q) where
    # lines[d][i][t] is the point at position t on line i in direction d.

    # For direction d = m (0..48), v = (1, m):
    # A line can be identified by its y-intercept: when x=0, y = c.
    # Point on line: (t, c + t*m) for t in F_49.
    # So line c in direction m contains points (t, c + t*m) for t=0..48.

    # For direction d = 49, v = (0, 1):
    # A line is identified by x = c.
    # Point on line: (c, t) for t in 0..48.

    # Let's precompute: for each direction d, for each point p, which line it's on and its position.
    # line_id[d][p] and line_pos[d][p]

    line_id = np.zeros((num_dirs, q * q), dtype=np.int64)
    line_pos = np.zeros((num_dirs, q * q), dtype=np.int64)

    for m in range(q):
        # direction (1, m)
        for c in range(q):
            for t in range(q):
                x = t
                y = field_add(c, field_mul(t, m))
                pt = x + y * q
                line_id[m, pt] = c
                line_pos[m, pt] = t

    # direction (0, 1)
    d = q  # index 49
    for c in range(q):
        for t in range(q):
            x = c
            y = t
            pt = x + y * q
            line_id[d, pt] = c
            line_pos[d, pt] = t

    # For checking if a point p can be added to S with direction d:
    # All other q-1 points on the line must already be in S.
    # We can precompute for each (d, p) the list of other points on the same line.
    # other_points[d][p] = array of q-1 point indices

    # Actually, let's think differently. For a point p to be in S, it needs at least one
    # direction d such that all other points on line(d, p) are in S.
    # When we add a point to S, we might "break" the validity of other points that were
    # relying on a direction where this new point was the only one not in S... no wait.
    # If a point p' has direction d' where all other points on line(d', p') are in S,
    # and we add a new point p to S, this doesn't break p' unless p is on line(d', p')
    # and p was previously NOT in S. But if p was not in S, then p' couldn't have used d'
    # because p was on the line and not in S. So adding points to S only HELPS other points.
    # 
    # Wait, that's the key insight! Adding a point to S can only make it EASIER for other
    # points to satisfy the Nikodym condition. So greedy addition is safe: if every point
    # in S currently has a valid direction, and we add a new point p that has a valid
    # direction (all other points on some line through p are in S), then all existing points
    # in S still have their valid directions (they might even gain more options).

    # So the greedy algorithm is:
    # 1. Start with S = empty.
    # 2. Find all points that can be added (have a direction where all other q-1 points on that line are in S).
    #    Initially, no point can be added since S is empty and q-1 = 48 > 0.
    #    
    # Hmm, that's a problem. We need a seed. Let's start from the algebraic construction.

    # Algebraic baseline:
    # Unital U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a mod p
    # Norm((a,b)) = a^2 - K*b^2 mod p
    # Blocking set B = points in U where y is in Y = {(u^2 mod p, v) : u, v in F_p}
    # S_baseline = U \ B

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add((u * u % p, v))

    S = set()
    for xa in range(p):
        for xb in range(p):
            x = pair_to_elem(xa, xb)
            for ya in range(p):
                for yb in range(p):
                    y = pair_to_elem(ya, yb)
                    # Check if in unital: 2*xa + ya^2 - K*yb^2 == 0 mod p
                    if (2 * xa + ya * ya - K * yb * yb) % p == 0:
                        # In unital
                        if (ya, yb) not in y_special:
                            # In S (excluded from blocking set)
                            pt = x + y * q
                            S.add(pt)

    # Now greedily add points to S
    # For each point not in S, check if it can be added.
    # A point p can be added if there exists direction d such that all other points
    # on line(d, p) are in S.

    # Precompute: for each direction d and point p, the other q-1 points on the line.
    # For efficiency, precompute line_members[d][line_id] = list of q points on that line.

    line_members = []
    for d in range(num_dirs):
        members = {}
        for pt in range(q * q):
            lid = line_id[d, pt]
            if lid not in members:
                members[lid] = []
            members[lid].append(pt)
        # Convert to array
        line_members.append(members)

    # For each direction d and point p, count how many of the other q-1 points are in S.
    # If count == q-1, p can be added with direction d.

    # Maintain: for each (d, p), the number of other points on line(d,p) that are in S.
    # When we add a point p_new to S, for each direction d, p_new is on some line L.
    # For every other point p' on L, the count for (d, p') increases by 1.

    # counts[d][p] = number of other points on line(d, p) that are in S
    counts = np.zeros((num_dirs, q * q), dtype=np.int64)

    # Initialize counts from current S
    for d in range(num_dirs):
        for lid, pts in line_members[d].items():
            in_s = sum(1 for pt in pts if pt in S)
            for pt in pts:
                counts[d, pt] = in_s - (1 if pt in S else 0)

    # Greedy: repeatedly add any point that has at least one direction with counts[d][p] == q-1
    # To be efficient, maintain a list of "ready" points.

    def is_ready(p):
        return any(counts[d, p] == q - 1 for d in range(num_dirs))

    # Start with all points not in S as candidates
    # We'll use a queue-based approach

    changed = True
    while changed:
        changed = False
        for p in range(q * q):
            if p in S:
                continue
            if is_ready(p):
                # Add p to S
                S.add(p)
                changed = True
                # Update counts: for each direction d, p is on line lid = line_id[d, p]
                # For each other point p' on that line, counts[d, p'] += 1
                for d in range(num_dirs):
                    lid = line_id[d, p]
                    for p2 in line_members[d][lid]:
                        if p2 != p:
                            counts[d, p2] += 1

    # Build the result: N = F_49^2 \ S
    points = []
    for x in range(q):
        for y in range(q):
            pt = x + y * q
            if pt not in S:
                xa, xb = elem_to_pair(x)
                ya, yb = elem_to_pair(y)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
