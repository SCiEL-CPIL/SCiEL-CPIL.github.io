# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 610.0
START_TIME = time.time()


def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where F_49 = F_7[alpha], alpha^2 = 3.
    
    Strategy: Start with a known valid Nikodym set (the unital construction),
    then greedily add points to the complement S while maintaining validity.
    Uses an efficient validity check based on tracking which directions are
    still available for each point.
    """
    p = 7
    k = 3  # alpha^2 = 3
    q = p * p  # 49
    n_total = q * q  # 2401

    # Precompute field multiplication table: mul[i][j] = field element i * j
    # Elements are indexed 0..48, where index = a*7 + b represents a + b*alpha
    mul = [[0] * q for _ in range(q)]
    for i in range(q):
        ia, ib = i // 7, i % 7
        for j in range(q):
            ja, jb = j // 7, j % 7
            ia3 = (ia * ja + k * ib * jb) % p
            ib3 = (ia * jb + ja * ib) % p
            mul[i][j] = ia3 * 7 + ib3

    # Precompute field addition table
    add = [[0] * q for _ in range(q)]
    for i in range(q):
        ia, ib = i // 7, i % 7
        for j in range(q):
            ja, jb = j // 7, j % 7
            add[i][j] = ((ia + ja) % p) * 7 + ((ib + jb) % p)

    # Precompute field subtraction table
    sub = [[0] * q for _ in range(q)]
    for i in range(q):
        ia, ib = i // 7, i % 7
        for j in range(q):
            ja, jb = j // 7, j % 7
            sub[i][j] = ((ia - ja) % p) * 7 + ((ib - jb) % p)

    # Point indexing: idx = x * q + y, where x, y in 0..48
    # x = xa*7 + xb, y = ya*7 + yb

    # Directions: 50 directions total
    # Directions 0..48: (1, m) for m = 0..48
    # Direction 49: (0, 1)
    n_dirs = 50

    # Precompute line membership: for each direction d and point idx,
    # line_members[d][idx] = set of all points on the line through idx with direction d
    # Using lists for efficiency
    
    # For direction (1, m): line is y = m*x + c, so c = y - m*x
    # For direction (0, 1): line is x = c

    # Build line maps for each direction
    line_maps = []  # line_maps[d] = dict mapping c -> list of point indices
    for d in range(n_dirs):
        if d < q:
            m = d  # direction (1, m)
            line_map = {}
            for x in range(q):
                for y in range(q):
                    idx = x * q + y
                    c = sub[y, mul[m, x]]
                    if c not in line_map:
                        line_map[c] = []
                    line_map[c].append(idx)
        else:
            # direction (0, 1): x = c
            line_map = {}
            for x in range(q):
                for y in range(q):
                    idx = x * q + y
                    c = x
                    if c not in line_map:
                        line_map[c] = []
                    line_map[c].append(idx)
        line_maps.append(line_map)

    # For each direction d and point idx, precompute the line through idx
    # line_pts[d][idx] = list of all points on the line (including idx itself)
    line_pts = []
    for d in range(n_dirs):
        lm = line_maps[d]
        pts = [None] * n_total
        for c, members in lm.items():
            for idx in members:
                pts[idx] = members
        line_pts.append(pts)

    # Base construction: use a known valid Nikodym set
    # The set N = {all points} \ S where S is our complement
    # Start with S = empty, N = all points (trivially valid)
    # But we want a smaller N, so start with a non-trivial S
    
    # Use the unital-based construction as starting point
    # Squares in F_7: {0, 1, 2, 4}
    squares = set()
    for u in range(p):
        squares.add((u * u) % p)

    # S_bits: bit i is 1 if point i is in S (complement of N)
    # Start with points on the unital where y_a is a square
    S_bits = 0
    for idx in range(n_total):
        x = idx // q
        y = idx % q
        xa, xb = x // 7, x % 7
        ya, yb = y // 7, y % 7
        # Unital equation: 2*x_a + y_a^2 - 3*y_b^2 = 0 mod 7
        # i.e., 2*xa + ya*ya - k*yb*yb = 0 mod 7
        in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            S_bits |= (1 << idx)

    # Verify initial construction is valid
    # For each point j, check there exists a direction d such that all other points on line_d(j) are in N (not in S)
    
    # Efficient tracking: for each point j, track which directions are "available"
    # A direction d is available for j if no point in S lies on the line through j with direction d (other than j itself)
    
    # Precompute for each point c in S, which points j are disrupted
    # c_disrupts[c] = set of (j, d) pairs where c lies on line_d(j) and c != j
    
    # Better approach: for each point j and direction d, track count of S points on line_d(j) excluding j
    # s_count[d][j] = number of points in S on line_d(j) excluding j itself
    # A direction d is valid for j if s_count[d][j] == 0
    
    s_count = [[0] * n_total for _ in range(n_dirs)]
    
    # Initialize s_count
    for idx in range(n_total):
        if S_bits & (1 << idx):
            for d in range(n_dirs):
                pts = line_pts[d][idx]
                for other in pts:
                    if other != idx:
                        s_count[d][other] += 1

    # valid_dirs[j] = bitmask of directions d where s_count[d][j] == 0
    mask_50 = (1 << n_dirs) - 1
    valid_dirs = [0] * n_total
    for j in range(n_total):
        mask = 0
        for d in range(n_dirs):
            if s_count[d][j] == 0:
                mask |= (1 << d)
        valid_dirs[j] = mask

    # Check validity: every point must have at least one valid direction
    valid = True
    for j in range(n_total):
        if valid_dirs[j] == 0:
            valid = False
            break

    if not valid:
        # Fallback: use all points as N (S = empty)
        points = []
        for idx in range(n_total):
            x = idx // q
            y = idx % q
            xa, xb = x // 7, x % 7
            ya, yb = y // 7, y % 7
            points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    # Now greedily add points to S
    # Candidates: points not yet in S
    candidates = [i for i in range(n_total) if not (S_bits & (1 << i))]
    
    # Sort candidates by some heuristic - points that disrupt fewer valid directions first
    # For a candidate c, the disruption is: for each j where c is on line_d(j),
    # if d was the only valid direction for j, then adding c makes j invalid
    
    # Precompute disruption scores
    disruption_score = [0] * n_total
    for c in candidates:
        score = 0
        for d in range(n_dirs):
            pts = line_pts[d][c]
            for j in pts:
                if j == c:
                    continue
                # Check if d is the only valid direction for j
                if valid_dirs[j] == (1 << d):
                    score += 1
        disruption_score[c] = score

    # Sort by disruption score (lowest first)
    candidates.sort(key=lambda c: disruption_score[c])

    # Greedily add points
    i = 0
    while i < len(candidates):
        if _time_left() < 100:
            break
        
        c = candidates[i]
        
        # Check if c is still valid (not in S)
        if S_bits & (1 << c):
            i += 1
            continue
        
        # Check if adding c keeps validity
        # For each j and direction d where c is on line_d(j) and j != c:
        # if s_count[d][j] == 0 and valid_dirs[j] == (1 << d), then adding c invalidates j
        valid_add = True
        affected = []  # list of (j, d) pairs that would be affected
        
        for d in range(n_dirs):
            pts = line_pts[d][c]
            for j in pts:
                if j == c:
                    continue
                if s_count[d][j] == 0:
                    # Adding c would make s_count[d][j] = 1
                    # Check if d is the only valid direction for j
                    if valid_dirs[j] == (1 << d):
                        valid_add = False
                        break
                    else:
                        affected.append((j, d))
            if not valid_add:
                break
        
        if valid_add:
            # Add c to S
            S_bits |= (1 << c)
            
            # Update s_count and valid_dirs
            for d in range(n_dirs):
                pts = line_pts[d][c]
                for j in pts:
                    if j == c:
                        continue
                    old_count = s_count[d][j]
                    s_count[d][j] = old_count + 1
                    if old_count == 0:
                        # Direction d is no longer valid for j
                        valid_dirs[j] &= ~(1 << d)
        
        i += 1

    # Build final result: N = all points not in S
    points = []
    for idx in range(n_total):
        if not (S_bits & (1 << idx)):
            x = idx // q
            y = idx % q
            xa, xb = x // 7, x % 7
            ya, yb = y // 7, y % 7
            points.append([[xa, xb], [ya, yb]])

    if len(points) == 0:
        # Should not happen, but fallback
        points = [[[0, 0], [0, 0]]]

    arr = np.array(points, dtype=np.int64)
    return arr

# EVOLVE-BLOCK-END
