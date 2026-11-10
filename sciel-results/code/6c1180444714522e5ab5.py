import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    deadline = time.time() + 580  # safety margin before 630s limit

    # Field F_49 = F_7[alpha], alpha^2 = 3
    k = 3  # 3 is a quadratic non-residue mod 7

    # Precompute field operations using lookup tables for speed
    # Element represented as integer 0..48: a + b*alpha = a + b*7
    def add(x, y):
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        return (xa + ya) % p + ((xb + yb) % p) * p

    def sub(x, y):
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        return (xa - ya) % p + ((xb - yb) % p) * p

    # Precompute multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for x in range(q):
        xa, xb = x % p, x // p
        for y in range(q):
            ya, yb = y % p, y // p
            ra = (xa * ya + xb * yb * k) % p
            rb = (xa * yb + xb * ya) % p
            mul_table[x, y] = ra + rb * p

    def mul(x, y):
        return int(mul_table[x, y])

    # Precompute inverse table
    inv_table = np.full(q, -1, dtype=np.int32)
    for x in range(1, q):
        for y in range(1, q):
            if int(mul_table[x, y]) == 1:
                inv_table[x] = y
                break

    def inv(x):
        if x == 0:
            return 0
        return int(inv_table[x])

    def div(x, y):
        if y == 0:
            return 0
        return mul(x, inv(y))

    # Base construction: Hermitian unital based Nikodym set
    # U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - b^2*k
    # Special y set: y_a is a square mod 7
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    # Build base excluded set S (points on U where y NOT in Y)
    base_excluded = set()
    for xa in range(p):
        for xb in range(p):
            x = xa + xb * p
            trace_x = (2 * xa) % p
            for ya in range(p):
                for yb in range(p):
                    y = ya + yb * p
                    norm_y = (ya * ya - yb * yb * k) % p
                    if (trace_x + norm_y) % p == 0:
                        if (ya, yb) not in y_special:
                            base_excluded.add(x + y * q)

    # Start with S = base_excluded
    S = set(base_excluded)
    all_points = set(range(q * q))
    N = all_points - S

    # For each point, determine which directions are "blocked" by S
    # A direction v=(1,m) or v=(0,1) is valid for point x if all other points on the line are in N
    # i.e., no other point on the line is in S
    
    # Precompute lines for each direction
    # For direction (1, m): line through x with slope m contains points {x + t*(1, m) : t in F_49}
    # For direction (0, 1): vertical line through x
    
    # For greedy augmentation, we want to add points to S while maintaining:
    # Every point still has at least one valid direction
    
    # Key insight: when we add a point p to S, we only invalidate directions for p itself
    # (since p was previously in N, it needed a valid direction, and now it's in S so it doesn't need one)
    # But wait - we also need to ensure that for every OTHER point q, if the line through q in some direction
    # passes through p, then q still has ANOTHER valid direction.
    
    # Actually, let me reconsider. The condition is:
    # For every point x (in F_49^2), there exists a direction v such that
    # {x + t*v : t in F_49, t != 0} subset N
    # i.e., no other point on the line through x with direction v is in S.
    
    # So if we add point p to S, we need to check:
    # For every point x != p, and for every direction v, if the line through x with direction v
    # contains p, then x must have at least one OTHER direction v' whose line doesn't contain any point in S.
    
    # This is complex. Let me use a simpler approach:
    # Maintain for each point the set of valid directions.
    # When adding p to S, update the valid directions for all points whose lines (in any direction) pass through p.
    
    # For efficiency, let's precompute for each point p, which (x, v) pairs are affected.
    # Point p is on the line through x with direction (1, m) iff p = x + t*(1, m) for some t != 0
    # i.e., p - x = t*(1, m), so (p_x - x_x, p_y - x_y) = (t, t*m)
    # This means p_y - x_y = m * (p_x - x_x), and t = p_x - x_x (must be nonzero)
    
    # Similarly for vertical: p_x = x_x (same x-coordinate)
    
    # Let me implement the greedy augmentation
    
    # First, compute initial valid directions for each point
    # valid_dirs[x] = set of directions that work for x
    # Direction encoding: 0..48 for (1, m) where m = 0..48, 49 for (0, 1)
    
    n_points = q * q  # 2401
    # Map point index to (x, y) coordinates
    # Point index = x + y * q
    
    # For each point, compute which directions are valid
    # A direction is valid if no other point on the line is in S
    
    # Precompute: for each point x and direction v, what are the other points on the line?
    # This is expensive (2401 * 50 * 49 operations), but let's try
    
    # Actually, let's be smarter. For each direction, group points into lines.
    
    # Directions: 50 total (49 slopes + vertical)
    # For each direction, partition the 2401 points into 49 lines of 49 points each
    
    # Let's precompute line memberships
    # line_members[dir_idx][line_id] = list of points on that line
    
    # This might be too much memory. Let's use a different approach.
    
    # For the greedy step, when considering adding point p to S:
    # 1. p must currently be in N (not in S)
    # 2. For every other point x, if x currently has exactly one valid direction,
    #    and that direction's line contains p, then adding p would make x invalid.
    
    # So we need to track for each point:
    # - Number of valid directions
    # - Which specific directions are valid (if count is small)
    
    # Let me implement this more carefully
    
    # Precompute for each point and each direction, whether the line is "clean" (no S points other than possibly the point itself)
    
    # Actually, let's just do the greedy with direct checking, but optimize by only checking affected points
    
    # For point p, the points affected are those x such that p lies on some line through x.
    # For direction (1, m): x = p - t*(1, m) for t in F_49, t != 0, so x = (p_x - t, p_y - t*m)
    # For direction (0, 1): x = (p_x, p_y - t) for t != 0
    
    # So the affected points are all points on any line through p (excluding p itself)
    # That's 50 * 48 = 2400 points, which is almost all points.
    
    # This suggests that adding any point to S affects almost all other points' validity.
    # So we need to be very careful.
    
    # Alternative approach: instead of greedy addition, try to find a better base construction
    # or use a different strategy entirely.
    
    # Let me try a randomized local search: start with base S, then randomly swap points
    # (remove one from S, add another to S) and check if validity is maintained
    
    # But checking validity is expensive. Let me think of a faster way.
    
    # Actually, the key constraint is: for every point x, there exists a direction v such that
    # the line through x with direction v contains no points of S (other than x itself if x is in S).
    
    # Let me define: for each point x and direction v, let conflict(x, v) = number of points in S
    # on the line through x with direction v, excluding x itself.
    # Then x is "covered" by direction v iff conflict(x, v) = 0.
    # The Nikodym condition is: for every x, there exists v with conflict(x, v) = 0.
    
    # When we add p to S:
    # - For each x != p and each v such that p is on the line through x with direction v:
    #   conflict(x, v) increases by 1
    # - For x = p: p is now in S, so it doesn't need coverage (but we still track for consistency)
    
    # When we remove p from S:
    # - For each x != p and each v such that p is on the line through x with direction v:
    #   conflict(x, v) decreases by 1
    
    # So we can maintain conflict counts and check validity efficiently.
    
    # Let me implement this
    
    # Initialize conflict counts
    # conflict[x][v] for x in 0..2400, v in 0..49
    # v = 0..48: direction (1, v)
    # v = 49: direction (0, 1)
    
    num_dirs = q + 1  # 50
    conflict = np.zeros((n_points, num_dirs), dtype=np.int32)
    
    # For each point s in S, for each other point x, for each direction v where s is on the line through x:
    # conflict[x][v] += 1
    
    # This is O(|S| * 2400 * 50) which is too slow for |S| ~ 148
    # Let's compute it differently
    
    # For each direction v, for each line in that direction, count how many points of S are on the line
    # Then for each point x on that line, conflict[x][v] = (count - 1 if x in S else count)
    
    # Let me precompute line structures
    
    # For direction (1, m): line through (x0, y0) contains points {(x0 + t, y0 + t*m) : t in F_49}
    # The line is determined by (x0, y0 - m*x0) or equivalently by the intercept c = y0 - m*x0
    # For fixed m, lines are parameterized by c in F_49
    # Point (x, y) is on line (m, c) iff y - m*x = c
    
    # For vertical direction: line through (x0, y0) contains {(x0, y0 + t) : t in F_49}
    # Line determined by x0
    
    # Let me build line membership
    
    # For each direction, create a mapping from line_id to list of points
    # line_id for direction (1, m): c = y - m*x (in F_49, encoded as 0..48)
    # line_id for vertical: x (in F_49, encoded as 0..48)
    
    lines = {}  # (dir_idx, line_id) -> list of point indices
    
    for dir_idx in range(num_dirs):
        if dir_idx < q:
            m = dir_idx
            for x in range(q):
                for y in range(q):
                    pt = x + y * q
                    c = sub(y, mul(m, x))
                    key = (dir_idx, c)
                    if key not in lines:
                        lines[key] = []
                    lines[key].append(pt)
        else:
            # Vertical
            for x in range(q):
                for y in range(q):
                    pt = x + y * q
                    key = (dir_idx, x)
                    if key not in lines:
                        lines[key] = []
                    lines[key].append(pt)
    
    # Now compute initial conflict
    for (dir_idx, line_id), pts in lines.items():
        count_in_S = sum(1 for pt in pts if pt in S)
        for pt in pts:
            if pt in S:
                conflict[pt, dir_idx] = count_in_S - 1
            else:
                conflict[pt, dir_idx] = count_in_S
    
    # Check initial validity
    def is_valid():
        for x in range(n_points):
            if x in S:
                continue  # Points in S don't need coverage
            # Check if any direction has conflict 0
            has_valid = False
            for v in range(num_dirs):
                if conflict[x, v] == 0:
                    has_valid = True
                    break
            if not has_valid:
                return False
        return True
    
    # Verify base is valid
    assert is_valid(), "Base construction is not valid"
    
    # Greedy augmentation: try to add points from N to S
    # Try points in random order, add if validity is maintained
    
    random.seed(42)
    candidates = list(N)
    random.shuffle(candidates)
    
    added_count = 0
    for p_idx in candidates:
        if time.time() > deadline - 60:
            break
        if p_idx in S:
            continue
        
        # Try adding p_idx to S
        # Save old conflict values for affected points
        affected = []
        for v in range(num_dirs):
            # Find the line in direction v that contains p_idx
            x = p_idx % q
            y = p_idx // q
            if v < q:
                m = v
                c = sub(y, mul(m, x))
            else:
                c = x
            key = (v, c)
            line_pts = lines[key]
            for pt in line_pts:
                if pt != p_idx:
                    old_val = conflict[pt, v]
                    conflict[pt, v] = old_val + 1
                    affected.append((pt, v, old_val))
        
        # Check validity
        valid = True
        for x in range(n_points):
            if x in S or x == p_idx:
                continue
            has_valid_dir = False
            for v in range(num_dirs):
                if conflict[x, v] == 0:
                    has_valid_dir = True
                    break
            if not has_valid_dir:
                valid = False
                break
        
        if valid:
            S.add(p_idx)
            added_count += 1
        else:
            # Revert
            for (pt, v, old_val) in affected:
                conflict[pt, v] = old_val
    
    # Build result
    N = all_points - S
    points = []
    for pt in sorted(N):
        x = pt % q
        y = pt // q
        xa, xb = x % p, x // p
        ya, yb = y % p, y // p
        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
