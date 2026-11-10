# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 via greedy complement expansion.
    
    A set N is Nikodym if for every point x in F_49^2, there exists a direction v
    (either (1,m) for some m in F_49, or (0,1)) such that all other points on the
    line {x + t*v : t in F_49} are in N.
    
    We maximize the complement S = F_49^2 \ N by greedily adding points to S
    while maintaining validity.
    
    Returns:
        np.ndarray of shape (k, 2, 2) with k >= 1, listing points of N.
    """
    p = 7
    q = p * p  # 49
    
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 580  # 20s safety margin
    
    # Field parameter: alpha^2 = k where k is smallest QR mod 7
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    
    # Total points: q^2 = 2401
    total = q * q
    
    # Enumerate all points as linear indices 0..total-1
    # Point index = ((xa*p + xb)*q) + (ya*p + yb)
    # So: xa = idx // (p*q*q), xb = (idx // (p*q)) % p, ya = (idx // p) % q, yb = idx % p
    # Wait, let me redo: xa in [0,p), xb in [0,p), ya in [0,p), yb in [0,p)
    # x = xa + xb*alpha, y = ya + yb*alpha
    # Linear index: idx = ((xa*p + xb)*p + ya)*p + yb
    # So: yb = idx % p, ya = (idx // p) % p, xb = (idx // (p*p)) % p, xa = idx // (p*p*p)
    
    p3 = p * p * p  # 343
    p4 = p * p * p * p  # 2401
    
    # Precompute coordinates for all points
    # x_coords[i] = [xa, xb], y_coords[i] = [ya, yb]
    all_xa = np.zeros(total, dtype=np.int64)
    all_xb = np.zeros(total, dtype=np.int64)
    all_ya = np.zeros(total, dtype=np.int64)
    all_yb = np.zeros(total, dtype=np.int64)
    
    for idx in range(total):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = idx // (p * p * p)
        all_xa[idx] = xa
        all_xb[idx] = xb
        all_ya[idx] = ya
        all_yb[idx] = yb
    
    # Precompute all lines
    # Directions: (1, m) for m in F_49 (49 directions) + (0, 1) (1 direction) = 50 directions
    # Each direction has q = 49 lines (one per point not on the "base" line)
    # Actually, for direction (1,m): lines are parameterized by y-intercept
    # For direction (0,1): vertical lines, parameterized by x
    
    # Let's precompute for each point and each direction, the set of other points on that line
    # This is expensive but let's try to vectorize
    
    # Number of directions: 50
    n_dirs = q + 1  # 49 + 1 = 50
    
    # For each direction d and each point i, compute the 48 other points on the line through i with direction d
    # line_points[d, i, j] = index of the j-th other point (j=0..47)
    
    # Actually, let's think about this differently.
    # For a given point x and direction v, the line is {x + t*v : t in F_49}.
    # The other 48 points are x + t*v for t in F_49 \ {0}.
    
    # Let's precompute the mapping from (point, direction) to the 48 other points.
    # This is 2401 * 50 * 48 = ~5.7M entries. Might be memory heavy but let's try.
    
    # Actually, let's be smarter. For each direction, we can precompute the line structure.
    
    # Direction (1, m) where m = (ma, mb) in F_49:
    # Line through (x, y) with direction (1, m): points are (x + t, y + t*m) for t in F_49
    # The "base" point is at t=0. Other points at t = 1, 2, ..., 48 (in field arithmetic).
    
    # Direction (0, 1):
    # Line through (x, y) with direction (0, 1): points are (x, y + t) for t in F_49
    # Other points at t = 1, 2, ..., 48.
    
    # Let's precompute for each direction and each point, the list of other 48 points.
    # We'll store this as a 3D array: other_points[dir_idx, point_idx, other_idx]
    
    # But 50 * 2401 * 48 = 5,762,400 integers. That's about 46 MB for int32. Should be OK.
    
    # First, let's build the field multiplication table for F_49.
    # mul[a, b] = a * b in F_49, where a, b are linear indices 0..48
    # a = (aa, ab), b = (ba, bb)
    # a*b = (aa*ba - ab*bb*k, aa*bb + ab*ba) mod p
    
    mul_table = np.zeros((q, q), dtype=np.int64)
    for a in range(q):
        aa = a // p
        ab = a % p
        for b in range(q):
            ba = b // p
            bb = b % p
            res_a = (aa * ba - ab * bb * k) % p
            res_b = (aa * bb + ab * ba) % p
            mul_table[a, b] = res_a * p + res_b
    
    # Precompute other_points
    # For each of the 50 directions and each of the 2401 points, compute the 48 other points.
    
    # Direction index 0..48: (1, m) where m = direction index
    # Direction index 49: (0, 1)
    
    other_points = np.zeros((n_dirs, total, q - 1), dtype=np.int64)
    
    for d in range(n_dirs):
        if d < q:
            # Direction (1, m) where m = d
            md = d
            for i in range(total):
                xa = all_xa[i]
                xb = all_xb[i]
                ya = all_ya[i]
                yb = all_yb[i]
                
                for t in range(1, q):
                    # Point: (x + t, y + t*m)
                    # t is in F_49, so t = (ta, tb)
                    ta = t // p
                    tb = t % p
                    
                    # x' = x + t = (xa + ta, xb + tb) mod p
                    xa_new = (xa + ta) % p
                    xb_new = (xb + tb) % p
                    
                    # y' = y + t*m
                    tm = mul_table[t, md]
                    tma = tm // p
                    tmb = tm % p
                    
                    ya_new = (ya + tma) % p
                    yb_new = (yb + tmb) % p
                    
                    # Linear index of new point
                    new_idx = xa_new * p3 + xb_new * p * p + ya_new * p + yb_new
                    other_points[d, i, t - 1] = new_idx
        else:
            # Direction (0, 1)
            for i in range(total):
                xa = all_xa[i]
                xb = all_xb[i]
                ya = all_ya[i]
                yb = all_yb[i]
                
                for t in range(1, q):
                    # Point: (x, y + t) where t = (ta, tb)
                    ta = t // p
                    tb = t % p
                    
                    ya_new = (ya + ta) % p
                    yb_new = (yb + tb) % p
                    
                    new_idx = xa * p3 + xb * p * p + ya_new * p + yb_new
                    other_points[d, i, t - 1] = new_idx
    
    # Now we have other_points[d, i, j] = the j-th other point on line through i with direction d
    
    # Greedy expansion:
    # S = set of points in complement (initially empty)
    # N = all points not in S
    # For each point x, it's valid if there exists a direction d such that
    # all other_points[d, x, :] are in N (i.e., not in S).
    
    # We want to maximize |S| by adding points one at a time.
    
    # Strategy: Try adding each point to S. If after adding, all points are still valid,
    # keep it in S. Otherwise, don't add it.
    
    # This is O(total * total * n_dirs * q) in the worst case, which is too slow.
    # Let's think of a better approach.
    
    # Alternative: For each point x, precompute which directions are "available"
    # (i.e., all other points on the line are not in S). Initially all 50 directions are available.
    # When we add a point y to S, for each point x and each direction d where y is on the
    # line through x with direction d, we need to check if all other points on that line are still in N.
    
    # Actually, let's just do a simple greedy: for each point in some order, try to add it to S.
    # Check validity after each addition.
    
    # To speed up the validity check, maintain for each point x the count of available directions.
    # Initially, all points have 50 available directions.
    # When we add y to S, for each point x, for each direction d where y is among the other_points[d, x, :],
    # we need to check if all other points on that line are still in N.
    
    # This is still complex. Let's just do a simpler approach:
    # 1. Start with S = empty.
    # 2. For each point x (in some order), try adding x to S.
    # 3. Check if the resulting N is still valid (all points have at least one valid line).
    # 4. If valid, keep x in S. Otherwise, remove x from S.
    
    # The validity check is O(total * n_dirs * q) = O(2401 * 50 * 48) ≈ 5.7M operations per check.
    # With 2401 points, that's ~13.7B operations total. Too slow in Python.
    
    # Let's use a smarter approach: maintain the count of available directions per point.
    
    # available_dirs[x] = number of directions d such that all other_points[d, x, :] are not in S.
    
    # Initially, available_dirs[x] = 50 for all x.
    
    # When we add y to S:
    # For each point x and each direction d where y is in other_points[d, x, :]:
    #   Check if all other_points[d, x, :] are still not in S (excluding y which is now in S).
    #   If not, decrement available_dirs[x].
    
    # But we also need to handle the case where y = x (we're adding x itself to S).
    # In that case, for x itself, we need to check if there's still a direction where all other points are in N.
    
    # Let's implement this more carefully.
    
    # First, precompute for each point y and each direction d, the set of points x such that y is in other_points[d, x, :].
    # This is the "reverse" mapping.
    
    # reverse[d, y] = list of (x, j) such that other_points[d, x, j] = y
    
    # Actually, for a fixed direction d and point y, the points x such that y is on the line through x with direction d
    # are exactly the points on the same line as y with direction d (including y itself, but we exclude y since
    # other_points excludes the base point).
    
    # For direction (1, m): the line through y = (ya, yb) with direction (1, m) consists of points
    # (ya + t, yb + t*m) for t in F_49. The base point is at t=0, which is y itself.
    # So the other 48 points are at t = 1, ..., 48.
    # For any of these other points x = (ya + t, yb + t*m), y is in other_points[d, x, :].
    
    # So for direction d and point y, the points x such that y is in other_points[d, x, :] are exactly
    # the other 48 points on the line through y with direction d.
    
    # This means: when we add y to S, for each direction d, for each of the 48 other points x on the line
    # through y with direction d, we need to check if all other_points[d, x, :] are still in N.
    
    # Let's precompute the line membership:
    # line_members[d, y] = list of 48 points x such that y is in other_points[d, x, :]
    # Which is the same as: the 48 other points on the line through y with direction d.
    
    # But we already have other_points[d, y, :] which gives the 48 other points on the line through y.
    # And by symmetry, y is in other_points[d, x, :] for each x in other_points[d, y, :].
    
    # So line_members[d, y] = other_points[d, y, :]
    
    # Great, so we can use other_points directly.
    
    # Now, the greedy algorithm:
    
    # S = set()  # complement
    # in_S = np.zeros(total, dtype=bool)
    # available = np.full(total, n_dirs, dtype=np.int64)  # count of available directions per point
    
    # For each point x (in some order):
    #   Try adding x to S.
    #   For each direction d:
    #     Check if all other_points[d, x, :] are not in S.
    #     If yes, x has direction d available.
    #   If x has at least one available direction, and all other points still have at least one available direction,
    #   then keep x in S.
    
    # But checking "all other points still have at least one available direction" is expensive.
    
    # Let's simplify: just check if x itself still has at least one available direction after adding to S.
    # This is a necessary but not sufficient condition. We might add points that break other points.
    
    # To be safe, let's do a full validity check after each addition, but only for the affected points.
    
    # Actually, let's just do a simple greedy with full validity checks, but limit the number of iterations
    # based on time.
    
    # Given the time constraints, let's try a different approach:
    # Use the algebraic construction as a starting point, then try to expand the complement.
    
    # Start with the baseline S (excluded points from the Hermitian unital construction).
    
    # Build the initial S from the baseline construction.
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    in_S = np.zeros(total, dtype=bool)
    for idx in range(total):
        xa = all_xa[idx]
        xb = all_xb[idx]
        ya = all_ya[idx]
        yb = all_yb[idx]
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            in_S[idx] = True
    
    # Now try to expand S by adding more points.
    # For each point not in S, try adding it and check validity.
    
    # To check validity efficiently, maintain the count of available directions per point.
    
    # available[d, x] = 1 if all other_points[d, x, :] are not in S, else 0.
    # avail_count[x] = sum_d available[d, x]
    
    # Initialize available and avail_count.
    
    available = np.zeros((n_dirs, total), dtype=bool)
    for d in range(n_dirs):
        for i in range(total):
            # Check if all other_points[d, i, :] are not in S
            others = other_points[d, i, :]
            if not np.any(in_S[others]):
                available[d, i] = True
    
    avail_count = available.sum(axis=0)  # shape (total,)
    
    # Check initial validity: all points should have avail_count >= 1
    if np.any(avail_count < 1):
        # Invalid initial state, fall back to baseline
        points = []
        for idx in range(total):
            if not in_S[idx]:
                xa = all_xa[idx]
                xb = all_xb[idx]
                ya = all_ya[idx]
                yb = all_yb[idx]
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
    
    # Now greedily try to add points to S.
    # Order: try points not in S, in some order.
    
    not_in_S = np.where(~in_S)[0]
    
    # Shuffle for diversity? Or just use natural order.
    # Let's use a random order for better results.
    rng = np.random.RandomState(42)
    rng.shuffle(not_in_S)
    
    added = 0
    for idx in not_in_S:
        if time.time() > deadline:
            break
        
        # Try adding idx to S
        in_S[idx] = True
        
        # Update available for all points affected by adding idx.
        # A point x is affected if idx is in other_points[d, x, :] for some d.
        # Which means x is in other_points[d, idx, :] for some d.
        
        affected = set()
        for d in range(n_dirs):
            others = other_points[d, idx, :]
            for x in others:
                affected.add(x)
        
        # Also, idx itself is affected (we need to check if it still has available directions).
        affected.add(idx)
        
        # For each affected point x, recompute available[d, x] for all d.
        for x in affected:
            for d in range(n_dirs):
                others = other_points[d, x, :]
                if np.any(in_S[others]):
                    available[d, x] = False
                else:
                    available[d, x] = True
        
        avail_count = available.sum(axis=0)
        
        # Check if all points still have at least one available direction.
        if np.any(avail_count < 1):
            # Invalid, revert
            in_S[idx] = False
            # Recompute available for affected points
            for x in affected:
                for d in range(n_dirs):
                    others = other_points[d, x, :]
                    if np.any(in_S[others]):
                        available[d, x] = False
                    else:
                        available[d, x] = True
        else:
            added += 1
    
    # Build the result: N = all points not in S
    points = []
    for idx in range(total):
        if not in_S[idx]:
            xa = all_xa[idx]
            xb = all_xb[idx]
            ya = all_ya[idx]
            yb = all_yb[idx]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
