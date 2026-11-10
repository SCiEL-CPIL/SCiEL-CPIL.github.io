# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    
    Strategy:
    1. Start with the Hermitian unital-based baseline (excluded set S = U \ B).
    2. Use greedy expansion with witness tracking to safely remove more points
       from N (adding to complement C), ensuring no point loses its last valid line.
    3. Return the resulting N as an array of shape (k, 2, 2).
    
    The Nikodym condition: for every point x in F_49^2, there exists a direction
    v (either (1,m) for some m in F_49, or (0,1)) such that all other 48 points
    on the line {x + t*v : t in F_49} belong to N.
    """
    p = 7
    k_const = 3  # alpha^2 = 3, smallest quadratic non-residue mod 7
    
    # Field arithmetic helpers
    def f49_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def f49_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0*b0 + a1*b1*3) + (a0*b1 + a1*b0)*alpha
        a0, a1 = a
        b0, b1 = b
        return ((a0 * b0 + a1 * b1 * k_const) % p, (a0 * b1 + a1 * b0) % p)
    
    def f49_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    # Total points: 49 * 49 = 2401
    n_total = 2401
    
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)
    
    # Generate all 49 slopes m in F_49
    slopes = [(ma, mb) for ma in range(p) for mb in range(p)]
    
    # Precompute lines for all 50 directions (49 slopes + vertical)
    # For each direction d and each point idx, store the 49 points on the line
    # Memory: 50 * 2401 * 49 ~ 5.88M integers, manageable
    
    # Actually, let's be smarter. For each direction, the 2401 points partition into
    # 49 lines of 49 points each. We'll store for each direction and each point,
    # the line ID and the other 48 points on that line.
    
    # But even simpler: for the greedy algorithm, we need to know for each point
    # which lines pass through it, and for each line which points are on it.
    
    # Let's precompute: for each direction d (0..49), and each point idx,
    # the set of other 48 points on the line through idx in direction d.
    
    # Direction 0..48: v = (1, m) where m = slopes[d]
    # Direction 49: v = (0, 1)
    
    # For direction v = (1, m) with m = (ma, mb):
    # Line through (xa, xb, ya, yb) in direction (1, m):
    # Points: (xa + ta, tb, ya + ta*ma + tb*mb*3, yb + ta*mb + tb*ma) for ta, tb in F_7
    # where t = ta + tb*alpha
    
    # For direction v = (0, 1):
    # Line through (xa, xb, ya, yb):
    # Points: (xa, xb, ya + ta, yb + tb) for ta, tb in F_7
    
    # Let's precompute for each direction and each point, the 48 other points.
    # We'll store this as a list of lists: lines[direction][point_idx] = array of 48 point indices
    
    # Actually, for memory efficiency, let's compute on the fly or use a more compact representation.
    
    # Alternative approach: for each point, track how many "clean" directions it has.
    # A direction d is "clean" for point p if all 48 other points on line(p, d) are in N.
    
    # Let's precompute the line membership. For each direction d, the 49 lines partition
    # the 2401 points. We'll store for each direction d, an array of shape (2401,) giving
    # the line_id for each point, and for each line, the 49 points on it.
    
    # 50 directions * 49 lines * 49 points = 119,050 entries. Very manageable.
    
    line_members = []  # line_members[d][l] = array of 49 point indices
    point_line_id = []  # point_line_id[d][idx] = line_id for point idx in direction d
    
    for d in range(50):
        members = []
        lid = np.zeros(n_total, dtype=np.int32)
        
        if d < 49:
            ma, mb = slopes[d]
            # For direction (1, m): line through (xa, xb, ya, yb)
            # The line is determined by the "invariant" values.
            # Two points are on the same line iff they differ by t*(1, m) for some t in F_49.
            # x' = x + t, y' = y + t*m
            # So x - y/m is constant? No, m might be 0.
            # Let's use: for a fixed line, all points satisfy:
            # (xa + ta, tb, ya + ta*ma + tb*mb*3, yb + ta*mb + tb*ma)
            # The free parameters are ta, tb. The "base" point can be taken as the one with ta=0, tb=0,
            # but that's not unique. Instead, note that:
            # xa' - ya' * (something)... this is getting complicated.
            
            # Simpler: iterate over all 49 possible "base" lines.
            # A line in direction (1, m) can be parameterized by a representative point.
            # Two points (xa, xb, ya, yb) and (xa', xb', ya', yb') are on the same line
            # iff (xa', xb', ya', yb') - (xa, xb, ya, yb) = t*(1, ma, mb) for some t in F_49.
            # t = (ta, tb), so:
            # xa' - xa = ta
            # xb' - xb = tb
            # ya' - ya = ta*ma + tb*mb*3
            # yb' - yb = ta*mb + tb*ma
            # From first two: ta = xa' - xa, tb = xb' - xb
            # Check: ya' - ya = (xa'-xa)*ma + (xb'-xb)*mb*3
            #         yb' - yb = (xa'-xa)*mb + (xb'-xb)*ma
            
            # So the line is determined by the values:
            # c1 = ya - xa*ma - xb*mb*3 (mod 7)
            # c2 = yb - xa*mb - xb*ma (mod 7)
            
            # For each (c1, c2) in F_7 x F_7, the line is:
            # {(xa, xb, ya, yb) : ya = c1 + xa*ma + xb*mb*3, yb = c2 + xa*mb + xb*ma}
            # for xa, xb in F_7.
            
            for c1 in range(p):
                for c2 in range(p):
                    pts = []
                    for xa in range(p):
                        for xb in range(p):
                            ya = (c1 + xa * ma + xb * mb * k_const) % p
                            yb = (c2 + xa * mb + xb * ma) % p
                            idx = point_to_idx(xa, xb, ya, yb)
                            pts.append(idx)
                            lid[idx] = len(members)
                    members.append(np.array(pts, dtype=np.int32))
        else:
            # Vertical direction v = (0, 1)
            # Line through (xa, xb, ya, yb): {(xa, xb, ya + ta, yb + tb) : ta, tb in F_7}
            # Determined by (xa, xb).
            for xa in range(p):
                for xb in range(p):
                    pts = []
                    for ta in range(p):
                        for tb in range(p):
                            ya = ta
                            yb = tb
                            idx = point_to_idx(xa, xb, ya, yb)
                            pts.append(idx)
                            lid[idx] = len(members)
                    members.append(np.array(pts, dtype=np.int32))
        
        line_members.append(members)
        point_line_id.append(lid)
    
    # Initial excluded set: Hermitian unital points with y_a in QR
    # QR = {0, 1, 2, 4}
    QR = {0, 1, 2, 4}
    
    in_N = np.ones(n_total, dtype=bool)
    
    # Apply initial exclusion
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
                    if in_unital and ya in QR:
                        idx = point_to_idx(xa, xb, ya, yb)
                        in_N[idx] = False
    
    # Greedy expansion: try to remove more points from N
    # For each point p not yet in N, check if we can safely add it to N... 
    # Wait, we want to MINIMIZE N, so we want to REMOVE points from N.
    # But we start with some points already excluded. We want to exclude MORE.
    
    # Actually, the greedy approach is: start with N = all points, then greedily remove points.
    # But we have a good starting point. Let's try to remove more points from N.
    
    # For a point p to be removed from N, we need to ensure that for every point q,
    # q still has at least one clean direction after removal.
    
    # Removing p only affects points q that share a line with p in some direction.
    # Specifically, for each direction d, let L be the line through p in direction d.
    # For each q in L \ {p}, the direction d is no longer clean for q if p was the only
    # "bad" point on L for q... wait, no.
    
    # A direction d is clean for q if all other 48 points on line(q, d) are in N.
    # If we remove p from N, then for any q such that p is on line(q, d), the direction d
    # is no longer clean for q (unless q = p, but p is being removed).
    
    # So for each q that shares a line with p in some direction d, if d was the ONLY clean
    # direction for q, then q loses its last clean direction, which is invalid.
    
    # We need to track for each point q, how many clean directions it has.
    
    # Let's compute the number of clean directions for each point.
    # clean_count[q] = number of directions d such that all other 48 points on line(q, d) are in N.
    
    def compute_clean_counts(in_N):
        clean = np.zeros(n_total, dtype=np.int32)
        for d in range(50):
            for l in range(49):
                pts = line_members[d][l]
                # For each point in this line, count how many other points are in N
                in_N_line = in_N[pts]
                # For point pts[i], the number of other points in N is sum(in_N_line) - in_N_line[i]
                # Direction d is clean for pts[i] if sum(in_N_line) - in_N_line[i] == 48
                total_in = np.sum(in_N_line)
                for i in range(49):
                    if total_in - in_N_line[i] == 48:
                        clean[pts[i]] += 1
        return clean
    
    # Initial clean counts
    clean_counts = compute_clean_counts(in_N)
    
    # Verify initial state is valid: every point should have at least 1 clean direction
    # (This should be true for the baseline construction)
    
    # Greedy removal: try to remove points from N
    # For each point p in N, check if removing it keeps all points with clean_count >= 1.
    
    # To make this efficient, we process points in some order and check validity.
    
    # Let's try a simple greedy: iterate through all points, try to remove each one.
    # If removal is valid, remove it. Repeat until no more points can be removed.
    
    # For efficiency, we'll do multiple passes.
    
    deadline = 550  # seconds
    start_time = time.time()
    
    # Try to remove points
    changed = True
    while changed:
        changed = False
        for idx in range(n_total):
            if time.time() - start_time > deadline:
                break
            if not in_N[idx]:
                continue
            
            # Try removing idx
            # Check if any point loses its last clean direction
            # Points affected: those sharing a line with idx in some direction
            # For each direction d, the line through idx in direction d
            # All other 48 points on that line are affected
            
            valid = True
            for d in range(50):
                l = point_line_id[d][idx]
                pts = line_members[d][l]
                in_N_line = in_N[pts]
                total_in = np.sum(in_N_line)
                # After removing idx, total_in decreases by 1
                new_total = total_in - 1
                for i in range(49):
                    qi = pts[i]
                    if qi == idx:
                        continue
                    # Current clean status for direction d: total_in - in_N_line[i] == 48
                    # New clean status: new_total - in_N_line[i] == 48
                    was_clean = (total_in - in_N_line[i] == 48)
                    is_clean = (new_total - in_N_line[i] == 48)
                    if was_clean and not is_clean:
                        # qi loses direction d
                        if clean_counts[qi] == 1:
                            # qi had only this one clean direction, now it has 0 -> invalid
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                in_N[idx] = False
                # Update clean counts
                clean_counts = compute_clean_counts(in_N)
                changed = True
        
        if time.time() - start_time > deadline:
            break
    
    # Build result
    points = []
    for idx in range(n_total):
        if in_N[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
