# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _generate_seed_excluded():
    """Return boolean array where True = point is in S (excluded from N)."""
    squares = set((u * u) % p for u in range(p))
    excluded = np.zeros(N_TOTAL, dtype=np.int8)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Unital condition: 2*xa + ya^2 - yb^2*K = 0 mod 7
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        # y is special if ya is a square mod 7
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
    return excluded

def _compute_bad_counts(excluded):
    """
    For each point x, count how many directions v have at least one S point on the
    punctured line through x in direction v.
    
    A point x is "covered" (can be in N) iff bad_count[x] == 0.
    A point x in S is "validly excluded" iff bad_count[x] == 1 (only itself on line for some dir).
    
    We track: for each point, number of directions where the punctured line intersects S.
    bad[x] = number of directions where punctured line through x hits S.
    Point x is covered iff bad[x] == 0.
    """
    # bad[x] = count of directions where exists t!=0: x+t*v in S
    # Equivalently: for direction v, x is "bad" in that direction iff the line through x in dir v
    # contains any S point other than x.
    
    # For each direction, compute which points are bad in that direction.
    # A point x is bad in direction v iff there exists s in S, s != x, such that s - x is parallel to v.
    # i.e., s = x + t*v for some t != 0.
    
    # Alternative: for each s in S, it makes x = s + t*v bad for all t != 0, for each direction v.
    # But that's |S| * 49 * 50 operations which is too many.
    
    # Better: for each direction v, for each line in that direction, count S points on it.
    # A point x on the line is bad in direction v iff count_S(line) >= 1 and (x not in S or count_S(line) >= 2).
    # Actually: x is bad in dir v iff there exists s in S on the line with s != x.
    #   iff count_S(line) - (1 if x in S else 0) >= 1
    #   iff count_S(line) > (1 if x in S else 0)
    
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    
    # Precompute all points as arrays for vectorization
    # For each of the 50 directions, process all lines.
    
    # Directions: v=(0,1) [vertical] and v=(1, m) for m in F_49
    # For v=(0,1): lines are parameterized by (xa, xb). Line contains all (xa, xb, ya, yb) for ya,yb in F_7.
    #   49 lines, each with 49 points.
    
    # For v=(1, m): lines parameterized by... 
    #   Point (xa,xb,ya,yb), direction (1, ma+mb*alpha).
    #   t = ta + tb*alpha.
    #   x + t = (xa+ta, xb+tb)
    #   y + t*m: t*m = (ta*ma + 3*tb*mb) + (ta*mb + tb*ma)*alpha
    #   So y_a' = ya + ta*ma + 3*tb*mb, y_b' = yb + ta*mb + tb*ma
    #   The line through (xa,xb,ya,yb) in dir (1,m) consists of all points (xa+ta, xb+tb, ya+ta*ma+3*tb*mb, yb+ta*mb+tb*ma) for ta,tb in F_7.
    #   To parameterize lines: fix the "intercept". As t varies, (x,y) traces a line.
    #   The line is determined by: y - m*x = const (in F_49). 
    #   y - m*x = (ya + yb*a) - (ma+mb*a)(xa+xb*a)
    #   = ya + yb*a - ma*xa - ma*xb*a - mb*a*xa - mb*a*xb*a
    #   = (ya - ma*xa) + (yb - ma*xb - mb*xa - 3*mb*xb)*a
    #   Wait: mb*a*xa = mb*xa*a, mb*a*xb*a = mb*xb*a^2 = 3*mb*xb
    #   So: y - m*x = (ya - ma*xa - 3*mb*xb) + (yb - ma*xb - mb*xa)*a
    #   This is a constant along the line. So lines in direction (1,m) are parameterized by (c_a, c_b) where:
    #   c_a = ya - ma*xa - 3*mb*xb
    #   c_b = yb - ma*xb - mb*xa
    #   There are 49 such lines for each m.
    
    # For each direction, iterate over all 49 lines, count S on each, then mark bad points.
    
    # Direction 0: v = (0, 1), vertical
    # Lines parameterized by (xa, xb), 49 lines
    for xa in range(p):
        for xb in range(p):
            # Line: all (xa, xb, ya, yb) for ya, yb in 0..6
            count_s = 0
            for ya in range(p):
                for yb in range(p):
                    if excluded[_point_to_idx(xa, xb, ya, yb)]:
                        count_s += 1
            if count_s > 0:
                for ya in range(p):
                    for yb in range(p):
                        idx = _point_to_idx(xa, xb, ya, yb)
                        if count_s > (1 if excluded[idx] else 0):
                            bad[idx] += 1
    
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    for ma in range(p):
        for mb in range(p):
            # Lines parameterized by (c_a, c_b)
            # c_a = ya - ma*xa - 3*mb*xb
            # c_b = yb - ma*xb - mb*xa
            # For each (c_a, c_b), the line consists of points where these hold.
            # We can iterate over xa, xb and compute ya, yb:
            #   ya = c_a + ma*xa + 3*mb*xb
            #   yb = c_b + ma*xb + mb*xa
            # But different (xa,xb) give different (ya,yb), and we need all 49 points on the line.
            # Actually, for a fixed (c_a, c_b), as (xa, xb) ranges over F_7^2, 
            # (ya, yb) = (c_a + ma*xa + 3*mb*xb, c_b + ma*xb + mb*xa)
            # This gives 49 points (one for each (xa,xb) pair... wait, 7*7=49 pairs, but do they give 49 distinct points?
            # The map (xa,xb) -> (xa,xb, ya,yb) is injective since (xa,xb) determines the point's x-coord.
            # So yes, 49 distinct points per line.
            
            for ca in range(p):
                for cb in range(p):
                    count_s = 0
                    line_points = []
                    for xa in range(p):
                        for xb in range(p):
                            ya = (ca + ma * xa + 3 * mb * xb) % p
                            yb = (cb + ma * xb + mb * xa) % p
                            idx = _point_to_idx(xa, xb, ya, yb)
                            line_points.append(idx)
                            if excluded[idx]:
                                count_s += 1
                    if count_s > 0:
                        for idx in line_points:
                            if count_s > (1 if excluded[idx] else 0):
                                bad[idx] += 1
    
    return bad

def _count_uncovered(excluded, bad):
    """
    Count points that are NOT covered by the Nikodym condition.
    A point x is covered iff bad[x] == 0 (no direction has another S point on its line).
    But wait: for x in S, it's covered iff there exists a direction where x is the ONLY S point on the line.
    That means: for some direction v, count_S(line_v(x)) == 1.
    In terms of bad: bad[x] counts directions where punctured line hits S.
    If x in S and bad[x] == 0: for ALL directions, no other S point on line. So x is covered.
    If x in S and bad[x] > 0: for some direction, another S point exists. But maybe for ANOTHER direction, x is alone.
    
    Hmm, my bad count is: number of directions where punctured line intersects S.
    For x in S: x is covered iff there exists a direction where punctured line does NOT intersect S.
    i.e., bad[x] < 50 (total directions).
    
    For x not in S: x is covered iff for ALL... no, iff there EXISTS a direction where punctured line does NOT intersect S.
    i.e., bad[x] < 50.
    
    Wait, I think I have it backwards. Let me re-derive.
    
    Nikodym condition: for every point x, there exists a direction v such that for all t != 0, x + t*v is in N (not in S).
    
    So x is covered iff there exists v such that the punctured line in direction v through x contains NO S points.
    
    bad[x] = number of directions v where the punctured line through x in direction v DOES contain at least one S point.
    
    x is covered iff bad[x] < 50 (i.e., there's at least one direction where punctured line has no S).
    x is uncovered iff bad[x] == 50 (all 50 directions have another S point on the line).
    
    So uncovered count = number of x with bad[x] == 50.
    """
    return int(np.sum(bad == 50))

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Generate initial S from seed
    excluded = _generate_seed_excluded()
    
    # Compute initial bad counts
    bad = _compute_bad_counts(excluded)
    uncovered = _count_uncovered(excluded, bad)
    
    elapsed = time.time() - start
    if elapsed > 600:
        # Just return seed
        pass
    
    # Greedy: try to remove points from S to reduce uncovered count
    # A point s in S can be removed if doing so doesn't increase uncovered count.
    # Removing s: for each direction v, the lines that had s on them now have one fewer S point.
    # This can only decrease bad counts (or keep same), never increase.
    # So removing s from S can only decrease or maintain uncovered count.
    # Therefore, we can always remove any s from S!
    # Wait, that means we should remove ALL of S, giving N = entire space.
    # But that's trivially valid (every point is on a line with all other points in N).
    # But the metric is 2401 - |N|, so we want N as LARGE as possible, i.e., S as SMALL as possible.
    # If S = empty, N = all 2401 points, complement = 0. Score = 0.
    # We want to MAXIMIZE complement = 2401 - |N| = |S|.
    # So we want S as LARGE as possible while maintaining validity!
    
    # I had it backwards! The metric is the complement size, which is |S|.
    # We want to MAXIMIZE |S| subject to the Nikodym condition on N = complement of S.
    
    # So the seed gives some |S|. We want to ADD points to S while maintaining validity.
    # A point x can be added to S iff after adding, the Nikodym condition still holds for ALL points.
    # Adding x to S: for each other point y, the punctured lines through y may now contain x.
    # This can only increase bad counts, potentially making some points uncovered.
    # So we can add x iff for all y, bad[y] < 50 after the addition.
    
    # Actually, let me reconsider. The task says:
    # "every point x of F_49^2 (in N or not) must lie on a line ... whose points other than x all belong to N"
    
    # So for x in S: x must lie on a line where all OTHER points are in N (not in S).
    # For x in N: x must lie on a line where all OTHER points are in N (not in S).
    
    # In both cases: there exists a direction v such that the punctured line through x in direction v has NO S points.
    
    # So my bad count analysis is correct: x is validly placed iff bad[x] < 50.
    # The Nikodym condition holds iff ALL points have bad[x] < 50.
    
    # To maximize |S|, we want to add as many points to S as possible while keeping all bad[x] < 50.
    
    # Greedy: start with seed S, try adding each point not in S, if it doesn't make any bad[x] == 50, add it.
    
    # But this is expensive. Let me think of a better approach.
    
    # Actually, the seed construction is already quite good. Let me just try to add points.
    
    # For each candidate point c not in S:
    #   Adding c to S increases bad[x] by 1 for all x such that c is on the punctured line through x in some direction.
    #   c is on punctured line through x in direction v iff x = c + t*v for some t != 0.
    #   So x = c + t*v, t != 0. For each direction v (50 directions), there are 48 such x values.
    #   Total: 50 * 48 = 2400 points x get their bad count increased by 1.
    #   We need to check that none of these 2400 points have bad[x] == 49 before the addition (would become 50).
    
    # This is feasible! Let's implement it.
    
    # Precompute for each point c, the list of (x_idx, direction) pairs... 
    # Actually, for each c, the affected x's are: for each of 50 directions v, for each t in F_49\{0}, x = c + t*v.
    # That's 50 * 48 = 2400 points. For each, we check if bad[x] == 49.
    
    # Total work: |not in S| * 2400 checks. |not in S| ≈ 2401 - |S_seed|.
    # |S_seed|: let me estimate. Unital has 49*49 = 2401 points? No.
    # Unital condition: 2*xa + ya^2 - 3*yb^2 = 0 mod 7.
    # For each (ya, yb), the condition determines xa (since 2 is invertible mod 7, xa = (3*yb^2 - ya^2) * 4 mod 7, since 2^{-1} = 4 mod 7).
    # Wait: 2*xa = -ya^2 + 3*yb^2, so xa = (-ya^2 + 3*yb^2) * 4 mod 7.
    # For each (ya, yb) pair (49 pairs), there's exactly 1 value of xa. xb is free (7 values).
    # So unital has 49 * 7 = 343 points.
    # Of these, y is special if ya is a square. Squares mod 7: {0, 1, 2, 4}. So 4 values of ya.
    # For each special ya (4 values), yb can be anything (7 values): 4*7 = 28 special y's.
    # For each special y, xa is determined, xb free: 28 * 7 = 196 points are NOT in S.
    # Non-special ya: 3 values (3, 5, 6). For each, 7 values of yb: 3*7 = 21 non-special y's.
    # S has 21 * 7 = 147 points.
    # So |S_seed| = 147, |N_seed| = 2401 - 147 = 2254.
    # Complement = 147.
    
    # We want to grow S from 147 to as large as possible.
    
    # Let me implement the greedy addition.
    
    in_S = excluded.copy()
    bad = _compute_bad_counts(in_S)
    
    # Build list of candidates not in S
    candidates = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            candidates.append(idx)
    
    # For each candidate, precompute the 2400 affected point indices
    # Actually, let's do it on the fly for memory efficiency.
    
    # Precompute t values and direction multipliers
    # For direction (0,1): t = (ta, tb), x = c + (0, t) = (ca, cb, ya+ta, yb+tb)
    # For direction (1, m): t = (ta, tb), x = c + (t, t*m)
    #   x_a = ca + ta
    #   x_b = cb + tb  
    #   y_a = ya + ta*ma + 3*tb*mb
    #   y_b = yb + ta*mb + tb*ma
    
    # Precompute all (ta, tb) pairs except (0,0)
    t_pairs = [(ta, tb) for ta in range(p) for tb in range(p) if not (ta == 0 and tb == 0)]
    
    # For speed, precompute affected indices for each candidate
    # But 2254 candidates * 2400 indices = 5.4M indices, ~43MB as int32. OK.
    
    # Actually, let's just iterate and check.
    
    added = 0
    for c_idx in candidates:
        if time.time() > _DEADLINE - 10:
            break
        
        ca, cb, cy_a, cy_b = _idx_to_point(c_idx)
        
        # Check if adding c would make any point have bad == 50
        # Affected points: for each direction, for each t != 0, x = c + t*v
        can_add = True
        
        # Direction (0,1): x = (ca, cb, cy_a+ta, cy_b+tb)
        for ta, tb in t_pairs:
            x_a = ca
            x_b = cb
            y_a = (cy_a + ta) % p
            y_b = (cy_b + tb) % p
            x_idx = _point_to_idx(x_a, x_b, y_a, y_b)
            if bad[x_idx] >= 49:
                can_add = False
                break
        
        if not can_add:
            continue
        
        # Directions (1, m) for ma, mb in 0..6
        for ma in range(p):
            for mb in range(p):
                for ta, tb in t_pairs:
                    x_a = (ca + ta) % p
                    x_b = (cb + tb) % p
                    y_a = (cy_a + ta * ma + 3 * tb * mb) % p
                    y_b = (cy_b + ta * mb + tb * ma) % p
                    x_idx = _point_to_idx(x_a, x_b, y_a, y_b)
                    if bad[x_idx] >= 49:
                        can_add = False
                        break
                if not can_add:
                    break
            if not can_add:
                break
        
        if can_add:
            # Add c to S
            in_S[c_idx] = 1
            added += 1
            # Update bad counts for affected points
            for ta, tb in t_pairs:
                x_a = ca
                x_b = cb
                y_a = (cy_a + ta) % p
                y_b = (cy_b + tb) % p
                x_idx = _point_to_idx(x_a, x_b, y_a, y_b)
                bad[x_idx] += 1
            
            for ma in range(p):
                for mb in range(p):
                    for ta, tb in t_pairs:
                        x_a = (ca + ta) % p
                        x_b = (cb + tb) % p
                        y_a = (cy_a + ta * ma + 3 * tb * mb) % p
                        y_b = (cy_b + ta * mb + tb * ma) % p
                        x_idx = _point_to_idx(x_a, x_b, y_a, y_b)
                        bad[x_idx] += 1
    
    # Build result: N = complement of S
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
