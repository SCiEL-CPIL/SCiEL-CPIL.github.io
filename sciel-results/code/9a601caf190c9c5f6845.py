# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    
    Strategy: Use a Hermitian unital U, then find a minimal blocking set B within U
    such that every point outside U can be covered by a line through some b in B
    that intersects U only at b. Then N = F_49^2 \\ (U \\ B).
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    p = 7
    q = p * p  # 49
    
    # Find smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    
    # Generate all points in F_49^2
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append((xa, xb, ya, yb))
    
    # Point is in Hermitian unital U: 2*xa + ya^2 - yb^2*k ≡ 0 (mod p)
    def in_unital(xa, xb, ya, yb):
        return (2 * xa + ya * ya - yb * yb * k) % p == 0
    
    # Find all points in U
    U_points = []
    for pt in all_points:
        if in_unital(pt[0], pt[1], pt[2], pt[3]):
            U_points.append(pt)
    
    # For a blocking set B ⊆ U, N = F_49^2 \\ (U \ B)
    # N must satisfy: for every x in F_49^2, there's a line through x
    # such that all other points on that line are in N.
    
    # A point x is covered by B if there exists b in B such that the line
    # through x and b intersects U only at b (or x is in B itself).
    
    # Key insight: For x not in U, we need a line through x that avoids U \ B.
    # This means the line should intersect U only at points in B.
    
    # For efficiency, precompute for each pair of points whether the line through them
    # intersects U in only 1 point (the second point), assuming first is in U.
    
    # Actually, let's think differently. For a given B ⊆ U:
    # - Points in B are covered (trivially)
    # - Points in U \ B: need a line through them where all other U-points are in B
    # - Points not in U: need a line through them where all U-points on line are in B
    
    # For the seed's approach, B was large. Let's try to find a smaller B.
    
    # Precompute: for each point b in U, find all lines through b
    # and count how many U-points each line contains (excluding b)
    
    # Direction v = (1, m) where m = (ma, mb), or v = (0,1)
    # Line through b=(xa,xb,ya,yb) with direction v=(va,vb):
    # points are b + t*v for t in F_49
    
    # For vertical lines (v=(0,1)): line is {(xa,xb, ya+t, yb) : t in F_p}
    # Wait, we're in F_q, so t has q values.
    
    # Let me represent F_49 elements as pairs (a,b) for a+b*alpha
    # Addition: (a1,b1) + (a2,b2) = (a1+a2, b1+b2) mod p
    # Multiplication: (a1,b1)*(a2,b2) = (a1*a2 + b1*b2*k, a1*b2 + a2*b1) mod p
    
    def f49_add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)
    
    def f49_mul(x, y):
        return ((x[0]*y[0] + x[1]*y[1]*k) % p, (x[0]*y[1] + x[1]*y[0]) % p)
    
    def f49_sub(x, y):
        return ((x[0] - y[0]) % p, (x[1] - y[1]) % p)
    
    # A line through point (x,y) with direction (1,m) or (0,1)
    # For direction (1,m): points are ((x + t, y + t*m) for t in F_49
    # For direction (0,1): points are ((x, y + t) for t in F_49
    
    # Precompute all lines and their intersection with U
    # There are q + 1 = 50 directions, and for each direction, q lines = 49*50 = 2450 lines
    # Each line has q = 49 points
    
    # For each line, store the set of U-points on it
    
    # Directions: m in F_49 (49 directions) plus vertical
    # Represent direction as (dx, dy) where for non-vertical: (1, m), vertical: (0, 1)
    
    directions = []
    # Non-vertical directions: (1, m) for m in F_49
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (dx, dy)
    # Vertical direction
    directions.append(((0, 1), (0, 0)))
    
    # For each direction, there are q lines (offset by y-intercept for non-vertical, x for vertical)
    # Actually, for direction (dx, dy), lines are parameterized by a starting point.
    # To avoid duplicates, fix the "first" point in some canonical way.
    
    # Simpler: for each direction, iterate over all possible lines by choosing
    # a representative point on the line.
    
    # For non-vertical (1, m): line determined by its intersection with x=0, i.e., point (0, y0)
    # For vertical (0,1): line determined by its x-coordinate, i.e., line x = x0
    
    # Let's enumerate all lines properly.
    
    # Precompute: for each line, the list of U-points on it
    # And for each point in U, the list of lines through it and how many U-points each has
    
    # This is complex. Let me use a different approach.
    
    # Given time constraints, let's try a specific B structure:
    # Take B to be a set of points in U such that for every line through a point
    # in U\B, the line intersects U in only that one point OR all other U-intersections
    # are in B.
    
    # Simplest valid B: the entire U. Then S = empty, N = F_49^2, complement = 0. Bad.
    
    # Better: find minimal B such that coverage holds.
    
    # Let's try B as a line section of U. U is a Hermitian unital, which has
    # p+1 = 8 points on each line that intersects it.
    
    # Actually, a Hermitian unital in F_q^2 has q+1 = 50 points total (for q=49).
    # Wait, let me recount.
    
    # |U| = number of points satisfying 2*xa + ya^2 - yb^2*k ≡ 0
    # For each (ya, yb), xa is determined: xa = (ya^2 - yb^2*k)/2 mod p
    # There are p^2 = 49 possible (ya, yb), so |U| = 49? No wait...
    
    # xa is determined by (ya, yb), so for each of the 49 pairs (ya, yb), there's
    # exactly one xa. But xb is free (7 choices). So |U| = 49 * 7 = 343.
    
    # That's a lot. The unital has 343 points.
    
    # The seed excluded S = U \ B where B was points in U with y in Y (y_a is a square).
    # Y has p * p/2 ≈ 7*4 = 28 points on average? No, Y = {(u^2, v) : u,v in F_p}
    # |Y| = p * p = 49? No, u^2 takes (p+1)/2 = 4 values, so |Y| = 4 * 7 = 28.
    
    # So |B| = |U ∩ (F_p × Y)|. For each (ya, yb) in Y, there are 7 points in U.
    # |B| ≈ 28 * 7 = 196. |S| = 343 - 196 = 147. Complement = 147.
    
    # To improve, we need larger S, i.e., smaller B.
    
    # Let's try: B = points in U with ya = 0 (a single "fiber").
    # |B| = 7 (one ya, 7 xb values, ya=0, yb arbitrary? No...)
    
    # If ya=0: 2*xa + 0 - yb^2*k ≡ 0 => xa = yb^2*k/2
    # For each yb (7 values), xa is determined. xb is free (7 values).
    # So |B| = 7 * 7 = 49 if ya=0.
    
    # Is this B sufficient? We need to check if every point is covered.
    
    # Given complexity, let me try a computational approach with a smaller B.
    
    # Try B = {points in U with ya = 0}
    B_candidates = []
    for pt in U_points:
        if pt[2] == 0:  # ya = 0
            B_candidates.append(pt)
    
    # Check if this B works
    def check_nikodym(B, U_set, all_pts):
        """Check if N = F_q^2 \\ (U \ B) is a valid Nikodym set."""
        B_set = set(B)
        U_set = set(U_set)
        N_set = set(all_pts) - (U_set - B_set)
        
        # For each point x in F_q^2, check if there's a direction such that
        # all other points on the line through x in that direction are in N
        # Directions: (1, m) for m in F_49, and (0, 1)
        
        for x in all_pts:
            xa, xb, ya, yb = x
            x_coord = (xa, xb)
            y_coord = (ya, yb)
            
            covered = False
            # Try non-vertical directions (1, m)
            for ma in range(p):
                for mb in range(p):
                    m = (ma, mb)
                    # Line: x + t*(1, m) for t in F_49
                    # Points: (x_coord + t, y_coord + t*m)
                    all_in_N = True
                    for ta in range(p):
                        for tb in range(p):
                            if ta == 0 and tb == 0:
                                continue  # Skip x itself
                            t = (ta, tb)
                            tx = f49_add(x_coord, t)
                            ty = f49_add(y_coord, f49_mul(t, m))
                            pt = (tx[0], tx[1], ty[0], ty[1])
                            if pt not in N_set:
                                all_in_N = False
                                break
                        if not all_in_N:
                            break
                    if all_in_N:
                        covered = True
                        break
                if covered:
                    break
            
            if not covered:
                # Try vertical direction (0, 1)
                all_in_N = True
                for ta in range(p):
                    for tb in range(p):
                        if ta == 0 and tb == 0:
                            continue
                        t = (ta, tb)
                        tx = x_coord  # x unchanged
                        ty = f49_add(y_coord, t)
                        pt = (tx[0], tx[1], ty[0], ty[1])
                        if pt not in N_set:
                            all_in_N = False
                            break
                    if not all_in_N:
                        break
                if all_in_N:
                    covered = True
            
            if not covered:
                return False, x
        
        return True, None
    
    # The check above is O(2401 * 50 * 49) ≈ 5.9M operations, feasible
    # But N_set membership is O(1) with set, so it should be fine.
    
    # However, this is for one B. Let's try the B with ya=0 first.
    
    U_set = set(U_points)
    B_set = set(B_candidates)
    N_set = set(all_points) - (U_set - B_set)
    
    # Quick check: is N non-empty?
    if len(N_set) == 0:
        # Fallback: return seed's construction
        pass
    
    # Let's just return the seed's construction for safety, but try to improve.
    # Actually, let me try a few B candidates and pick the best valid one.
    
    best_complement = 0
    best_N = None
    
    # Candidate 1: ya = 0
    B1 = [pt for pt in U_points if pt[2] == 0]
    
    # Candidate 2: ya = 0 and yb = 0 (single point? No, xa is determined, xb free)
    # ya=0, yb=0: xa = 0, so points are (0, xb, 0, 0) for xb in 0..6. |B| = 7
    
    B2 = [pt for pt in U_points if pt[2] == 0 and pt[3] == 0]
    
    # Candidate 3: ya in {0,1}
    B3 = [pt for pt in U_points if pt[2] in (0, 1)]
    
    candidates = [B1, B2, B3]
    
    for B in candidates:
        if not B:
            continue
        valid, _ = check_nikodym(B, U_points, all_points)
        if valid:
            complement = len(U_points) - len(B)
            if complement > best_complement:
                best_complement = complement
                best_N = set(all_points) - (set(U_points) - set(B))
    
    # If none valid, use seed's construction
    if best_N is None:
        # Seed's B: y in Y where Y = {(u^2, v)}
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))
        
        B_seed = [pt for pt in U_points if (pt[2], pt[3]) in y_special]
        best_N = set(all_points) - (set(U_points) - set(B_seed))
    
    # Convert to array
    points = list(best_N)
    return np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in points], dtype=np.int64)


# EVOLVE-BLOCK-END
