import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s kill

    p = 7
    k = 3  # alpha^2 = 3, non-residue mod 7

    # Try to find the largest valid complement by trying different blocking sets
    # The complement S = U \ B where U is the Hermitian unital
    # We want to minimize |B| while maintaining Nikodym property

    # Precompute: for each point, determine if it's in the unital
    # U = {(xa, xb, ya, yb) : 2*xa + ya^2 - k*yb^2 = 0 mod p}

    # Build all points in the unital
    unital_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        unital_points.append((xa, xb, ya, yb))

    # For a given blocking set Y (subset of F_49 for y-coordinates),
    # S = {points in U where y not in Y}
    # N = F_49^2 \ S
    # We need to verify Nikodym property

    def verify_nikodym(excluded_set):
        """Check if F_49^2 \ excluded_set is a valid Nikodym set."""
        if time.time() - start_time > deadline:
            return False, None

        excluded = set(excluded_set)
        all_points = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        all_points.add((xa, xb, ya, yb))

        N = all_points - excluded

        # Field operations
        def f_add(e1, e2):
            return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

        def f_mul(e1, e2):
            a, b = e1
            c, d = e2
            return ((a * c + b * d * k) % p, (a * d + b * c) % p)

        def f_sub(e1, e2):
            return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)

        def f_inv(e):
            a, b = e
            norm = (a * a - b * b * k) % p
            if norm == 0:
                raise ValueError("zero")
            norm_inv = pow(norm, p - 2, p)
            return ((a * norm_inv) % p, (-b * norm_inv) % p)

        def f_div(e1, e2):
            return f_mul(e1, f_inv(e2))

        # All possible directions: v = (1, m) for m in F_49, or v = (0, 1)
        # For each point x, check if there exists a direction v such that
        # all points x + t*v (t != 0) are in N

        # Build lookup for N
        N_set = set(N)

        # Check each point
        for x in all_points:
            xa, xb, ya, yb = x
            found = False

            # Try direction (0, 1): line is {(xa,xb), (xa,xb), (xa,xb+y)} for y in F_49
            # Points: x + t*(0,1) = (xa, xb + t) for t in F_49
            # t != 0 means all other 48 points
            ok = True
            for tb in range(p):
                for ta in range(p):
                    t = (ta, tb)
                    if t == (0, 0):
                        continue
                    # x + t*(0,1) = (xa, xb) + (0, t) = (xa, xb + t)
                    # (0,1) as vector: first coord is 0, second is 1
                    # x + t*v where v=(0,1): (xa + t*0, xb + t*1) = (xa, xb + t)
                    new_xa = xa
                    new_xb = (xb + tb) % p  # t = ta + tb*alpha, but v=(0,1) so t*v = (0, t)
                    # Wait, t*v where v = (0,1): (t*0, t*1) = (0, t)
                    # So x + t*v = (xa + 0, xb + t) = (xa, xb + t)
                    # t is a field element, xb + t means field addition
                    new_xb_pair = f_add((xb, 0), (ta, tb))  # Hmm, this isn't right
                    # Actually xb is a field element (xb_a, xb_b)... no, xb is just an integer 0..6
                    # The point is (x, y) where x = (xa, xb) in F_49, y = (ya, yb) in F_49
                    # Direction v = (0, 1) means v_x = 0 in F_49, v_y = 1 in F_49
                    # x + t*v = (x_x + t*v_x, x_y + t*v_y) = ((xa,xb) + t*(0,0), (ya,yb) + t*(1,0))
                    # = ((xa,xb), (ya,yb) + t*(1,0))
                    # t*(1,0) = t*1 = t = (ta, tb)
                    # So new point is (xa, xb, ya+ta, yb+tb)
                    ny_a = (ya + ta) % p
                    ny_b = (yb + tb) % p
                    if (xa, xb, ny_a, ny_b) not in N_set:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                found = True
            else:
                # Try directions (1, m) for m in F_49
                for ma in range(p):
                    for mb in range(p):
                        if time.time() - start_time > deadline:
                            return False, None
                        # v = (1, m) where m = (ma, mb)
                        # x + t*v = (x_x + t*1, x_y + t*m) = ((xa,xb) + t, (ya,yb) + t*m)
                        ok = True
                        for ta in range(p):
                            for tb in range(p):
                                t = (ta, tb)
                                if t == (0, 0):
                                    continue
                                # new_x = (xa,xb) + t = (xa+ta, xb+tb)
                                nx_a = (xa + ta) % p
                                nx_b = (xb + tb) % p
                                # new_y = (ya,yb) + t*m
                                tm = f_mul(t, (ma, mb))
                                ny_a = (ya + tm[0]) % p
                                ny_b = (yb + tm[1]) % p
                                if (nx_a, nx_b, ny_a, ny_b) not in N_set:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            found = True
                            break
                    if found:
                        break
            if not found:
                return False, None

        return True, None

    # Strategy: try different blocking sets Y (subsets of F_49 for y-coord)
    # Start with the known working one, then try to expand the complement

    # Known working: Y = {(u^2 mod 7, v) for u,v in F_7} -> y_a is a square
    # This gives |S| = 147

    # Try: can we use a smaller Y?
    # Y must contain at least the y-values needed for the blocking set property

    # Let's try Y = {(0, v) for v in F_7} (y_a = 0 only)
    # This gives |B| = 343/7 = 49, |S| = 294

    # But we need to verify! Let's try a few candidates and verify

    candidates = []

    # Candidate 1: y_a is a square (known working)
    y_sq = set()
    for u in range(p):
        for v in range(p):
            y_sq.add((u * u % p, v))
    candidates.append(y_sq)

    # Candidate 2: y_a = 0
    y_zero = set()
    for v in range(p):
        y_zero.add((0, v))
    candidates.append(y_zero)

    # Candidate 3: y_a is a square OR y_a = 0 (same as squares since 0 is a square)
    # Already covered

    # Candidate 4: y in {(0,v), (1,v)} 
    y_01 = set()
    for v in range(p):
        y_01.add((0, v))
        y_01.add((1, v))
    candidates.append(y_01)

    # Candidate 5: y_a is a cube (all elements since F_7^* is cyclic of order 6, cubes are {1,6})
    # Cubes mod 7: 0^3=0, 1^3=1, 2^3=8=1, 3^3=27=6, 4^3=64=1, 5^3=125=6, 6^3=216=6
    # So cubes are {0, 1, 6}
    y_cube = set()
    for u in range(p):
        for v in range(p):
            if (u * u * u) % p in {0, 1, 6}:
                y_cube.add((u * u * u % p, v))
    candidates.append(y_cube)

    # For each candidate, compute the excluded set and verify
    best_complement_size = 0
    best_excluded = None

    for Y in candidates:
        if time.time() - start_time > deadline:
            break

        # Compute excluded set S = {points in U where y not in Y}
        excluded = set()
        for (xa, xb, ya, yb) in unital_points:
            if (ya, yb) not in Y:
                excluded.add((xa, xb, ya, yb))

        comp_size = len(excluded)
        if comp_size <= best_complement_size:
            continue

        # Verify Nikodym property
        valid, _ = verify_nikodym(excluded)
        if valid:
            best_complement_size = comp_size
            best_excluded = excluded
            print(f"Found valid complement of size {comp_size}")

    # If no new valid set found, use the known working one
    if best_excluded is None:
        Y = y_sq
        best_excluded = set()
        for (xa, xb, ya, yb) in unital_points:
            if (ya, yb) not in Y:
                best_excluded.add((xa, xb, ya, yb))

    # Build the Nikodym set N = F_49^2 \ best_excluded
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in best_excluded:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
