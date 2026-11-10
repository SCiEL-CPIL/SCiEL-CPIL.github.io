# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using the Hermitian unital approach.
    
    The set N is the complement of S, where S is a subset of the Hermitian unital U.
    We aim to make S as large as possible (minimize |N|) while ensuring that for
    every point x, there exists a line through x (in directions (1,m) or (0,1))
    such that the other q-1 points on the line are in N.
    
    The baseline valid construction uses S = U \\ B, where B is the set of points
    in U with y-coordinate in a specific set Y. We try to shrink B (and thus grow
    S) by using a smaller Y, then validate the resulting Nikodym set.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    p = 7
    q = p * p
    # alpha^2 = 3, since 3 is the smallest quadratic non-residue mod 7
    k = 3
    
    def field_eq(a, b, c, d):
        """Check if a + b*alpha == c + d*alpha"""
        return (a % p) == (c % p) and (b % p) == (d % p)
    
    def field_sub(a, b, c, d):
        """Return (a,b) - (c,d)"""
        return ((a - c) % p, (b - d) % p)
    
    def field_mul(a, b, c, d):
        """Return (a + b*alpha) * (c + d*alpha)"""
        # (ac + bd*k, ad + bc)
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)
    
    # Hermitian unital: U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace(x) = 2 * xa
    # Norm(y) = ya^2 - yb^2 * k
    # So: 2*xa + ya^2 - yb^2 * k == 0 (mod p)
    
    # For each x, there is exactly one y in U (since yb is determined by ya)
    # Actually, let's list all points in U
    U_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                # 2*xa + ya^2 - yb^2 * k == 0 => yb^2 = (2*xa + ya^2) / k
                rhs = (2 * xa + ya * ya) % p
                # Solve yb^2 * k == rhs => yb^2 == rhs * k^{-1}
                # k = 3, k^{-1} mod 7 = 5 (since 3*5=15=1 mod 7)
                target_sq = (rhs * 5) % p
                # Find yb such that yb^2 == target_sq mod p
                for yb in range(p):
                    if (yb * yb) % p == target_sq:
                        U_points.append((xa, xb, ya, yb))
    
    # Total points in F_49^2
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    
    # Validate Nikodym set
    # N is a Nikodym set if for every point x in F_49^2, there exists a direction
    # v = (1, m) for some m in F_49, or v = (0, 1), such that the line
    # {x + t*v : t in F_49} has all points except x in N.
    
    def validate_nikodym(N_set, timeout=60.0):
        """
        N_set: set of points in N (each point is (xa, xb, ya, yb))
        Returns True if N is a Nikodym set, False otherwise.
        """
        t_start = time.time()
        
        for x in all_points:
            xa, xb, ya, yb = x
            if x in N_set:
                continue  # We need to find a line through x where other points are in N
            
            found_valid_line = False
            
            # Try direction (0, 1)
            # Line: {(xa, xb, ya + t, yb) : t in F_7}
            # Other points: t = 1, 2, ..., 6
            ok = True
            for t in range(1, p):
                pt = (xa, xb, (ya + t) % p, yb)
                if pt not in N_set:
                    ok = False
                    break
            if ok:
                found_valid_line = True
            
            # Try directions (1, m) for m in F_49
            if not found_valid_line:
                for ma in range(p):
                    for mb in range(p):
                        # Direction v = (1, m) where m = ma + mb*alpha
                        # Line: {x + t*v : t in F_49}
                        # x + t*v = (xa + t*1, xb + t*0, ya + t*ma, yb + t*mb)
                        # Wait, t is in F_49, so t = ta + tb*alpha
                        # x + t*v:
                        #   x_coord = xa + t*1 = xa + t
                        #   y_coord = ya + t*m
                        # So the line is parameterized by t in F_49.
                        # The other q-1 points are for t != 0.
                        
                        ok = True
                        for ta in range(p):
                            for tb in range(p):
                                if ta == 0 and tb == 0:
                                    continue
                                # t = ta + tb*alpha
                                # x_coord = xa + t = (xa + ta, tb)
                                # y_coord = ya + t*m
                                # t*m = (ta + tb*alpha)(ma + mb*alpha)
                                #     = (ta*ma + tb*mb*k, ta*mb + tb*ma)
                                y_x = (ya + ta*ma + tb*mb*k) % p
                                y_b = (yb + ta*mb + tb*ma) % p
                                pt = ((xa + ta) % p, tb, y_x, y_b)
                                if pt not in N_set:
                                    ok = False
                                    break
                                if time.time() > t_start + timeout:
                                    return False
                            if not ok:
                                break
                        if ok:
                            found_valid_line = True
                            break
                    if found_valid_line:
                        break
            
            if not found_valid_line:
                return False
            
            if time.time() > start_time + 550.0:
                return False
        
        return True
    
    # Strategy: Try different Y sets (y-coordinates for B)
    # B = {points in U with y in Y}
    # S = U \ B
    # N = F_49^2 \ S
    
    # Candidate Y sets, from small to large (to maximize S = U \ B)
    # Y is a set of y-values (ya, yb)
    
    # Smallest possible: empty Y (B = empty, S = U)
    # Then N = F_49^2 \ U. But this is likely not a Nikodym set.
    
    # Let's try Y = {(ya, yb) : ya in {0}}
    y_candidates = []
    
    # Single y-value
    for ya in range(p):
        for yb in range(p):
            y_candidates.append({(ya, yb)})
    
    # Two y-values with same ya
    for ya in range(p):
        y_set = set()
        for yb in range(p):
            y_set.add((ya, yb))
        y_candidates.append(y_set)
    
    # Original: all squares
    y_orig = set()
    for u in range(p):
        for v in range(p):
            y_orig.add(((u * u) % p, v))
    y_candidates.append(y_orig)
    
    # Try candidates
    best_result = None
    
    for y_set in y_candidates:
        if time.time() > start_time + 500.0:
            break
        
        # Build B = {points in U with y in y_set}
        B_set = set()
        for pt in U_points:
            xa, xb, ya, yb = pt
            if (ya, yb) in y_set:
                B_set.add(pt)
        
        # S = U \ B
        S_set = set(U_points) - B_set
        
        # N = F_49^2 \ S
        N_set = set(all_points) - S_set
        
        # Check validity
        if validate_nikodym(N_set):
            # Convert to array
            points = np.array(list(N_set), dtype=np.int64).reshape(-1, 2, 2)
            if best_result is None or len(points) < len(best_result):
                best_result = points
            if time.time() > start_time + 580.0:
                break
    
    if best_result is None:
        # Fallback to the known valid construction
        y_set = y_orig
        B_set = set()
        for pt in U_points:
            xa, xb, ya, yb = pt
            if (ya, yb) in y_set:
                B_set.add(pt)
        S_set = set(U_points) - B_set
        N_set = set(all_points) - S_set
        best_result = np.array(list(N_set), dtype=np.int64).reshape(-1, 2, 2)
    
    return best_result


# EVOLVE-BLOCK-END
