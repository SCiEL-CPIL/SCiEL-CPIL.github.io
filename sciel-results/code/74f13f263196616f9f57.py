# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Strategy: Find a valid Nikodym set by excluding a small set of points S
    such that for every point x in F_49^2, there exists a direction v where
    all other points on the line through x in direction v are in N = F_49^2 \ S.
    
    We use the unital-based construction:
    - Define a unital U (a curve in F_49^2)
    - Choose a set Y of "safe" y-coordinates
    - Exclude points in U whose y-coordinate is NOT in Y
    - Verify and optimize
    """
    start = time.time()
    deadline = start + 610  # safety margin before 630s

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    # Precompute field operations as lookup tables for speed
    # Represent field element as index 0..48: index = a + b*7 where elem = a + b*alpha
    
    # Build multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    add_table = np.zeros((q, q), dtype=np.int32)
    
    for a1 in range(p):
        for b1 in range(p):
            idx1 = a1 + b1 * p
            for a2 in range(p):
                for b2 in range(p):
                    idx2 = a2 + b2 * p
                    # Addition
                    add_table[idx1, idx2] = ((a1 + a2) % p) + ((b1 + b2) % p) * p
                    # Multiplication: (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + b1*b2*alpha^2 + (a1*b2 + a2*b1)*alpha
                    ra = (a1 * a2 + b1 * b2 * k) % p
                    rb = (a1 * b2 + a2 * b1) % p
                    mul_table[idx1, idx2] = ra + rb * p
    
    # Precompute: for each direction m (index), and each point (x,y) (as indices),
    # compute the line membership
    # Point index: pt_idx = x_idx * q + y_idx
    # Total points: q*q = 2401
    
    total_points = q * q
    
    # For a given excluded set (as boolean array of size total_points),
    # we need to check: for every point x, exists direction v such that
    # all points on line through x in dir v (except x) are NOT excluded.
    
    # Precompute for each direction m (0..48 for v=(1,m), plus 49 for v=(0,1)):
    # For each point x, the list of other points on the line
    # But that's 50 * 2401 * 48 entries... too many to precompute fully.
    
    # Instead, for a given excluded mask, check validity efficiently.
    
    def check_validity(excluded_mask):
        """
        excluded_mask: boolean array of shape (q, q) where True means excluded (not in N)
        Returns True if N = ~excluded_mask is a valid Nikodym set.
        """
        # For each point, check if there's a valid direction
        # A point x is "covered" by direction v if all other points on line are in N
        # i.e., no excluded points on the line except possibly x itself.
        
        # For direction v=(1, m): line through x=(x0,y0) is {(x0+t, y0+t*m) : t in F_49}
        # For direction v=(0,1): line through x=(x0,y0) is {(x0, y0+t) : t in F_49}
        
        # Precompute for each m, the line structure
        # For each m, we can group points into lines.
        
        # For efficiency, for each direction, check if any point is NOT covered.
        
        # Directions: m=0..48 for v=(1,m), and m=49 for v=(0,1)
        
        for m in range(q):  # v = (1, m)
            m_idx = m
            # For each point (x0, y0), the line contains points (x0+t, y0+t*m)
            # A point (x0,y0) is covered by this direction if 
            # sum of excluded on line (excluding self) == 0
            # i.e., line_sum - excluded[x0,y0] == 0
            
            # Compute line sums: for each line, sum of excluded values
            # Lines for direction (1,m): parameterized by intercept
            # y - m*x = c, where c is in F_49
            # c = y0 - m*x0
            
            line_sums = np.zeros(q, dtype=np.int32)  # indexed by c_idx
            # For each point, compute its line index c = y - m*x
            # c_idx = c as index
            
            for x_idx in range(q):
                x_a = x_idx // p
                x_b = x_idx % p
                for y_idx in range(q):
                    y_a = y_idx // p
                    y_b = y_idx % p
                    # c = y - m*x in F_49
                    # m*x: mul_table[m_idx, x_idx]
                    mx = mul_table[m_idx, x_idx]
                    # c = y - mx
                    # subtraction: y + (-mx)
                    # -mx = (p - mx_a) + (p - mx_b)*p if mx != 0
                    mx_a = mx // p
                    mx_b = mx % p
                    neg_mx_a = (-mx_a) % p
                    neg_mx_b = (-mx_b) % p
                    c_idx = ((y_a + neg_mx_a) % p) + ((y_b + neg_mx_b) % p) * p
                    line_sums[c_idx] += excluded_mask[x_idx, y_idx]
            
            # Now for each point, check if line_sum[c] - excluded[x,y] == 0
            # If yes, this direction covers the point.
            # We need: for ALL points, at least one direction covers them.
            
            # Instead, find points NOT covered by this direction:
            # points where line_sum[c] - excluded[x,y] > 0
            
            not_covered = np.zeros((q, q), dtype=bool)
            for x_idx in range(q):
                for y_idx in range(q):
                    x_a = x_idx // p
                    x_b = x_idx % p
                    y_a = y_idx // p
                    y_b = y_idx % p
                    mx = mul_table[m_idx, x_idx]
                    mx_a = mx // p
                    mx_b = mx % p
                    neg_mx_a = (-mx_a) % p
                    neg_mx_b = (-mx_b) % p
                    c_idx = ((y_a + neg_mx_a) % p) + ((y_b + neg_mx_b) % p) * p
                    if line_sums[c_idx] - excluded_mask[x_idx, y_idx] > 0:
                        not_covered[x_idx, y_idx] = True
            
            # Points covered by this direction: ~not_covered
            # If all points are covered, we're done
            if not not_covered.any():
                return True
            
            # Otherwise, continue to next direction
            # But we need to track which points are still uncovered
            # Let's restructure: maintain a set of uncovered points
            
            # Actually, let's do it differently: for each point, check all directions
            # This is O(q * q * q) = 49^3 ≈ 117k which is fine
            
            # Let's just do the full check per point
            break  # We'll do per-point check below
        
        # Per-point check: for each point, try all directions
        for x_idx in range(q):
            x_a = x_idx // p
            x_b = x_idx % p
            for y_idx in range(q):
                y_a = y_idx // p
                y_b = y_idx % p
                
                found_dir = False
                
                # Try v = (1, m) for m in 0..48
                for m_idx in range(q):
                    # Check all t != 0: point (x + t, y + t*m) must be in N (not excluded)
                    valid = True
                    for t_idx in range(q):
                        if t_idx == 0:
                            continue
                        # new_x = x + t
                        nx_idx = add_table[x_idx, t_idx]
                        # new_y = y + t*m
                        tm_idx = mul_table[t_idx, m_idx]
                        ny_idx = add_table[y_idx, tm_idx]
                        if excluded_mask[nx_idx, ny_idx]:
                            valid = False
                            break
                    if valid:
                        found_dir = True
                        break
                
                # Try v = (0, 1)
                if not found_dir:
                    valid = True
                    for t_idx in range(q):
                        if t_idx == 0:
                            continue
                        # new_x = x + 0 = x
                        # new_y = y + t
                        ny_idx = add_table[y_idx, t_idx]
                        if excluded_mask[x_idx, ny_idx]:
                            valid = False
                            break
                    if valid:
                        found_dir = True
                
                if not found_dir:
                    return False
        
        return True
    
    # Now, search for a good excluded set.
    # The unital construction: U = {(x,y) : Tr(x) + N(y) = 0}
    # Exclude points in U where y not in Y.
    
    # Tr(x) = 2*x_a mod p
    # N(y) = y_a^2 - y_b^2 * k mod p
    
    # Let's try different Y sets to minimize |S|
    
    best_excluded = None
    best_size = q * q + 1
    
    # Try various Y sets
    # Y can be any subset of F_49 (represented as boolean array of size q)
    
    # Start with the original: Y = {(u^2 mod p, v) : u,v in F_7}
    # That's 4*7 = 28 elements (since u^2 mod 7: 0,1,2,4)
    
    # Actually, let's be smarter. The size of S is:
    # |S| = |{(x,y) in U : y not in Y}|
    # = |U| - |{(x,y) in U : y in Y}|
    
    # |U| = q (it's a unital)
    # We want to maximize |{(x,y) in U : y in Y}|
    
    # For each y in F_49, how many x satisfy Tr(x) + N(y) = 0?
    # Tr(x) = 2*x_a, so 2*x_a = -N(y) mod p
    # x_a = -N(y)/2 mod p (unique since 2 != 0 mod 7)
    # x_b is free: p choices
    # So for each y, there are p points in U with that y-coordinate.
    
    # Thus |{(x,y) in U : y in Y}| = p * |Y|
    # |S| = q - p*|Y|
    
    # To minimize |S|, maximize |Y|, so Y = F_49, giving |S| = 0.
    # But that gives N = F_49^2 which is trivially valid but not minimal.
    
    # Wait, we want to MINIMIZE |N| = maximize |S|? No!
    # The metric is 2401 - |N| = |S|, and higher is better.
    # So we want to MAXIMIZE |S| while keeping N valid.
    
    # So we want to exclude as many points as possible!
    
    # With the unital construction, |S| = q - p*|Y|.
    # To maximize |S|, minimize |Y|.
    # But N must be valid.
    
    # If Y is empty, S = U, |S| = 49.
    # N = F_49^2 \ U.
    # Is this valid? Let's check.
    
    # Actually, let's try different constructions.
    
    # Alternative: exclude points on multiple lines or curves.
    
    # Let's try: for each slope m, exclude one point per line of slope m.
    # But that might not work.
    
    # Let's just try the unital with small Y and verify.
    
    # Try Y of various sizes
    for y_size in range(0, q + 1):
        if time.time() > deadline - 30:
            break
        
        # Generate a specific Y set of size y_size
        # Use first y_size elements
        Y_mask = np.zeros(q, dtype=bool)
        Y_mask[:y_size] = True
        
        # Compute excluded set
        excluded = np.zeros((q, q), dtype=bool)
        for x_idx in range(q):
            x_a = x_idx // p
            x_b = x_idx % p
            tr_x = (2 * x_a) % p
            for y_idx in range(q):
                y_a = y_idx // p
                y_b = y_idx % p
                norm_y = (y_a * y_a - y_b * y_b * k) % p
                if (tr_x + norm_y) % p == 0 and not Y_mask[y_idx]:
                    excluded[x_idx, y_idx] = True
        
        s_size = excluded.sum()
        if s_size <= best_size:
            continue  # We want larger S
        
        # Check validity
        if check_validity(excluded):
            best_excluded = excluded.copy()
            best_size = s_size
            print(f"Found valid set with |S| = {s_size}, |N| = {total_points - s_size}")
    
    # If no valid set found, use the original construction
    if best_excluded is None:
        # Original: Y = {(u^2 mod p, v)}
        Y_mask = np.zeros(q, dtype=bool)
        for u in range(p):
            for v in range(p):
                y_a = (u * u) % p
                y_b = v
                y_idx = y_a + y_b * p
                Y_mask[y_idx] = True
        
        excluded = np.zeros((q, q), dtype=bool)
        for x_idx in range(q):
            x_a = x_idx // p
            x_b = x_idx % p
            tr_x = (2 * x_a) % p
            for y_idx in range(q):
                y_a = y_idx // p
                y_b = y_idx % p
                norm_y = (y_a * y_a - y_b * y_b * k) % p
                if (tr_x + norm_y) % p == 0 and not Y_mask[y_idx]:
                    excluded[x_idx, y_idx] = True
        
        best_excluded = excluded
        best_size = excluded.sum()
    
    # Build the point list
    points = []
    for x_idx in range(q):
        for y_idx in range(q):
            if not best_excluded[x_idx, y_idx]:
                x_a = x_idx // p
                x_b = x_idx % p
                y_a = y_idx // p
                y_b = y_idx % p
                points.append([[x_a, x_b], [y_a, y_b]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
