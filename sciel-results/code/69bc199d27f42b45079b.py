"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns the set of denominators for the Egyptian fraction expansion of 5/31
    that minimizes the largest denominator.
    Uses number-theoretic divisor enumeration for 3-term and 4-term solutions.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s guard for 90s limit
    
    # Target: 5/31
    # For 3-term: 1/a + 1/b + 1/c = 5/31 with a < b < c
    # Fix c (largest), then 1/a + 1/b = 5/31 - 1/c = (5c - 31)/(31c)
    # So (a+b)/(ab) = (5c-31)/(31c), meaning 31c(a+b) = ab(5c-31)
    # Rearranging: ab(5c-31) - 31ca - 31cb = 0
    # (a - 31c/(5c-31))(b - 31c/(5c-31)) = (31c/(5c-31))^2
    # Let D = 5c - 31. Need D > 0 so c >= 7.
    # 31c(a+b) = ab*D => ab*D = 31c(a+b)
    # For integer a,b: we can iterate a from 7 to c-1, compute required b.
    
    best_max_denom = 10000000
    best_solution = None
    
    # Try 3-term solutions: search c from 7 upward
    # For each c, we need 1/a + 1/b = (5c-31)/(31c) with 7 <= a < b < c
    # 1/a = (5c-31)/(31c) - 1/b = ((5c-31)b - 31c) / (31c*b)
    # So a = 31c*b / ((5c-31)*b - 31c)
    # For each c, iterate b from a_min to c-1 and check if a is integer.
    
    # Actually more efficient: for fixed c, iterate a from 7 to c-1
    # rem = 5/31 - 1/c = (5c-31)/(31c)
    # 1/b = rem - 1/a = (5c-31)/(31c) - 1/a = (a*(5c-31) - 31c) / (31c*a)
    # b = 31c*a / (a*(5c-31) - 31c)
    # Need denominator > 0 and divides numerator.
    
    for c in range(7, 1000):
        if time.time() > deadline:
            break
        D = 5 * c - 31  # must be > 0 for c >= 7
        if D <= 0:
            continue
        # rem = D / (31*c)
        # 1/a + 1/b = D/(31c)
        # b = 31c*a / (a*D - 31c)
        # Need a*D - 31c > 0, i.e., a > 31c/D
        # Also a < b < c, so a < c
        # And a >= 7
        a_min = max(7, (31 * c) // D + 1)
        if a_min >= c:
            continue
        for a in range(a_min, c):
            denom = a * D - 31 * c
            if denom <= 0:
                continue
            numer = 31 * c * a
            if numer % denom != 0:
                continue
            b = numer // denom
            if b <= a or b >= c:
                continue
            # Valid solution: [a, b, c]
            if c < best_max_denom:
                best_max_denom = c
                best_solution = [a, b, c]
                # If we found a small one, we can break early for this c
                # but keep searching for smaller c values
    
    # Try 4-term solutions if 3-term best is still large
    if best_max_denom > 50 and time.time() < deadline:
        # 1/a + 1/b + 1/c + 1/d = 5/31 with a < b < c < d
        # Fix d (largest), then 1/a + 1/b + 1/c = 5/31 - 1/d
        # For efficiency, fix d and c, then solve for a, b
        for d in range(7, 200):
            if time.time() > deadline:
                break
            if d >= best_max_denom:
                break
            D_d = 5 * d - 31
            if D_d <= 0:
                continue
            # rem_d = D_d / (31*d) = 1/a + 1/b + 1/c
            # Fix c: 1/a + 1/b = rem_d - 1/c = (D_d*c - 31d) / (31d*c)
            # Let D_c = D_d * c - 31*d
            # Need D_c > 0, so c > 31*d/D_d
            # Then b = 31*d*c*a / (a*D_c - 31*d*c)
            c_min = max(7, (31 * d) // D_d + 1)
            if c_min >= d:
                continue
            for c in range(c_min, d):
                D_c = D_d * c - 31 * d
                if D_c <= 0:
                    continue
                # 1/a + 1/b = D_c / (31*d*c)
                # b = 31*d*c*a / (a*D_c - 31*d*c)
                a_min = max(7, (31 * d * c) // D_c + 1)
                if a_min >= c:
                    continue
                for a in range(a_min, c):
                    denom = a * D_c - 31 * d * c
                    if denom <= 0:
                        continue
                    numer = 31 * d * c * a
                    if numer % denom != 0:
                        continue
                    b = numer // denom
                    if b <= a or b >= c:
                        continue
                    # Valid 4-term solution
                    if d < best_max_denom:
                        best_max_denom = d
                        best_solution = [a, b, c, d]
    
    # If no solution found, fall back
    if best_solution is None:
        best_solution = [8, 31, 248]
    
    return best_solution
# EVOLVE-BLOCK-END
