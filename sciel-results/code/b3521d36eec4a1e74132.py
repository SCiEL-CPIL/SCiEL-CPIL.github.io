# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    # alpha^2 = 3
    k = 3
    
    # Precompute field operations if needed, but we work with pairs (a,b)
    # Point is [[xa, xb], [ya, yb]]
    
    # Generate all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append((xa, xb, ya, yb))
    
    # Define Unital U: 2*xa + ya^2 - 3*yb^2 == 0 mod 7
    def in_unital(pt):
        xa, xb, ya, yb = pt
        return (2 * xa + ya * ya - 3 * yb * yb) % p == 0

    U = [pt for pt in all_points if in_unital(pt)]
    
    # Baseline N: Exclude U points with ya not in {0,1,2,4}
    y_squares = {0, 1, 2, 4}
    S_baseline = set()
    for pt in U:
        xa, xb, ya, yb = pt
        if ya not in y_squares:
            S_baseline.add(pt)
    
    N_baseline = [pt for pt in all_points if pt not in S_baseline]
    
    # Try to exclude all U
    S_full = set(U)
    N_full = [pt for pt in all_points if pt not in S_full]
    
    # Verifier
    def is_valid_nikodym(N_list):
        N_set = set(N_list)
        # Precompute lines?
        # For each point x, check if there is a line L through x such that L\{x} subset N_set
        
        # Directions: (1, m) for m in F_q, and (0, 1)
        # F_q elements are pairs (a,b)
        field_elems = [(a, b) for a in range(p) for b in range(p)]
        
        # Helper to add points
        def add_pt(p1, p2):
            return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)
        
        def mul_pt(p1, p2):
            # (a+b alpha)(c+d alpha) = ac + bd alpha^2 + (ad+bc) alpha
            a, b = p1
            c, d = p2
            return ((a*c + b*d*k) % p, (a*d + b*c) % p)
        
        def sub_pt(p1, p2):
            return ((p1[0] - p2[0]) % p, (p1[1] - p2[1]) % p)
            
        # Check each point
        for x in all_points:
            xa, xb, ya, yb = x
            found_witness = False
            
            # Check direction (0, 1) -> vertical line x = const
            # Points: (xa, xb + t_b, ya, yb + t_a) ? No, direction (0,1) means (dx, dy) = (0, 1).
            # In field coords: v = (0, 1).
            # x + t*v = (xa + t*0, xb + t*0, ya + t*1, yb + t*1) ?
            # Wait, point is (x, y) in F_q^2.
            # x = (xa, xb), y = (ya, yb).
            # v = (vx, vy).
            # Line: x + t*v = (x + t*vx, y + t*vy).
            
            # Direction 1: v = (0, 1) in F_q^2.
            # vx = (0,0), vy = (1,0) ? No, (0,1) usually means (0, 1) in F_p^2?
            # The task says v = (1, m) or v = (0, 1).
            # (0, 1) likely means vx = 0, vy = 1 (the element 1 in F_q).
            
            # Let's iterate over all directions v
            # v can be (vx_a, vx_b, vy_a, vy_b)
            # v != 0.
            
            # To speed up, we can just check if there exists ANY line.
            # But 2401 points * 2450 lines is too slow.
            
            # Optimization:
            # For x in N, we need a line L with L\{x} subset N.
            # This is equivalent to: there exists a direction v such that for all t != 0, x + t*v in N.
            
            # We will check this for N_full.
            
            # Precompute N_set for O(1) lookup
            # N_set is already a set of tuples
            
            # Directions:
            # v = (1, m) for m in F_q (49 directions)
            # v = (0, 1) (1 direction)
            # Total 50 directions.
            
            # For each direction, check the line.
            
            # m in F_q
            for ma in range(p):
                for mb in range(p):
                    # v = (1, m)
                    # vx = (1, 0)
                    # vy = (ma, mb)
                    vx = (1, 0)
                    vy = (ma, mb)
                    
                    valid = True
                    for t in range(1, q):
                        # t in F_q
                        ta = t // p
                        tb = t % p
                        
                        # x + t*v
                        # x_x + t*vx_x
                        # x_y + t*vy_y
                        
                        # x_x = (xa, xb)
                        # t*vx_x = t*(1,0) = (t, 0)
                        # sum_x = (xa + t, xb)
                        
                        # x_y = (ya, yb)
                        # t*vy_y = t*(ma, mb)
                        # t*vy = (ta, tb) * (ma, mb)
                        # = (ta*ma + tb*mb*k, ta*mb + tb*ma)
                        
                        tvx = (t, 0)
                        tvy_a = (ta * ma + tb * mb * k) % p
                        tvy_b = (ta * mb + tb * ma) % p
                        
                        nx_a = (xa + tvx[0]) % p
                        nx_b = (xb + tvx[1]) % p
                        ny_a = (ya + tvy_a) % p
                        ny_b = (yb + tvy_b) % p
                        
                        npt = (nx_a, nx_b, ny_a, ny_b)
                        if npt not in N_set:
                            valid = False
                            break
                    
                    if valid:
                        found_witness = True
                        break
                
                if found_witness:
                    break
            
            if found_witness:
                continue
                
            # Check v = (0, 1)
            # vx = (0, 0)
            # vy = (1, 0)
            valid = True
            for t in range(1, q):
                ta = t // p
                tb = t % p
                
                # t*vy = t*(1,0) = (t, 0)
                tvy_a = t
                tvy_b = 0
                
                nx_a = xa
                nx_b = xb
                ny_a = (ya + tvy_a) % p
                ny_b = (yb + tvy_b) % p
                
                npt = (nx_a, nx_b, ny_a, ny_b)
                if npt not in N_set:
                    valid = False
                    break
            
            if not valid:
                return False
                
        return True

    # Check N_full
    if is_valid_nikodym(N_full):
        return np.array(N_full, dtype=np.int64)
    else:
        # Fallback to baseline
        return np.array(N_baseline, dtype=np.int64)

# EVOLVE-BLOCK-END
