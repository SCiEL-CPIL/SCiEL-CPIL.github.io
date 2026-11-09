# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses baseline construction with greedy improvement attempt.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    q = p * p
    
    # F_49 arithmetic
    def add_f49(p1, p2):
        return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)
    
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + 3*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute lines
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    
    slope_lines = {}
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines[key].append(i)
            slope_lines[(ma, mb)] = dict(lines)
    
    # Initial N: exclude points on unital with y_a in QR
    k = 3
    y_special = set((u*u % p, v) for u in range(p) for v in range(p))
    
    in_N = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0 and (ya, yb) in y_special:
            in_N[i] = False
    
    # Check if we can exclude more points by testing if unital points with y_a NOT in QR can be excluded
    # For each such point, check if there's a witness line
    def has_witness_line(pt_idx, in_N_arr):
        """Check if point pt_idx has at least one witness line given current in_N."""
        xa, xb, ya, yb = all_points[pt_idx]
        
        # Check vertical line
        vert = vert_lines[(xa, xb)]
        if all(in_N_arr[j] for j in vert if j != pt_idx):
            return True
        
        # Check slope lines
        for ma in range(p):
            for mb in range(p):
                if ma == 0 and mb == 0:
                    continue
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                sl = slope_lines[(ma, mb)].get(key, [])
                if all(in_N_arr[j] for j in sl if j != pt_idx):
                    return True
        return False
    
    # Try to exclude additional unital points
    if time.time() < deadline - 60:
        for i, (xa, xb, ya, yb) in enumerate(all_points):
            if time.time() > deadline - 30:
                break
            if in_N[i]:
                continue
            # Already excluded, skip
            # Try to exclude points that are on unital but not yet excluded
            if (2*xa + ya*ya - yb*yb*k) % p == 0 and (ya, yb) not in y_special:
                # Check if this point can be excluded (has witness line)
                # Temporarily mark as excluded and check all points
                in_N[i] = False
                # Check if all points still have witness lines
                valid = True
                for j in range(n):
                    if not has_witness_line(j, in_N):
                        valid = False
                        break
                if valid:
                    # Keep it excluded
                    pass
                else:
                    # Revert
                    in_N[i] = True
    
    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
