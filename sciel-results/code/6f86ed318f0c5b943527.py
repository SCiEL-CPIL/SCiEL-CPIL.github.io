# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    # alpha^2 = 3
    k = 3
    
    # Field operations for F_49 = F_7[alpha]
    # Element is (a, b) representing a + b*alpha
    def add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
    
    def sub(e1, e2):
        return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)
    
    def mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1a2 + (a1b2 + a2b1)alpha + b1b2*alpha^2
        # alpha^2 = k
        a = (a1 * a2 + b1 * b2 * k) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)
        
    def inv(e):
        # (a + b*alpha)^-1 = (a - b*alpha) / (a^2 - b^2*k)
        a, b = e
        denom = (a * a - b * b * k) % p
        if denom == 0:
            return (0, 0) # Should not happen for non-zero
        inv_denom = pow(denom, p - 2, p)
        a_inv = (a * inv_denom) % p
        b_inv = (-b * inv_denom) % p
        return (a_inv, b_inv)
        
    def is_zero(e):
        return e[0] == 0 and e[1] == 0
        
    # Points in F_49^2
    # Index 0..2400
    # Point x = (x1, x2) where x1, x2 in F_49
    # x1 = (xa, xb), x2 = (ya, yb)
    
    # Generate all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    # Map point to index
    point_to_idx = {pt: i for i, pt in enumerate(points)}
    
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # v is a vector in F_49^2
    directions = []
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            v = ((1, 0), m) # (1, m)
            directions.append(v)
    v_inf = ((0, 0), (1, 0)) # (0, 1)
    directions.append(v_inf)
    
    # Precompute lines
    # line_points[x_idx][v_idx] = set of point indices on the line
    # This might be too large: 2401 * 2450 * 49 points?
    # 2401 * 2450 ~ 6M entries. Each entry is a list of 49 ints.
    # 6M * 49 * 4 bytes ~ 1.2 GB. Too much.
    # We can compute on the fly or store more compactly.
    # Or just store the set of points on the line for each (x, v).
    # Let's compute the "bad" status on the fly.
    
    # Initial S: Unital U
    # Tr(x) + Nm(y) = 0
    # Tr(x) = 2*xa
    # Nm(y) = ya^2 - yb^2*k
    S = set()
    for i, pt in enumerate(points):
        x1, x2 = pt
        xa, xb = x1
        ya, yb = x2
        tr_x = (2 * xa) % p
        nm_y = (ya * ya - yb * yb * k) % p
        if (tr_x + nm_y) % p == 0:
            S.add(i)
            
    # Helper to check if point x is covered by direction v given S
    def is_covered(x_idx, v_idx, S_set):
        x = points[x_idx]
        v = directions[v_idx]
        # Check if S_set intersect (x + t*v) is subset of {x}
        # i.e., no other point in S_set is on the line
        # We iterate t in F_49, t != 0
        # t = (ta, tb)
        for ta in range(p):
            for tb in range(p):
                if ta == 0 and tb == 0:
                    continue
                t = (ta, tb)
                # y = x + t*v
                # v = (v1, v2)
                v1, v2 = v
                # t*v = (t*v1, t*v2)
                tv1 = mul(t, v1)
                tv2 = mul(t, v2)
                y1 = add(x[0], tv1)
                y2 = add(x[1], tv2)
                y = (y1, y2)
                y_idx = point_to_idx[y]
                if y_idx in S_set and y_idx != x_idx:
                    return False
        return True
        
    # Main loop
    start_time = time.time()
    deadline = 600 - 30 # Safety margin
    
    while time.time() < deadline:
        bad_points = []
        for x_idx in range(q * q):
            covered = False
            for v_idx in range(len(directions)):
                if is_covered(x_idx, v_idx, S):
                    covered = True
                    break
            if not covered:
                bad_points.append(x_idx)
                
        if not bad_points:
            break
            
        # Repair
        # Strategy:
        # 1. If any bad point is in S, remove it.
        # 2. Else, for a bad point in N, remove points from S blocking it.
        
        removed_any = False
        bad_in_S = [x for x in bad_points if x in S]
        if bad_in_S:
            # Remove one
            x_to_remove = bad_in_S[0]
            S.remove(x_to_remove)
            removed_any = True
        else:
            # All bad points are in N
            # Pick one bad point
            x_idx = bad_points[0]
            # Find direction v with minimum |S intersect L|
            min_count = float('inf')
            best_v = -1
            best_points_to_remove = set()
            
            for v_idx in range(len(directions)):
                # Collect points of S on the line
                pts_on_line = []
                x = points[x_idx]
                v = directions[v_idx]
                v1, v2 = v
                for ta in range(p):
                    for tb in range(p):
                        if ta == 0 and tb == 0:
                            continue
                        t = (ta, tb)
                        tv1 = mul(t, v1)
                        tv2 = mul(t, v2)
                        y1 = add(x[0], tv1)
                        y2 = add(x[1], v2) # Wait, y2 = x[1] + t*v2
                        y2 = add(x[1], tv2)
                        y = (y1, y2)
                        y_idx = point_to_idx[y]
                        if y_idx in S:
                            pts_on_line.append(y_idx)
                            
                if len(pts_on_line) < min_count:
                    min_count = len(pts_on_line)
                    best_v = v_idx
                    best_points_to_remove = set(pts_on_line)
                    
            if best_points_to_remove:
                for y_idx in best_points_to_remove:
                    S.remove(y_idx)
                removed_any = True
            else:
                # Should not happen if x is bad and x not in S
                # If S intersect L is empty for some v, x would be covered.
                # If S intersect L is not empty for all v, min_count > 0.
                break
        if not removed_any:
             break

    # Construct result
    # N is the complement of S
    # Return points in N
    result = []
    for i in range(q * q):
        if i not in S:
            result.append(points[i])
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
