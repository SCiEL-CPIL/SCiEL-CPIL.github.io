import numpy as np
import time

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 10.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Point is ((xa, xb), (ya, yb))
    
    # 1. Construct Parabola P: y = x^2
    # x = xa + xb*alpha
    # x^2 = (xa^2 + 3*xb^2) + (2*xa*xb)*alpha
    P_points = []
    for xa in range(p):
        for xb in range(p):
            ya = (xa * xa + 3 * xb * xb) % p
            yb = (2 * xa * xb) % p
            P_points.append((xa, xb, ya, yb))
            
    P_set = set(P_points)
    
    # 2. Find Tangents to P
    # Tangent at x: y - y_x = m (x - x_x)
    # In F_49, slope m.
    # Actually, easier to check collinearity.
    # A line L contains x1, x2.
    # We need to find all lines that intersect P in exactly 1 point.
    
    # Precompute all lines?
    # Number of lines in F_q^2 is q(q+1) = 49*50 = 2450.
    # Each line has 49 points.
    # Total point-line incidences: 2450 * 49 = 120050.
    
    # Let's represent lines by (slope m, intercept c) or vertical (x=c).
    # m in F_49, c in F_49.
    # Vertical: x = c.
    
    # We want to find for each point p, the number of tangents passing through it.
    # A line L is tangent to P if |L cap P| == 1.
    
    # Iterate through all lines, check intersection with P.
    # If intersection is 1 point, it's a tangent.
    # Mark the points on that tangent.
    
    tangent_points = np.zeros(_Q2, dtype=int) # Count of tangents through each point
    
    # Helper to get point index
    def pt_idx(xa, xb, ya, yb):
        x = xa * p + xb
        y = ya * p + yb
        return x * q + y
        
    # Helper to get point coords from index
    def idx_pt(idx):
        y = idx % q
        x = idx // q
        ya, yb = y // p, y % p
        xa, xb = x // p, x % p
        return xa, xb, ya, yb

    # Iterate non-vertical lines: y = m*x + c
    # m, c in F_49
    # Represent m, c as (ma, mb), (ca, cb)
    
    # We need to check intersection with P.
    # P is given by y = x^2.
    # So we solve x^2 = m*x + c.
    # x^2 - m*x - c = 0.
    
    # To avoid complex field arithmetic in inner loop, we can precompute x^2 - m*x for all x?
    # Or just iterate x in F_49 and check if (x, x^2) is on the line.
    # Line check: y == m*x + c.
    
    # Let's precompute m*x + c for all x?
    # Too much memory/time if done naively.
    
    # Alternative:
    # For each line, count how many points of P are on it.
    # If count == 1, it's a tangent.
    
    # There are 2450 lines.
    # For each line, iterate 49 points of P?
    # 2450 * 49 = 120k operations. Very fast.
    
    # Let's store P points as indices.
    P_indices = [pt_idx(xa, xb, ya, yb) for xa, xb, ya, yb in P_points]
    
    # Generate all lines
    # Line represented by (type, param1, param2)
    # type 0: y = m*x + c. m=(ma,mb), c=(ca,cb)
    # type 1: x = c. c=(ca,cb)
    
    # We need to evaluate if a point (xa,xb, ya,yb) is on the line.
    # y = m*x + c
    # ya + yb*alpha = (ma+mb*alpha)*(xa+xb*alpha) + ca + cb*alpha
    # = (ma*xa + 3*mb*xb) + (ma*xb + mb*xa)*alpha + ca + cb*alpha
    # ya = ma*xa + 3*mb*xb + ca
    # yb = ma*xb + mb*xa + cb
    
    # Precompute P points coords.
    P_coords = P_points
    
    for ma in range(p):
        for mb in range(p):
            for ca in range(p):
                for cb in range(p):
                    count = 0
                    tangent_pt_idx = -1
                    for xa, xb, ya, yb in P_coords:
                        ya_calc = (ma * xa + 3 * mb * xb + ca) % p
                        yb_calc = (ma * xb + mb * xa + cb) % p
                        if ya_calc == ya and yb_calc == yb:
                            count += 1
                            tangent_pt_idx = pt_idx(xa, xb, ya, yb)
                    if count == 1:
                        tangent_points[tangent_pt_idx] += 1
                        
            if time.time() > _DEADLINE - _MARGIN:
                break
        if time.time() > _DEADLINE - _MARGIN:
            break
    if time.time() > _DEADLINE - _MARGIN:
        break
    if time.time() > _DEADLINE - _MARGIN:
        break
        
    # Vertical lines x = c
    for ca in range(p):
        for cb in range(p):
            count = 0
            tangent_pt_idx = -1
            for xa, xb, ya, yb in P_coords:
                if xa == ca and xb == cb:
                    count += 1
                    tangent_pt_idx = pt_idx(xa, xb, ya, yb)
            if count == 1:
                tangent_points[tangent_pt_idx] += 1
                
    # Candidate points K: not in P, and tangent_points[idx] == 0
    candidates = []
    for i in range(_Q2):
        if i not in P_indices and tangent_points[i] == 0:
            candidates.append(i)
            
    # Greedy addition
    S_indices = set(P_indices)
    
    # To check validity of S:
    # For each x in F_49^2, check if there is a line L through x with L cap S subseteq {x}.
    
    def is_valid(S_set):
        # S_set is a set of indices
        # For each x (0..2400), check lines through x.
        # There are 49 lines through x.
        # We need at least one line L such that:
        # if x in S: L cap S == {x}
        # if x not in S: L cap S == empty
        
        # Precompute lines through each point?
        # Or iterate lines and check?
        
        # Let's iterate x.
        for x_idx in range(_Q2):
            xa, xb, ya, yb = idx_pt(x_idx)
            
            # Check 49 lines through x.
            # Non-vertical: y = m*x + c. c is determined by x.
            # y - m*x = c.
            # ya + yb*a - (ma+mb*a)(xa+xb*a) = ca + cb*a
            # ca = ya - ma*xa - 3*mb*xb
            # cb = yb - ma*xb - mb*xa
            
            found = False
            for ma in range(p):
                for mb in range(p):
                    ca = (ya - ma * xa - 3 * mb * xb) % p
                    cb = (yb - ma * xb - mb * xa) % p
                    
                    # Check intersection of this line with S
                    # Line: y = (ma,mb)x + (ca,cb)
                    # We need to check if any point in S is on this line.
                    # And if x is in S, we allow x to be on the line.
                    
                    # Optimization:
                    # If x not in S, we need NO points from S on the line.
                    # If x in S, we need NO points from S \ {x} on the line.
                    
                    # Iterate S? |S| is small.
                    bad = False
                    for s_idx in S_set:
                        if s_idx == x_idx:
                            continue
                        sa, sb, sya, syb = idx_pt(s_idx)
                        sya_calc = (ma * sa + 3 * mb * sb + ca) % p
                        syb_calc = (ma * sb + mb * sa + cb) % p
                        if sya_calc == sya and syb_calc == syb:
                            bad = True
                            break
                            
                    if not bad:
                        found = True
                        break
                if found:
                    break
                    
            # Check vertical line x = xa + xb*a
            if not found:
                bad = False
                for s_idx in S_set:
                    if s_idx == x_idx:
                        continue
                    sa, sb, sya, syb = idx_pt(s_idx)
                    if sa == xa and sb == xb:
                        bad = True
                        break
                if not bad:
                    found = True
                    
            if not found:
                return False
                
        return True

    # Greedy loop
    # Sort candidates by some heuristic?
    # Maybe random order with restarts.
    
    best_S = set(P_indices)
    
    for restart in range(10):
        if time.time() > _DEADLINE - _MARGIN:
            break
            
        S_current = set(P_indices)
        cand_list = candidates[:]
        np.random.shuffle(cand_list)
        
        for c in cand_list:
            if time.time() > _DEADLINE - _MARGIN:
                break
                
            S_test = S_current | {c}
            if is_valid(S_test):
                S_current = S_test
                if len(S_current) > len(best_S):
                    best_S = S_current
                    
    # Construct N
    N = []
    for i in range(_Q2):
        if i not in best_S:
            xa, xb, ya, yb = idx_pt(i)
            N.append([[xa, xb], [ya, yb]])
            
    return np.array(N, dtype=np.int64)
