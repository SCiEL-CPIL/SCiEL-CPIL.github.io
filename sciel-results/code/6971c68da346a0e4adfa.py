# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses baseline construction and greedy improvement by adding points from the unital.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    q = p * p
    k = 3 # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)

    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute lines
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # We represent lines by their points.
    # For efficiency, we map point index to list of line indices passing through it.
    # And line index to set of point indices on it.
    
    # Generate all lines
    lines = [] # list of lists of point indices
    point_to_lines = [[] for _ in range(n)]
    
    # Vertical lines (x constant)
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    pt = (xa, xb, ya, yb)
                    idx = point_idx[pt]
                    line_pts.append(idx)
            lines.append(line_pts)
            for idx in line_pts:
                point_to_lines[idx].append(len(lines)-1)
                
    # Slope lines
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            m = (ma, mb)
            # Group points by y - m*x
            groups = {}
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49(m, (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                if key not in groups:
                    groups[key] = []
                groups[key].append(i)
            
            for key, pts in groups.items():
                lines.append(pts)
                for idx in pts:
                    point_to_lines[idx].append(len(lines)-1)

    # Baseline S
    y_special = set((u*u % p, v) for u in range(p) for v in range(p))
    
    in_N = [True] * n
    in_S = [False] * n
    
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0 and (ya, yb) not in y_special:
            in_N[i] = False
            in_S[i] = True
            
    # Calculate W(x) = number of lines through x disjoint from S
    # A line is disjoint from S if no point on it is in S.
    # First, mark lines that are NOT disjoint from S.
    line_has_S = [False] * len(lines)
    for i in range(n):
        if in_S[i]:
            for l_idx in point_to_lines[i]:
                line_has_S[l_idx] = True
                
    W = [0] * n
    for i in range(n):
        if in_N[i]:
            for l_idx in point_to_lines[i]:
                if not line_has_S[l_idx]:
                    W[i] += 1
        else:
            W[i] = 0 # Not needed for N points, but for S points we check witness later
            
    # Check initial validity
    # For x in N, W[x] must be > 0.
    # For x in S, there must be a line L through x such that L intersect S = {x}.
    # This is equivalent to: L has no other points in S.
    # i.e., L \ {x} is disjoint from S.
    
    def is_valid_S_point(x):
        # Check if x has a witness line
        for l_idx in point_to_lines[x]:
            # Check if line l_idx contains any other point in S
            ok = True
            for y in lines[l_idx]:
                if y != x and in_S[y]:
                    ok = False
                    break
            if ok:
                return True
        return False

    # Initial check
    for i in range(n):
        if in_N[i] and W[i] == 0:
            # Invalid baseline? Should not happen.
            pass
        if in_S[i] and not is_valid_S_point(i):
            # Invalid baseline?
            pass

    # Greedy improvement: add points from U \ S
    # Candidates are points in U not in S
    candidates = []
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if in_N[i] and (2*xa + ya*ya - yb*yb*k) % p == 0:
            candidates.append(i)
            
    # Sort candidates? Maybe by W[i] ascending (most constrained first) or descending?
    # Let's just iterate.
    
    added = 0
    for c in candidates:
        if time.time() > deadline - 10:
            break
            
        # Tentatively add c to S
        in_N[c] = False
        in_S[c] = True
        
        # Update W for points affected by adding c
        # Lines through c that were previously disjoint from S are now NOT disjoint.
        # For each such line L, for each x in L \ {c}, W[x] decreases by 1.
        valid = True
        affected_points = []
        
        for l_idx in point_to_lines[c]:
            # Check if L was disjoint from S BEFORE adding c
            # i.e., L had no points in S (since c was not in S).
            # We can check if line_has_S[l_idx] was False.
            # But line_has_S is global. We need to know if it was False before.
            # Since we only add points, line_has_S only changes from False to True.
            # If line_has_S[l_idx] is True now, it might have been True before or False before (due to c).
            # We need to track which lines became "blocked" by c.
            
            # A line L is blocked by c if L contains c and L contained no other S points.
            # This is equivalent to: L \ {c} is disjoint from S_old.
            # We can check this by looking at points in L.
            
            # Optimization:
            # If W[x] was 1 for some x on L, and L is the only witness, x becomes invalid.
            
            # Let's just re-evaluate validity for all N points?
            # Too slow.
            
            # Let's check if any x on L (x != c) has W[x] == 1 and L was its only witness.
            # We don't store "only witness", just count.
            # If W[x] becomes 0, it's invalid.
            
            # We need to know if L was disjoint from S_old.
            # L is disjoint from S_old iff no y in L \ {c} is in S_old.
            # Since c is the only new point, if L had any S point, it was in S_old.
            # So L is "newly blocked" if L \ {c} has no S points.
            
            has_other_S = False
            for y in lines[l_idx]:
                if y != c and in_S[y]:
                    has_other_S = True
                    break
            
            if not has_other_S:
                # L was disjoint from S_old, now contains c.
                # So L is no longer a witness for any x in L \ {c}.
                for y in lines[l_idx]:
                    if y != c:
                        W[y] -= 1
                        affected_points.append(y)
                        if W[y] < 0:
                            valid = False
                            break
                if not valid:
                    break
        
        if valid:
            # Check S validity for c
            if is_valid_S_point(c):
                added += 1
                # Keep change
            else:
                # Revert
                in_N[c] = True
                in_S[c] = False
                for y in affected_points:
                    W[y] += 1
        else:
            # Revert
            in_N[c] = True
            in_S[c] = False
            for y in affected_points:
                W[y] += 1

    # Build result
    pts = [all_points[i] for i in range(n) if in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
