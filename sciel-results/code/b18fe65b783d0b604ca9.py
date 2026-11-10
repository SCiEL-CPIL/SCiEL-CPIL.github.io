import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    p = 7
    k = 3
    q = p * p
    
    def to_idx(a, b):
        return a * p + b
    def from_idx(idx):
        return (idx // p, idx % p)
    
    # Build unital
    total_pts = q * q
    U_mask = np.zeros(total_pts, dtype=np.bool_)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - k*yb*yb) % p == 0:
                        x_idx = to_idx(xa, xb)
                        y_idx = to_idx(ya, yb)
                        pt_idx = x_idx * q + y_idx
                        U_mask[pt_idx] = True
    
    # Build lines as lists of point indices
    lines = []
    for c in range(q):
        pts = [c * q + y for y in range(q)]
        lines.append(pts)
    
    # Precompute mul and add tables
    mul_t = np.zeros((q, q), dtype=np.int32)
    add_t = np.zeros((q, q), dtype=np.int32)
    for a in range(q):
        a0, a1 = from_idx(a)
        for b in range(q):
            b0, b1 = from_idx(b)
            r0 = (a0*b0 + a1*b1*k) % p
            r1 = (a0*b1 + a1*b0) % p
            mul_t[a, b] = to_idx(r0, r1)
            s0 = (a0+b0) % p
            s1 = (a1+b1) % p
            add_t[a, b] = to_idx(s0, s1)
    
    for m in range(q):
        for b in range(q):
            pts = []
            for x in range(q):
                y = add_t[mul_t[m, x], b]
                pts.append(x * q + y)
            lines.append(pts)
    
    # For each line, which U points it contains
    line_U = []
    for L in lines:
        line_U.append([pt for pt in L if U_mask[pt]])
    
    # C ⊆ U, N = all - C
    # Constraint: for each x in C, ∃ line through x with L∩C = {x}
    
    # Greedy: start C = U, remove violating points
    C_mask = U_mask.copy()
    
    # For speed, precompute for each point its lines
    pt_lines = [[] for _ in range(total_pts)]
    for i, L in enumerate(lines):
        for pt in L:
            pt_lines[pt].append(i)
    
    # Use a queue-based approach for efficiency
    from collections import deque
    
    # Check which points currently violate
    def check_violate(pt):
        if not C_mask[pt]:
            return False
        for li in pt_lines[pt]:
            L = lines[li]
            # Check if any other point in L is in C
            has_other = False
            for other in L:
                if other != pt and C_mask[other]:
                    has_other = True
                    break
            if not has_other:
                return False  # has a free line, not violating
        return True  # all lines have another C point, violating
    
    # Iteratively remove violators
    changed = True
    while changed:
        changed = False
        for pt in range(total_pts):
            if C_mask[pt] and check_violate(pt):
                C_mask[pt] = False
                changed = True
        if time.time() > deadline - 30:
            break
    
    N_mask = ~C_mask
    N_points = [pt for pt in range(total_pts) if N_mask[pt]]
    
    points = []
    for pt_idx in N_points:
        x_idx = pt_idx // q
        y_idx = pt_idx % q
        xa, xb = from_idx(x_idx)
        ya, yb = from_idx(y_idx)
        points.append([[xa, xb], [ya, yb]])
    
    if not points:
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
