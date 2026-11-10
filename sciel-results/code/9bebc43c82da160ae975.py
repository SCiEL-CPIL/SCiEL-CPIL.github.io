import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    # alpha^2 = 3
    k = 3
    
    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0b0 + a0b1*alpha + a1b0*alpha + a1b1*alpha^2
        # = (a0b0 + a1b1*k) + (a0b1 + a1b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def to_idx(pt):
        return pt[0][0] * p * p + pt[0][1] * p + pt[1][0] * p + pt[1][1]
        
    # Generate all points
    points = []
    for x0 in range(p):
        for x1 in range(p):
            for y0 in range(p):
                for y1 in range(p):
                    points.append(((x0, x1), (y0, y1)))
                    
    # Identify Unital U: Trace(x) + Norm(y) = 0
    # Trace(x) = 2*x0
    # Norm(y) = y0^2 - y1^2*k
    U_indices = set()
    for i, pt in enumerate(points):
        x = pt[0]
        y = pt[1]
        trace = (2 * x[0]) % p
        norm = (y[0]*y[0] - y[1]*y[1]*k) % p
        if (trace + norm) % p == 0:
            U_indices.add(i)
            
    # Precompute lines
    # Directions: (1, m) for m in F_q, and (0, 1)
    # We need lines for each point to check disjointness.
    # Instead of storing all lines, we can compute on the fly or store line sets.
    # Number of lines is 2450.
    # Let's store for each point, the list of lines passing through it.
    # And for each line, the list of points.
    
    # Directions
    directions = []
    for m0 in range(p):
        for m1 in range(p):
            directions.append(((1, 0), (m0, m1))) # v = (1, m)
    directions.append(((0, 0), (0, 1))) # v = (0, 1) -> x=0, y varies. Wait, v=(0,1) means x is const.
    
    # Actually, v=(0,1) is the vertical direction.
    # v=(1,m) covers all non-vertical lines.
    
    lines = []
    point_to_lines = [[] for _ in range(q*q)]
    
    for d_idx, v in enumerate(directions):
        # For each point on the line, we need to define the line uniquely.
        # A line is defined by a point and a direction.
        # But we need unique lines.
        # There are q lines for each direction.
        # For v=(1,m), lines are y = m*x + c.
        # For v=(0,1), lines are x = c.
        
        if v[0][0] == 0: # vertical
            for c in range(p):
                # x = (c, 0) + ...? No, x is a pair.
                # x = (c, 0) is not right. x is in F_q.
                # x = c_alpha where c_alpha is a specific element?
                # The line is { (x, y) | x = x0 } for some x0 in F_q.
                for x0 in range(p):
                    for x1 in range(p):
                        x0_val = (x0, x1)
                        # Points on line: (x0_val, y) for all y in F_q
                        line_pts = []
                        for y0 in range(p):
                            for y1 in range(p):
                                y_val = (y0, y1)
                                pt_idx = to_idx(((x0_val), (y_val)))
                                line_pts.append(pt_idx)
                        # Add line
                        line_id = len(lines)
                        lines.append(line_pts)
                        for pt_idx in line_pts:
                            point_to_lines[pt_idx].append(line_id)
        else:
            # v = (1, m)
            m_val = v[1]
            for c0 in range(p):
                for c1 in range(p):
                    c_val = (c0, c1)
                    line_pts = []
                    for x0 in range(p):
                        for x1 in range(p):
                            x_val = (x0, x1)
                            # y = m*x + c
                            mx = mul(m_val, x_val)
                            y_val = add(mx, c_val)
                            pt_idx = to_idx((x_val, y_val))
                            line_pts.append(pt_idx)
                    line_id = len(lines)
                    lines.append(line_pts)
                    for pt_idx in line_pts:
                        point_to_lines[pt_idx].append(line_id)
                        
    # State
    # S: set of indices in U
    S = set()
    # disjoint_count[x] = number of lines through x with 0 points in S
    disjoint_count = [0] * (q*q)
    
    # Initialize disjoint_count
    # Initially S is empty, so all lines are disjoint.
    # Each point is on q+1 lines.
    for i in range(q*q):
        disjoint_count[i] = len(point_to_lines[i])
        
    # Greedy add
    # Try to add points from U
    # Order: random
    candidates = list(U_indices)
    
    best_S = set()
    best_size = 0
    
    start_time = time.time()
    limit = 600.0
    
    # We can do multiple random restarts
    for restart in range(100):
        random.shuffle(candidates)
        S = set()
        disjoint_count = [len(point_to_lines[i]) for i in range(q*q)]
        # line_point_count[l] = number of points in S on line l
        line_point_count = [0] * len(lines)
        
        for p_idx in candidates:
            if time.time() - start_time > limit - 10:
                break
            # Check if adding p_idx is valid
            # A line L through p_idx becomes "non-disjoint" if it was disjoint before.
            # If L was disjoint, all other points on L lose one disjoint line.
            # If any other point x on L (x not in S) has disjoint_count[x] == 1, it will become 0 -> Invalid.
            
            valid = True
            # We need to check lines through p_idx that are currently disjoint.
            # A line L is disjoint if line_point_count[L] == 0.
            # When we add p_idx, line_point_count[L] becomes 1.
            # For all x in L \ {p_idx}, disjoint_count[x] decreases by 1.
            # If x not in S and disjoint_count[x] becomes 0, invalid.
            
            # Optimization: Only check lines through p_idx with count 0.
            for l_id in point_to_lines[p_idx]:
                if line_point_count[l_id] == 0:
                    # This line will lose disjointness
                    for x_idx in lines[l_id]:
                        if x_idx == p_idx:
                            continue
                        if x_idx in S:
                            continue
                        # x_idx is not in S
                        if disjoint_count[x_idx] == 1:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                S.add(p_idx)
                for l_id in point_to_lines[p_idx]:
                    if line_point_count[l_id] == 0:
                        # Line becomes non-disjoint
                        for x_idx in lines[l_id]:
                            if x_idx == p_idx:
                                continue
                            if x_idx in S:
                                continue
                            disjoint_count[x_idx] -= 1
                    line_point_count[l_id] += 1
                # Note: p_idx itself is in S, so its disjoint_count doesn't matter for validity (condition 1).
                # But we might need to update it if we consider it for future? No, S is fixed.
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    # Construct N
    # N = F_q^2 \ S
    # Return points in N
    n_points = []
    for i in range(q*q):
        if i not in best_S:
            n_points.append(points[i])
            
    return np.array(n_points, dtype=np.int64)
