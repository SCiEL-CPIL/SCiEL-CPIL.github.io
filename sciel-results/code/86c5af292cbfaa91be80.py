import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Element (a, b) represents a + b*alpha
    # Multiplication: (a, b) * (c, d) = (ac + bd*3, ad + bc)
    
    def field_mul(a, b):
        # a, b are tuples (int, int)
        return ((a[0]*b[0] + a[1]*b[1]*3) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def field_add(a, b):
        return ((a[0]+b[0]) % p, (a[1]+b[1]) % p)
        
    def field_sub(a, b):
        return ((a[0]-b[0]) % p, (a[1]-b[1]) % p)

    # Generate all points
    points = []
    point_to_idx = {}
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    idx = len(points)
                    pt = ((a, b), (c, d))
                    points.append(pt)
                    point_to_idx[pt] = idx

    num_points = len(points) # 2401
    
    # Generate all lines
    # Lines are defined by direction v and a point x0.
    # Directions: (1, m) for m in F_49, and (0, 1).
    # There are 49 + 1 = 50 directions.
    # For each direction, there are 49 parallel lines.
    # Total lines = 50 * 49 = 2450.
    
    directions = []
    for m_a in range(p):
        for m_b in range(p):
            directions.append(((1, 0), (m_a, m_b))) # v = (1, m)
    directions.append(((0, 1), (0, 0))) # v = (0, 1)
    
    lines = [] # List of lists of point indices
    line_id = 0
    # We need to uniquely identify lines.
    # A line can be represented by (direction, offset).
    # For v=(1,m), the line through x0=(x0a, x0b, x0c, x0d) is y = x0 + t*v.
    # To avoid duplicates, we can normalize the representative point.
    # For v=(1,m), x0 should have x0_a = 0? No, x is the first coordinate.
    # Let x = (x_a, x_b). The line is { (x_a + t, x_b + t*m) + y0 * (0,1) ... }?
    # No, the line is { x0 + t*v | t in F_49 }.
    # If v=(1,m), then the x-coordinate of points on the line is x0_x + t.
    # Since t ranges over F_49, the x-coordinate ranges over all F_49.
    # So for a fixed y-coordinate "offset", there is exactly one line for each y?
    # Let's just generate all lines by picking a base point and direction, and normalizing.
    
    seen_lines = set()
    
    for v in directions:
        va, vb = v[0] # v is (vx, vy) where vx, vy are field elements
        # Actually v is ((1,0), (ma,mb)) or ((0,1),(0,0)).
        vx_a, vx_b = v[0]
        vy_a, vy_b = v[1]
        
        # Iterate over all possible base points x0
        # To avoid duplicates, we only pick x0 such that it is the "first" point on the line in some ordering.
        # Simpler: iterate all x0, generate line, check if seen.
        for x0_idx in range(num_points):
            x0 = points[x0_idx]
            # Generate points on line
            line_pts = set()
            for t_a in range(p):
                for t_b in range(p):
                    t = (t_a, t_b)
                    # x = x0 + t*v
                    tx_a = (x0[0][0] + t_a*vx_a + t_b*vx_b*3) % p
                    tx_b = (x0[0][1] + t_a*vx_b + t_b*vx_a) % p
                    ty_a = (x0[1][0] + t_a*vy_a + t_b*vy_b*3) % p
                    ty_b = (x0[1][1] + t_a*vy_b + t_b*vy_a) % p
                    pt = ((tx_a, tx_b), (ty_a, ty_b))
                    line_pts.add(pt)
            
            # Canonical form of line: sorted tuple of points
            canonical = tuple(sorted(line_pts))
            if canonical not in seen_lines:
                seen_lines.add(canonical)
                lines.append([point_to_idx[pt] for pt in line_pts])
                line_id += 1

    num_lines = len(lines)
    
    # Precompute lines through each point
    lines_through_point = [[] for _ in range(num_points)]
    for l_idx in range(num_lines):
        for p_idx in lines[l_idx]:
            lines_through_point[p_idx].append(l_idx)

    # Greedy algorithm to maximize S
    # We will try multiple random orderings and keep the best S.
    
    best_S_size = 0
    best_S = set()
    
    # Number of trials
    trials = 100
    
    for trial in range(trials):
        np.random.seed(42 + trial)
        order = np.random.permutation(num_points)
        
        S = set()
        C = np.zeros(num_lines, dtype=np.int32) # Count of S points on each line
        n0 = np.full(num_points, 50, dtype=np.int32) # Count of lines through y with C=0
        n1 = np.zeros(num_points, dtype=np.int32) # Count of lines through y with C=1
        in_S = np.zeros(num_points, dtype=bool)
        
        # Initial state: S is empty.
        # All lines have C=0.
        # For any y, all 50 lines through y have C=0.
        # So n0[y] = 50, n1[y] = 0.
        # All y not in S are valid (n0 > 0).
        
        for x_idx in order:
            # Try to add x_idx to S
            # Store old values to revert if invalid
            old_C = np.zeros(num_lines, dtype=np.int32)
            old_n0 = np.zeros(num_points, dtype=np.int32)
            old_n1 = np.zeros(num_points, dtype=np.int32)
            
            # Only store for affected lines and points?
            # All points are affected in terms of n0/n1?
            # Yes, because x is on 50 lines, and those 50 lines cover all points.
            # So all n0/n1 change.
            # But we can just copy the arrays? 2401 * 2 + 2450 = ~7000 ints. Very fast.
            old_C[:] = C
            old_n0[:] = n0
            old_n1[:] = n1
            
            # Update C for lines through x_idx
            for l_idx in lines_through_point[x_idx]:
                C[l_idx] += 1
                
            # Update n0, n1 for all points
            # For each line L, the count changed from old_C[L] to C[L].
            # For each point y on L, update n0/n1.
            # This is O(num_lines * 49) = 2450 * 49 ~ 120k ops.
            # We can do it more efficiently by iterating points?
            # For each point y, it is on 50 lines.
            # We need to know how many of those 50 lines changed from 0->1 or 1->2.
            
            # Let's iterate lines and update points on them.
            for l_idx in lines_through_point[x_idx]:
                old_c = old_C[l_idx]
                new_c = C[l_idx]
                
                # Only 0->1 and 1->2 matter for n0/n1
                if old_c == 0 and new_c == 1:
                    for y_idx in lines[l_idx]:
                        n0[y_idx] -= 1
                        n1[y_idx] += 1
                elif old_c == 1 and new_c == 2:
                    for y_idx in lines[l_idx]:
                        n1[y_idx] -= 1
                # else: no change to n0/n1
            
            # Check validity
            valid = True
            for y_idx in range(num_points):
                if in_S[y_idx]:
                    if n1[y_idx] == 0:
                        valid = False
                        break
                else:
                    if n0[y_idx] == 0:
                        valid = False
                        break
            
            if valid:
                S.add(x_idx)
                in_S[x_idx] = True
            else:
                # Revert
                C[:] = old_C
                n0[:] = old_n0
                n1[:] = old_n1
                # in_S[x_idx] was not set to True, so no need to revert it.
        
        if len(S) > best_S_size:
            best_S_size = len(S)
            best_S = set(S)
            
    # Construct the result array N = F_49^2 \ S
    # The task asks for N, the Nikodym set.
    # "returning an integer array of shape (k, 2, 2), k >= 1, listing the points of a set N."
    # "Metric: the complement size 2401 - |N|"
    # So we want N to be small, meaning S to be large.
    # The return value is the points in N.
    
    points_in_N = []
    for idx in range(num_points):
        if idx not in best_S:
            pt = points[idx]
            points_in_N.append([[pt[0][0], pt[0][1]], [pt[1][0], pt[1][1]]])
            
    return np.array(points_in_N, dtype=np.int64)
