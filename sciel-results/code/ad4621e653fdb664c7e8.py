import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Returns an integer array of shape (k, 2, 2) representing points in N.
    """
    p = 7
    q = p * p
    
    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul_f(a, b):
        a1, a2 = a
        b1, b2 = b
        c0 = (a1 * b1 + a2 * b2 * 3) % p
        c1 = (a1 * b2 + a2 * b1) % p
        return (c0, c1)
    
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {elem: i for i, elem in enumerate(field_elements)}
    
    total_points = q * q
    
    # Generate lines
    line_points = []
    seen_lines = set()
    
    for x0_a in range(p):
        for x0_b in range(p):
            x0 = (x0_a, x0_b)
            for y0_a in range(p):
                for y0_b in range(p):
                    y0 = (y0_a, y0_b)
                    p_idx = elem_to_idx[x0] * q + elem_to_idx[y0]
                    
                    # Slopes m in F_49
                    for m_a in range(p):
                        for m_b in range(p):
                            m = (m_a, m_b)
                            pts = []
                            for x_a in range(p):
                                for x_b in range(p):
                                    x = (x_a, x_b)
                                    dx = ((x[0] - x0[0]) % p, (x[1] - x0[1]) % p)
                                    mx = mul_f(m, dx)
                                    y = add_f(mx, y0)
                                    pt_idx = elem_to_idx[x] * q + elem_to_idx[y]
                                    pts.append(pt_idx)
                            
                            pts_sorted = tuple(sorted(pts))
                            if pts_sorted not in seen_lines:
                                seen_lines.add(pts_sorted)
                                line_points.append(np.array(pts_sorted, dtype=np.int32))
                    
                    # Vertical line
                    pts = []
                    for y_a in range(p):
                        for y_b in range(p):
                            y = (y_a, y_b)
                            pt_idx = elem_to_idx[x0] * q + elem_to_idx[y]
                            pts.append(pt_idx)
                    pts_sorted = tuple(sorted(pts))
                    if pts_sorted not in seen_lines:
                        seen_lines.add(pts_sorted)
                        line_points.append(np.array(pts_sorted, dtype=np.int32))
    
    # Convert line_points to numpy arrays for faster access
    # line_points is a list of numpy arrays
    
    # Precompute point_lines: for each point, list of line indices
    point_lines = [[] for _ in range(total_points)]
    for i, line in enumerate(line_points):
        for pt in line:
            point_lines[pt].append(i)
    
    # Convert point_lines to numpy arrays?
    # They are small lists (50 elements). Lists are fine.
    
    def run_greedy_vectorized(seed):
        np.random.seed(seed)
        avail = np.full(total_points, q + 1, dtype=np.int32)
        in_S = np.zeros(total_points, dtype=bool)
        
        S_size = 0
        # We don't need an explicit active set if we check avail > 0
        # But we need to avoid picking points already in S.
        # in_S is updated.
        
        while True:
            # Find candidates: not in S and avail > 0
            # This is O(N)
            candidates = np.where((~in_S) & (avail > 0))[0]
            if len(candidates) == 0:
                break
            
            # Pick a point
            # Strategy: Random
            pt = candidates[np.random.randint(len(candidates))]
            
            in_S[pt] = True
            S_size += 1
            
            # Update avail for neighbors
            for line_idx in point_lines[pt]:
                pts = line_points[line_idx]
                # Decrement avail for all pts except pt
                # We can just decrement all and then increment pt?
                # No, pt is removed from candidates anyway.
                # But avail[pt] is not used.
                # So we can just decrement all pts in the line.
                avail[pts] -= 1
            
        return S_size, in_S

    start_time = time.time()
    best_size = 0
    best_in_S = None
    
    run_id = 0
    # Run as many as possible
    while time.time() - start_time < 580: # Safety margin
        size, in_S = run_greedy_vectorized(run_id)
        if size > best_size:
            best_size = size
            best_in_S = in_S.copy()
            # print(f"Run {run_id}: Size {size}")
        run_id += 1
        
        # If we find a very large set, maybe stop?
        # Max possible is 2401.
        # But likely much smaller.
        
    # Construct N from best_in_S
    if best_in_S is None:
        # Fallback: empty S? No, S must be non-empty?
        # "k >= 1" for N. So N must have at least 1 point.
        # If S is empty, N is all points. Valid.
        # But we want to maximize S.
        # If no runs finished, return all points.
        best_in_S = np.zeros(total_points, dtype=bool)
    
    points = []
    for i in range(total_points):
        if not best_in_S[i]:
            x_idx = i // q
            y_idx = i % q
            x = field_elements[x_idx]
            y = field_elements[y_idx]
            points.append([[x[0], x[1]], [y[0], y[1]]])
    
    return np.array(points, dtype=np.int64)
