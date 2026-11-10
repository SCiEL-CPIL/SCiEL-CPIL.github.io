# EVOLVE-BLOCK-START
import numpy as np

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_{49}^2 using a Greedy Removal algorithm.
    
    This approach starts with the full plane F_{49}^2 and iteratively removes points
    that can be removed while preserving the Nikodym property. A point x can be removed
    if, after removal, every point y in F_{49}^2 still has at least one witness line
    L such that L \ {y} is a subset of the remaining set N.
    
    We maintain:
    - in_N: boolean array indicating which points are in N.
    - H[L]: number of points in line L that are NOT in N.
    - C[y]: number of valid witness lines for point y.
    
    A line L is a valid witness for y if:
    - If y in N: H[L] == 0 (all other points in L are in N).
    - If y not in N: H[L] == 1 (all other points in L are in N, except y itself which is not).
    
    The greedy step checks if removing x keeps C[y] >= 1 for all y.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Point representation: idx = ((xa * p + xb) * p + ya) * p + yb
    # We need to generate all lines.
    # A line is defined by a point (x0, y0) and a slope m in F_{49} or infinity.
    # In F_{49}, elements are pairs (a, b).
    # Slope m can be (1, m_a, m_b) or vertical (0, 1).
    # Actually, lines in F_q^2 are of the form y = mx + c or x = c.
    # There are q slopes m and q intercepts c for non-vertical lines: q*q lines.
    # There are q vertical lines.
    # Total lines: q^2 + q = 49*50 = 2450.
    
    # Precompute line membership
    # line_id -> list of point indices
    # point_id -> list of line ids
    points_in_line = []
    lines_through_point = [[] for _ in range(total_points)]
    
    # Helper to get coordinates from index
    def idx_to_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return xa, xb, ya, yb
        
    def coords_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Field arithmetic helpers for F_7[alpha] with alpha^2=3
    # Element is (a, b) representing a + b*alpha
    def f_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
        
    def f_mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1a2 + b1b2*alpha^2 + (a1b2 + a2b1)alpha
        # alpha^2 = 3
        real = (a1 * a2 + 3 * b1 * b2) % p
        imag = (a1 * b2 + a2 * b1) % p
        return (real, imag)
        
    def f_inv(e):
        # Inverse of a + b*alpha is (a - b*alpha) / (a^2 - 3b^2)
        a, b = e
        norm = (a * a - 3 * b * b) % p
        if norm == 0:
            return (0, 0) # Should not happen for non-zero elements
        inv_norm = pow(norm, p - 2, p)
        real = (a * inv_norm) % p
        imag = (-b * inv_norm) % p
        return (real, imag)

    # Generate all points
    all_points = []
    for idx in range(total_points):
        all_points.append(idx_to_coords(idx))

    # Generate lines
    # Non-vertical lines: y = mx + c
    # m in F_49, c in F_49
    line_id = 0
    slopes = []
    for ma in range(p):
        for mb in range(p):
            slopes.append((ma, mb))
            
    intercepts = slopes # c is in F_49
    
    for m in slopes:
        for c in intercepts:
            line_pts = []
            for x in slopes: # x in F_49
                # y = mx + c
                y = f_add(f_mul(m, x), c)
                # Convert x, y to indices
                # x is (xa, xb), y is (ya, yb)
                idx = coords_to_idx(x[0], x[1], y[0], y[1])
                line_pts.append(idx)
            points_in_line.append(line_pts)
            for pt in line_pts:
                lines_through_point[pt].append(line_id)
            line_id += 1
            
    # Vertical lines: x = c
    for c in slopes:
        line_pts = []
        for y in slopes: # y in F_49
            idx = coords_to_idx(c[0], c[1], y[0], y[1])
            line_pts.append(idx)
        points_in_line.append(line_pts)
        for pt in line_pts:
            lines_through_point[pt].append(line_id)
        line_id += 1
        
    num_lines = line_id
    # num_lines should be 2450
    
    # State
    in_N = np.ones(total_points, dtype=bool)
    H = np.zeros(num_lines, dtype=np.int32) # Holes in line
    C = np.zeros(total_points, dtype=np.int32) # Valid witnesses for point
    
    # Initialize C
    # For all y in N (which is all), H[L]=0 for all L.
    # So all lines are valid witnesses.
    # Each point is on 50 lines.
    C[:] = 50
    
    # Greedy removal
    # We try to remove points in a random order to avoid local minima or bias
    # We can run multiple passes if time permits, but one pass is likely sufficient for a good score.
    # Let's do a single pass with random shuffle.
    indices = np.arange(total_points)
    np.random.shuffle(indices)
    
    for x_idx in indices:
        if not in_N[x_idx]:
            continue
            
        # Simulate removal of x_idx
        # Affected lines: lines_through_point[x_idx]
        # Affected points: all points in those lines except x_idx
        
        # We need to track delta_C for affected points
        # To avoid allocating a full array, we can use a dict or just check on the fly?
        # Since we need to verify ALL points have C>=1, and only affected points change,
        # we only need to check affected points.
        
        affected_points = set()
        delta_C = {}
        
        valid = True
        for L in lines_through_point[x_idx]:
            h_old = H[L]
            h_new = h_old + 1
            
            # For each point y in this line
            for y in points_in_line[L]:
                if y == x_idx:
                    continue
                
                # y is affected by the change in H[L]
                # Update delta_C[y]
                if y not in delta_C:
                    delta_C[y] = 0
                
                if in_N[y]:
                    # y in N: valid if H[L] == 0
                    if h_old == 0:
                        # Was valid, becomes invalid
                        delta_C[y] -= 1
                    # else: was invalid, still invalid
                else:
                    # y not in N: valid if H[L] == 1
                    if h_old == 0:
                        # Was invalid, becomes valid
                        delta_C[y] += 1
                    elif h_old == 1:
                        # Was valid, becomes invalid
                        delta_C[y] -= 1
                    # else: was invalid, still invalid
                
                affected_points.add(y)
        
        # Check if all affected points still have C >= 1
        for y in affected_points:
            if C[y] + delta_C[y] < 1:
                valid = False
                break
                
        if valid:
            # Accept removal
            in_N[x_idx] = False
            
            # Update H
            for L in lines_through_point[x_idx]:
                H[L] += 1
                
            # Update C
            for y in affected_points:
                C[y] += delta_C[y]

    # Construct result
    result_indices = [i for i in range(total_points) if in_N[i]]
    k = len(result_indices)
    
    result = np.zeros((k, 2, 2), dtype=np.int64)
    for i, idx in enumerate(result_indices):
        xa, xb, ya, yb = idx_to_coords(idx)
        result[i] = [[xa, xb], [ya, yb]]
        
    return result

# EVOLVE-BLOCK-END
