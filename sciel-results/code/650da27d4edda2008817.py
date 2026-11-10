import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
            
    # Generate points of the Hermitian Unital U in affine space
    # Equation: 2*x_a + y_a^2 - k*y_b^2 = 0
    # x_a = 4 * (y_a^2 - k*y_b^2) mod 7
    # x_b is free
    
    u_points = []
    u_indices = {}
    
    for ya in range(p):
        for yb in range(p):
            xa = (4 * (ya * ya - k * yb * yb)) % p
            for xb in range(p):
                pt = (xa, xb, ya, yb)
                idx = len(u_points)
                u_points.append(pt)
                u_indices[pt] = idx
                
    num_u = len(u_points) # Should be 343
    
    # Map all points in F_q^2 to indices
    all_points = []
    point_to_idx = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = (xa, xb, ya, yb)
                    idx = len(all_points)
                    all_points.append(pt)
                    point_to_idx[pt] = idx
                    
    # Identify points NOT in U
    u_set = set(u_points)
    ext_points = [pt for pt in all_points if pt not in u_set]
    ext_indices = [point_to_idx[pt] for pt in ext_points]
    
    # For each point in U, compute the tangent line and the set of external points it covers
    # Tangent at P=(xa0, xb0, ya0, yb0):
    # x_a + ya0*y_a - k*yb0*y_b = 4*(ya0^2 - k*yb0^2)
    # Note: x_b is not in the equation, so the line is parallel to x_b axis?
    # No, the equation is linear in xa, ya, yb. xb is free.
    # So the line is defined by xa + ya0*ya - k*yb0*yb = C.
    # For a fixed (ya, yb), xa is determined. xb is free.
    # So the line contains p points for each valid (ya, yb) pair?
    # No, the equation constrains xa, ya, yb.
    # Let's check the number of solutions.
    # xa = C - ya0*ya + k*yb0*yb.
    # For each (ya, yb), there is 1 xa.
    # xb is free (7 choices).
    # So there are 7*7 = 49 points on the line.
    
    covers = []
    for i in range(num_u):
        xa0, xb0, ya0, yb0 = u_points[i]
        C = (4 * (ya0 * ya0 - k * yb0 * yb0)) % p
        
        covered_ext = set()
        for ya in range(p):
            for yb in range(p):
                xa = (C - ya0 * ya + k * yb0 * yb) % p
                # Check if this point is in U
                # If it is, it's the point of tangency (or another point if secant, but tangent intersects only at P)
                # We only care about external points
                for xb in range(p):
                    pt = (xa, xb, ya, yb)
                    if pt not in u_set:
                        covered_ext.add(point_to_idx[pt])
                        
        covers.append(covered_ext)
        
    # Greedy Set Cover
    # Universe: set of indices of external points
    universe = set(ext_indices)
    covered = set()
    B_indices = set() # Indices in u_points to keep in N
    
    while covered != universe:
        best_idx = -1
        best_gain = -1
        for i in range(num_u):
            if i in B_indices:
                continue
            gain = len(covers[i] - covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        B_indices.add(best_idx)
        covered.update(covers[best_idx])
        
    # Construct S and N
    # S = U \ B
    # N = F_q^2 \ S = (F_q^2 \ U) U B
    
    s_set = set()
    for i in range(num_u):
        if i not in B_indices:
            s_set.add(u_points[i])
            
    n_points = []
    for pt in all_points:
        if pt not in s_set:
            n_points.append([[pt[0], pt[1]], [pt[2], pt[3]]])
            
    return np.array(n_points, dtype=np.int64)
