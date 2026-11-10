import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field operations for F_49 = F_7[alpha] with alpha^2 = 3
    k = 3  # alpha^2 = 3
    
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def sub_f(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul_f(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0b0 + a1b1*k) + (a0b1 + a1b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def div_f(a, b):
        # a / b = a * b^{-1}
        # b = b0 + b1*alpha
        # b * (b0 - b1*alpha) = b0^2 - b1^2*k
        inv_denom = (b[0]*b[0] - b[1]*b[1]*k) % p
        if inv_denom == 0:
            return None
        inv_denom_inv = pow(inv_denom, p-2, p)
        # b^{-1} = (b0 - b1*alpha) / (b0^2 - b1^2*k)
        inv_b = ((b[0] * inv_denom_inv) % p, (-b[1] * inv_denom_inv) % p)
        return mul_f(a, inv_b)
        
    # Generate all points in F_49^2
    # Point is [[x_a, x_b], [y_a, y_b]]
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append([xa, xb, ya, yb])
                    
    n_points = len(all_points)
    
    # Define the excluded set S as the line y = 0
    # Points on y=0 are those with y_a=0 and y_b=0
    excluded_indices = set()
    for i, pt in enumerate(all_points):
        # pt = [xa, xb, ya, yb]
        if pt[2] == 0 and pt[3] == 0:
            excluded_indices.add(i)
            
    # The Nikodym set N is the complement of S
    N_indices = [i for i in range(n_points) if i not in excluded_indices]
    
    # Verify the Nikodym condition
    # For every point x in F_49^2, there exists a direction v such that
    # all points on the line x + t*v (t in F_49) except x itself are in N.
    
    # Directions: v = (1, m) for m in F_49, and v = (0, 1) for infinity
    # m is represented as (ma, mb)
    
    def point_to_idx(xa, xb, ya, yb):
        # Convert field element coordinates to index
        # Index = ((xa * p + xb) * p + ya) * p + yb
        return ((xa * p + xb) * p + ya) * p + yb
        
    def idx_to_point(idx):
        yb = idx % p
        idx //= p
        ya = idx % p
        idx //= p
        xb = idx % p
        idx //= p
        xa = idx % p
        return (xa, xb, ya, yb)
        
    # Check if a point index is in N
    def in_N(idx):
        return idx not in excluded_indices
        
    # Verify each point
    for x_idx in range(n_points):
        xa, xb, ya, yb = idx_to_point(x_idx)
        x_pt = ((xa, xb), (ya, yb))
        
        found_valid_direction = False
        
        # Try all directions
        # Direction (1, m) for m in F_49
        for ma in range(p):
            for mb in range(p):
                m = (ma, mb)
                v = ((1, 0), m)  # v = (1, m)
                
                # Check all t in F_49
                valid_line = True
                for ta in range(p):
                    for tb in range(p):
                        t = (ta, tb)
                        if ta == 0 and tb == 0:
                            continue  # Skip t=0 (the point x itself)
                        
                        # Compute x + t*v
                        # t*v = (t*1, t*m) = (t, t*m)
                        tdx = t  # t * (1,0) = (ta, tb)
                        tdy = mul_f(t, m)
                        
                        # new point = x + t*v
                        new_xa = (xa + tdx[0]) % p
                        new_xb = (xb + tdx[1]) % p
                        new_ya = (ya + tdy[0]) % p
                        new_yb = (yb + tdy[1]) % p
                        
                        new_idx = point_to_idx(new_xa, new_xb, new_ya, new_yb)
                        
                        if not in_N(new_idx):
                            valid_line = False
                            break
                    if not valid_line:
                        break
                
                if valid_line:
                    found_valid_direction = True
                    break
            if found_valid_direction:
                break
        
        if found_valid_direction:
            continue
        
        # Try direction (0, 1) for infinity
        v = ((0, 0), (0, 1))  # v = (0, 1)
        
        # Check all t in F_49
        valid_line = True
        for ta in range(p):
            for tb in range(p):
                t = (ta, tb)
                if ta == 0 and tb == 0:
                    continue
                
                # t*v = (t*0, t*1) = (0, t)
                tdx = (0, 0)
                tdy = t
                
                new_xa = (xa + tdx[0]) % p
                new_xb = (xb + tdx[1]) % p
                new_ya = (ya + tdy[0]) % p
                new_yb = (yb + tdy[1]) % p
                
                new_idx = point_to_idx(new_xa, new_xb, new_ya, new_yb)
                
                if not in_N(new_idx):
                    valid_line = False
                    break
            if not valid_line:
                break
        
        if valid_line:
            found_valid_direction = True
        
        if not found_valid_direction:
            # This should not happen for a valid Nikodym set
            raise ValueError(f"Point {x_idx} does not satisfy Nikodym condition")
    
    # Construct the output array
    N = np.array([all_points[i] for i in N_indices], dtype=np.int64)
    
    return N
