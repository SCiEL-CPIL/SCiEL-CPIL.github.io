import numpy as np

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3 # alpha^2 = 3
    
    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0b0 + a1b1*alpha^2 + (a0b1 + a1b0)*alpha
        # alpha^2 = k
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + a1*b1*k) % p, (a0*b1 + a1*b0) % p)
        
    def inv(a):
        # Inverse in F_q
        # a = (a0, a1). a * a^p = a * (a0, -a1) = a0^2 + a1^2*k
        # inv(a) = a^p / (a0^2 + a1^2*k)
        a0, a1 = a
        norm = (a0*a0 + a1*a1*k) % p
        if norm == 0:
            return None
        norm_inv = pow(norm, p-2, p)
        a_p = (a0, (-a1) % p)
        return mul(a_p, (norm_inv, 0))

    # Generate all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
                    
    # Map point to index
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Initial complement C from current solution
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    C_indices = set()
    for i, pt in enumerate(all_points):
        (xa, xb), (ya, yb) = pt
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            C_indices.add(i)
            
    C = list(C_indices)
    
    # Helper: Check if C is a valid Nikodym complement
    # For every y in C, there exists a line through y such that line \ {y} is disjoint from C
    def is_valid_complement(C_set):
        # Precompute lines? 
        # For each point y in C, check lines through y.
        for y_idx in C_set:
            y = all_points[y_idx]
            found_line = False
            # Directions: (1, m) for m in F_q, and (0, 1)
            # F_q elements
            F_q_elems = [(a, b) for a in range(p) for b in range(p)]
            
            for v in F_q_elems:
                # Line: y + t*v
                # Check if all other points are NOT in C_set
                ok = True
                for t in F_q_elems:
                    if t == (0, 0): continue
                    # x = y + t*v
                    tx, ty = t
                    vx, vy = v
                    # t*v
                    tv = mul(t, v)
                    # y + tv
                    x = add(y, tv)
                    if x == y: continue # t=0 handled
                    x_idx = point_to_idx[x]
                    if x_idx in C_set:
                        ok = False
                        break
                if ok:
                    found_line = True
                    break
            if not found_line:
                return False
        return True
        
    # Try to grow C
    # This is very slow. 175 points * 50 lines * 49 points * 2401 candidates...
    # Too slow.
    
    # Just return the initial C
    points_in_N = []
    for i, pt in enumerate(all_points):
        if i not in C_indices:
            (xa, xb), (ya, yb) = pt
            points_in_N.append([[xa, xb], [ya, yb]])
            
    return np.array(points_in_N, dtype=np.int64)
