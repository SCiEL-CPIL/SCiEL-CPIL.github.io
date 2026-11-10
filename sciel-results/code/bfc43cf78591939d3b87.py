import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3, non-residue
    
    # Field operations
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def sub_f(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def mul_f(a, b):
        return ((a[0]*b[0] - k*a[1]*b[1]) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def inv_f(a):
        # a = a0 + a1*alpha
        # a * conj(a) = a0^2 - k*a1^2
        # inv = conj(a) / (a0^2 - k*a1^2)
        norm = (a[0]*a[0] - k*a[1]*a[1]) % p
        if norm == 0:
            return (0, 0) # should not happen for non-zero
        norm_inv = pow(norm, p-2, p)
        conj = (a[0], (-a[1]) % p)
        return mul_f(conj, (norm_inv, 0))

    def canon_dir(d):
        # d is (da, db) != (0,0)
        # Make first non-zero component 1
        if d[0] != 0:
            inv_d0 = inv_f((d[0], 0))
            # d * inv_d0
            # (d0, d1) * (d0_inv, 0) = (d0*d0_inv, d1*d0_inv)
            d1_scaled = (d[1] * inv_d0[0]) % p
            return (1, d1_scaled)
        else:
            # d0 == 0, d1 != 0
            return (0, 1)

    # Generate all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
    
    # Map points to indices for faster lookup?
    # Points are tuples of tuples. Hashable.
    
    S = []
    blocked = {} # point -> set of canonical directions
    
    candidates = set(points)
    
    start_time = time.time()
    
    while candidates:
        best_v = None
        best_cost = float('inf')
        
        # Optional: break if time limit approaching
        if time.time() - start_time > 500:
            break
            
        for v in candidates:
            cost = 0
            valid = True
            # Check against all u in S
            for u in S:
                d = sub_f(v, u)
                if d == (0, 0):
                    continue
                dc = canon_dir(d)
                if dc not in blocked[u]:
                    cost += 1
                    if len(blocked[u]) + 1 >= 50: # 50 lines total
                        valid = False
                        break
            
            if valid and cost < best_cost:
                best_cost = cost
                best_v = v
                
        if best_v is None:
            break
            
        # Add best_v to S
        S.append(best_v)
        candidates.remove(best_v)
        
        # Update blocked sets
        # For existing u in S (excluding best_v), add direction to best_v
        for u in S:
            if u == best_v:
                continue
            d = sub_f(best_v, u)
            dc = canon_dir(d)
            blocked[u].add(dc)
            
        # For best_v, calculate its blocked set from all other u in S
        bv_blocked = set()
        for u in S:
            if u == best_v:
                continue
            d = sub_f(best_v, u)
            dc = canon_dir(d)
            bv_blocked.add(dc)
        blocked[best_v] = bv_blocked
        
    # Construct N = complement of S
    S_set = set(S)
    N = []
    for pt in points:
        if pt not in S_set:
            # Convert to array format [[xa, xb], [ya, yb]]
            N.append([[pt[0][0], pt[0][1]], [pt[1][0], pt[1][1]]])
            
    return np.array(N, dtype=np.int64)
