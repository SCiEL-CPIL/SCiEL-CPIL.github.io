import numpy as np
import time
import sys

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by building the complement S greedily.
    S is the set of points NOT in N. We want to maximize |S|.
    Condition: for every x in F_49^2, there exists a direction v such that
    the line L(x, v) \\ {x} is disjoint from S.
    """
    p = 7
    q = p * p
    k_const = 3  # alpha^2 = 3 in F_7
    
    # Field operations in F_49 = F_7[alpha] where alpha^2 = 3
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul_f(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k_const) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def neg_f(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    # Generate all points in F_49^2
    # Point index: x_idx * q + y_idx where x_idx = xa*p + xb, y_idx = ya*p + yb
    num_points = q * q  # 2401
    
    # Precompute point coordinates
    point_coords = np.zeros((num_points, 4), dtype=np.int64)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = xa * p + xb
                    y_idx = ya * p + yb
                    pt_idx = x_idx * q + y_idx
                    point_coords[pt_idx] = [xa, xb, ya, yb]
    
    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # m represented as (ma, mb), index ma*p + mb
    num_dirs = q + 1  # 50
    
    def dir_to_coords(d_idx):
        if d_idx < q:
            ma = d_idx // p
            mb = d_idx % p
            return ((1, 0), (ma, mb))
        else:
            return ((0, 0), (0, 1))
    
    # Precompute line membership: for each point and direction, 
    # store the other 48 points on that line
    # line_members[pt_idx][d_idx] = list of other point indices (48 elements)
    line_members = [None] * (num_points * num_dirs)
    
    for pt_idx in range(num_points):
        xa, xb, ya, yb = point_coords[pt_idx]
        pt = ((xa, xb), (ya, yb))
        for d_idx in range(num_dirs):
            vx, vy = dir_to_coords(d_idx)
            members = []
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    t = (ta, tb)
                    # x = pt + t * v
                    tvx = mul_f(t, vx)
                    tvy = mul_f(t, vy)
                    nx = add_f(pt[0], tvx)
                    ny = add_f(pt[1], tvy)
                    n_xa, n_xb = nx
                    n_ya, n_yb = ny
                    n_idx = (n_xa * p + n_xb) * q + (n_ya * p + n_yb)
                    members.append(n_idx)
            line_members[pt_idx * num_dirs + d_idx] = members
    
    # Precompute reverse adjacency: for each point x, list of (y, d) where x is on line(y, d)
    # rev_adj[x_idx] = list of (y_idx, d_idx)
    rev_adj = [[] for _ in range(num_points)]
    for y_idx in range(num_points):
        for d_idx in range(num_dirs):
            for x_idx in line_members[y_idx * num_dirs + d_idx]:
                rev_adj[x_idx].append((y_idx, d_idx))
    
    # Greedy construction of S
    start_time = time.time()
    timeout = 580  # seconds, leave margin for 600s limit
    
    S = set()
    # safe_count[y] = number of directions d where line(y,d) \ {y} is disjoint from S
    safe_count = [num_dirs] * num_points
    
    # Use multiple random orderings and keep the best
    best_S_size = 0
    best_S = None
    
    np.random.seed(42)
    
    for trial in range(50):
        if time.time() - start_time > timeout - 30:
            break
        
        # Reset
        S.clear()
        for i in range(num_points):
            safe_count[i] = num_dirs
        
        # Random candidate order
        candidates = np.random.permutation(num_points)
        
        for x_idx in candidates:
            if time.time() - start_time > timeout:
                break
            
            # Check if x has a safe direction
            if safe_count[x_idx] == 0:
                continue
            
            # Check if adding x breaks any existing point in S
            conflict = False
            affected = []
            
            for y_idx, d_idx in rev_adj[x_idx]:
                if y_idx in S:
                    safe_count[y_idx] -= 1
                    affected.append(y_idx)
                    if safe_count[y_idx] == 0:
                        conflict = True
                        break
            
            if conflict:
                # Revert
                for y_idx in affected:
                    safe_count[y_idx] += 1
            else:
                S.add(x_idx)
        
        if len(S) > best_S_size:
            best_S_size = len(S)
            best_S = set(S)
        
        # Check if we've found a large enough S
        # Theoretically max |S| is bounded, but let's see what we get
        if best_S_size >= 100:  # If we get 100+ excluded, N has 2301 points
            # Try to do more trials with different strategies
            pass
    
    # Use the best S found
    if best_S is None:
        best_S = set()
    
    # Construct N = F_49^2 \ S
    n_points = []
    for i in range(num_points):
        if i not in best_S:
            xa, xb, ya, yb = point_coords[i]
            n_points.append([xa, xb, ya, yb])
    
    if not n_points:
        # Fallback: return all points
        result = np.zeros((num_points, 2, 2), dtype=np.int64)
        for i in range(num_points):
            xa, xb, ya, yb = point_coords[i]
            result[i] = [[xa, xb], [ya, yb]]
        return result
    
    result = np.array(n_points, dtype=np.int64).reshape(-1, 2, 2)
    
    # Final deadline check
    if time.time() - start_time > timeout:
        pass  # We're already returning, this is just a note
    
    return result
