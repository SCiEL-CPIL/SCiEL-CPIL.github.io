import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    num_points = q * q
    num_directions = q + 1 # 50
    
    k = 3
    
    def add_f49(a, b, c, d):
        return ((a + c) % p, (b + d) % p)
    
    def mul_f49(a, b, c, d):
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)
        
    def get_idx(xa, xb, ya, yb):
        return (xa * p + xb) * q + (ya * p + yb)
        
    def get_coords(idx):
        y_part = idx % q
        x_part = idx // q
        yb = y_part % p
        ya = y_part // p
        xb = x_part % p
        xa = x_part // p
        return xa, xb, ya, yb

    # Precompute direction indices
    # dir_idx[i][j] = direction from i to j
    # If i == j, undefined (0)
    
    start_time = time.time()
    
    # Map direction (ma, mb) to index
    # (0,1) -> 49
    # (ma, mb) -> ma * 7 + mb
    
    def dir_to_idx(ma, mb):
        if ma == 0 and mb == 1:
            return 49
        return ma * 7 + mb

    # Precompute coords for all points
    coords = [get_coords(i) for i in range(num_points)]
    
    # Precompute direction matrix
    # Use numpy for speed?
    # Or just a list of lists.
    dir_matrix = [[0]*num_points for _ in range(num_points)]
    
    for i in range(num_points):
        xi_a, xi_b, yi_a, yi_b = coords[i]
        for j in range(num_points):
            if i == j: continue
            xj_a, xj_b, yj_a, yj_b = coords[j]
            
            dx_a = (xj_a - xi_a) % p
            dx_b = (xj_b - xi_b) % p
            dy_a = (yj_a - yi_a) % p
            dy_b = (yj_b - yi_b) % p
            
            if dx_a == 0 and dx_b == 0:
                d_idx = 49
            else:
                den = (dx_a*dx_a - dx_b*dx_b*k) % p
                den_inv = pow(den, p-2, p)
                inv_a = (dx_a * den_inv) % p
                inv_b = (-dx_b * den_inv) % p
                m_a, m_b = mul_f49(dy_a, dy_b, inv_a, inv_b)
                d_idx = dir_to_idx(m_a, m_b)
                
            dir_matrix[i][j] = d_idx
            # Symmetry: direction from j to i is same?
            # Slope m. From i to j: (y_j - y_i)/(x_j - x_i).
            # From j to i: (y_i - y_j)/(x_i - x_j) = (-dy)/(-dx) = dy/dx.
            # Yes, same direction index.
            dir_matrix[j][i] = d_idx

    # Greedy Addition
    best_C = set()
    best_score = 0
    
    points_list = list(range(num_points))
    
    # We can try multiple random orders
    for restart in range(20):
        if time.time() > start_time + 580:
            break
            
        random.shuffle(points_list)
        C = set()
        # dir_counts[i][d] = count of points in C in direction d from i
        dir_counts = [[0]*num_directions for _ in range(num_points)]
        
        for x in points_list:
            # Check if x is blocked
            # x is blocked if min(dir_counts[x]) > 0
            if min(dir_counts[x]) > 0:
                continue
                
            # Check if adding x blocks any y in C
            can_add = True
            for y in C:
                d_idx = dir_matrix[y][x]
                if dir_counts[y][d_idx] == 0:
                    # This direction was empty, now it will have 1.
                    # Check if y becomes blocked (all counts > 0)
                    if min(dir_counts[y]) > 0:
                        can_add = False
                        break
            
            if can_add:
                C.add(x)
                for y in C:
                    if y == x: continue
                    d_idx = dir_matrix[y][x]
                    dir_counts[y][d_idx] += 1
                
                if len(C) > best_score:
                    best_score = len(C)
                    best_C = set(C)
                    
    # Construct N
    n_points = []
    for i in range(num_points):
        if i not in best_C:
            xa, xb, ya, yb = coords[i]
            n_points.append([[xa, xb], [ya, yb]])
            
    if not n_points:
        n_points = [[[0, 0], [0, 0]]]
        
    return np.array(n_points, dtype=np.int64)

