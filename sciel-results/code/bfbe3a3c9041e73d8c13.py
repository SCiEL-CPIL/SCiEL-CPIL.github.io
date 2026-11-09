import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    n_points = q * q
    
    def idx_to_elem(i):
        return (i // 7, i % 7)
    
    def elem_to_idx(e):
        return e[0] * 7 + e[1]
    
    def f_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def f_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def f_mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*3) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def f_inv(a):
        d = (a[0]*a[0] - 3*a[1]*a[1]) % p
        if d == 0:
            return (0, 0)
        d_inv = pow(d, -1, p)
        return ((a[0]*d_inv) % p, (-a[1]*d_inv) % p)

    dir_matrix = np.zeros((n_points, n_points), dtype=np.int8)
    
    for x_idx in range(n_points):
        xa = x_idx // 343
        xb = (x_idx // 49) % 7
        ya = (x_idx // 7) % 7
        yb = x_idx % 7
        x_elem = ((xa, xb), (ya, yb))
        
        for y_idx in range(n_points):
            if x_idx == y_idx:
                dir_matrix[x_idx][y_idx] = -1
                continue
            ya2 = y_idx // 343
            yb2 = (y_idx // 49) % 7
            za = (y_idx // 7) % 7
            zb = y_idx % 7
            y_elem = ((ya2, yb2), (za, zb))
            
            dx = f_sub(y_elem[0], x_elem[0])
            dy = f_sub(y_elem[1], x_elem[1])
            
            if dx[0] == 0 and dx[1] == 0:
                dir_matrix[x_idx][y_idx] = 49
            else:
                dx_inv = f_inv(dx)
                m = f_mul(dy, dx_inv)
                dir_matrix[x_idx][y_idx] = elem_to_idx(m)

    S = set()
    B = np.zeros(n_points, dtype=np.int64)
    counts = np.zeros(n_points, dtype=np.int32)
    Crit = []
    
    rng = np.random.default_rng(42)
    order = rng.permutation(n_points)
    
    start_time = time.time()
    deadline = start_time + 580
    
    for p in order:
        if time.time() > deadline:
            break
            
        if p in S:
            continue
            
        if counts[p] == 50:
            continue
            
        valid = True
        for x in Crit:
            v = dir_matrix[x][p]
            if v >= 0:
                if (B[x] >> v) & 1 == 0:
                    valid = False
                    break
            else:
                valid = False
                break
                
        if not valid:
            continue
            
        S.add(p)
        
        for x in range(n_points):
            if x == p:
                continue
            v = dir_matrix[x][p]
            if v >= 0:
                if (B[x] >> v) & 1 == 0:
                    B[x] |= (1 << v)
                    counts[x] += 1
                    if counts[x] == 49:
                        Crit.append(x)

    N_indices = [i for i in range(n_points) if i not in S]
    
    if len(N_indices) == 0:
        return np.array([[[0, 0], [0, 0]]], dtype=np.int64)
        
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        xa = i // 343
        xb = (i // 49) % 7
        ya = (i // 7) % 7
        yb = i % 7
        result[idx, 0, 0] = xa
        result[idx, 0, 1] = xb
        result[idx, 1, 0] = ya
        result[idx, 1, 1] = yb
        
    return result
