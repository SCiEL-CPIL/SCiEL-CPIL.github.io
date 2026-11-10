import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 256
    # 1. Identify valid nodes (no self-loop)
    squares = set()
    for k in range(1, 23):
        squares.add(k*k)
        
    valid_vals = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vals.append(v)
            
    M = len(valid_vals)
    if M == 0:
        return []
        
    # 2. Build adjacency masks
    adj = [0] * M
    
    sum_squares = set()
    for k in range(2, 23):
        sum_squares.add(k*k)
        
    for i in range(M):
        vi = valid_vals[i]
        mask_i = 0
        for j in range(i + 1, M):
            vj = valid_vals[j]
            s = vi + vj
            if s in sum_squares:
                mask_i |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] = mask_i
        
    best_set = []
    best_size = 0
    
    def greedy_coloring(candidates_mask):
        if candidates_mask == 0:
            return 0, []
        
        available = candidates_mask
        num_colors = 0
        order = []
        
        while available:
            num_colors += 1
            max_deg = -1
            best_v = -1
            m = available
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                deg = (adj[idx] & available).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = idx
                m ^= lsb
            
            if best_v == -1:
                break
                
            order.append((best_v, num_colors))
            available &= ~(adj[best_v] | (1 << best_v))
            
        return num_colors, order

    def expand(R_mask, C_mask, R_size):
        nonlocal best_set, best_size
        
        if C_mask == 0:
            if R_size > best_size:
                best_size = R_size
                s = []
                m = R_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    s.append(valid_vals[idx])
                    m ^= lsb
                best_set = s
            return
        
        if time.time() > deadline:
            return
            
        num_colors, order = greedy_coloring(C_mask)
        
        if R_size + num_colors <= best_size:
            return
            
        for v, c in order:
            if not (C_mask & (1 << v)):
                continue
                
            if R_size + c <= best_size:
                break
                
            new_R_mask = R_mask | (1 << v)
            new_C_mask = C_mask & ~(adj[v] | (1 << v))
            
            expand(new_R_mask, new_C_mask, R_size + 1)
            
            C_mask &= ~(1 << v)
            
            if R_size + C_mask.bit_count() <= best_size:
                break
                
    initial_C = (1 << M) - 1
    expand(0, initial_C, 0)
    
    if not best_set:
        return valid_vals[:1]
        
    return best_set
