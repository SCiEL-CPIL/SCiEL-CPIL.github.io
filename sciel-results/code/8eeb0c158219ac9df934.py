import random
import time
from itertools import combinations

def construct():
    n = 18
    start_time = time.time()
    time_limit = 100  # Leave buffer for 120s limit
    
    def max_tf_size_fast(adj_mask):
        """Fast check: return largest k in [12,11,10,9,8] such that 
        there exists a triangle-free induced subgraph of size k.
        If none, return 7."""
        for size in [12, 11, 10, 9, 8]:
            for combo in combinations(range(n), size):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                tf = True
                for i in range(len(combo)):
                    u = combo[i]
                    for j in range(i+1, len(combo)):
                        v = combo[j]
                        if adj_mask[u] & (1 << v):
                            if adj_mask[u] & adj_mask[v] & mask:
                                tf = False
                                break
                    if not tf:
                        break
                
                if tf:
                    return size
        
        return 7
    
    def is_k4_free_check(adj_mask, u, v):
        common = adj_mask[u] & adj_mask[v]
        bits = []
        m = common
        while m:
            lsb = m & (-m)
            w = lsb.bit_length() - 1
            bits.append(w)
            m ^= lsb
        
        for i in range(len(bits)):
            w = bits[i]
            for j in range(i+1, len(bits)):
                x = bits[j]
                if adj_mask[w] & (1 << x):
                    return False
        return True
    
    def random_k4_free():
        adj_mask = [0] * n
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if is_k4_free_check(adj_mask, u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        return adj_mask
    
    best_score = 18
    best_adj_mask = None
    
    for trial in range(50):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(trial * 999983 + 7)
        adj_mask = random_k4_free()
        
        for iteration in range(15):
            if time.time() - start_time > time_limit:
                break
            
            current_score = max_tf_size_fast(adj_mask)
            if current_score <= 7:
                break
            
            best_adj = adj_mask[:]
            best_sc = current_score
            
            missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                      if not (adj_mask[i] & (1 << j))]
            random.shuffle(missing)
            
            for u, v in missing[:25]:
                if time.time() - start_time > time_limit:
                    break
                if is_k4_free_check(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    sc = max_tf_size_fast(adj_mask)
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    
                    if sc < best_sc:
                        best_sc = sc
                        best_adj = adj_mask[:]
            
            adj_mask = best_adj
        
        final_score = max_tf_size_fast(adj_mask)
        if final_score < best_score:
            best_score = final_score
            best_adj_mask = adj_mask[:]
    
    if best_adj_mask is None:
        best_adj_mask = [0] * n
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj_mask[i] & (1 << j):
                edges.append([i, j])
    
    return edges
