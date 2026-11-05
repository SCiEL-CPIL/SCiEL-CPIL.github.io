import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_addition(neighbors, u, v):
        # Check if adding u-v keeps max common neighbors <= 2
        cn_uv = neighbors[u] & neighbors[v]
        if popcount(cn_uv) > 2:
            return False
            
        x = cn_uv
        while x:
            lsb = x & -x
            idx = lsb.bit_length() - 1
            x ^= lsb
            
            # Check pair (v, idx)
            if popcount(neighbors[v] & neighbors[idx]) >= 2:
                return False
                
            # Check pair (u, idx)
            if popcount(neighbors[u] & neighbors[idx]) >= 2:
                return False
                
        return True

    best_count = 0
    best_edges = []
    
    # Run multiple random restarts
    for restart in range(200): # Try many restarts
        if time.time() > deadline:
            break
            
        neighbors = [0] * n
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_list)
        
        current_count = 0
        current_edges = []
        
        for u, v in edges_list:
            if is_valid_addition(neighbors, u, v):
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                current_count += 1
                current_edges.append((u, v))
                
        if current_count > best_count:
            best_count = current_count
            best_edges = current_edges[:]
            
        # If we reach a high bar, we can stop
        if best_count >= 75: 
            break

    return [[u, v] for u, v in best_edges]
