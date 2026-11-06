import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s
    
    # 255 = 3 * 5 * 17
    # Candidates: x in [1, 255] with gcd(x, 255) > 1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    idx_255 = candidates.index(255)
    n = len(candidates)
    
    # Adjacency matrix as bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Initial clique: all multiples of 3
    init_clique = set()
    for i in range(n):
        if candidates[i] % 3 == 0:
            init_clique.add(i)
            
    best_size = len(init_clique)
    best_set = init_clique.copy()
    
    # Bron-Kerbosch with pivot
    def popcount(x):
        return x.bit_count()
        
    def bron_kerbosch(clique, candidates_mask):
        nonlocal best_size, best_set
        
        if time.time() > DEADLINE:
            return
            
        if popcount(candidates_mask) + len(clique) <= best_size:
            return
            
        if candidates_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_set = clique.copy()
            return
            
        # Pivot selection
        pivot = -1
        max_deg = -1
        temp = candidates_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            deg = popcount(adj[v] & candidates_mask)
            if deg > max_deg:
                max_deg = deg
                pivot = v
                
        # Branch on non-neighbors of pivot
        branch_mask = candidates_mask & ~(adj[pivot] if pivot >= 0 else 0)
        
        temp = branch_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            
            new_clique = clique | {v}
            new_candidates = candidates_mask & adj[v]
            
            bron_kerbosch(new_clique, new_candidates)
            
            candidates_mask &= ~(1 << v)
            
            if time.time() > DEADLINE:
                return

    # Start search
    init_clique_set = {idx_255}
    init_candidates = adj[idx_255]
    
    bron_kerbosch(init_clique_set, init_candidates)
    
    # Local search polish (optional, if time permits)
    if time.time() < DEADLINE:
        current_clique = best_set.copy()
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(n):
                if i in current_clique:
                    continue
                can_add = True
                for j in current_clique:
                    if not (adj[i] & (1 << j)):
                        can_add = False
                        break
                if can_add:
                    current_clique.add(i)
                    improved = True
        if len(current_clique) > best_size:
            best_size = len(current_clique)
            best_set = current_clique

    result = [candidates[i] for i in best_set]
    result.sort()
    
    if 255 not in result:
        result.append(255)
        
    return result
