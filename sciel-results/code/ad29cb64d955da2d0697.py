import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Target n
    target_n = 39
    
    def popcount(x):
        return x.bit_count()
    
    def _has_c4_bitset(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if popcount(common) >= 2:
                    return True
        return False

    def _find_indep_k_bitset(adj, n, k):
        def search_set(available, current):
            if len(current) == k:
                return current
            if available == 0:
                return None
            if len(current) + popcount(available) < k:
                return None
            
            # Heuristic: pick vertex with max degree in available?
            # Or min?
            # Min degree is better for independent set (less constraint).
            # But calculating degree in available is expensive.
            # Just pick lowest bit.
            v_bit = available & (-available)
            v = v_bit.bit_length() - 1
            
            # Option 1: Include v
            new_available = available & ~(adj[v] | v_bit)
            res = search_set(new_available, current + [v])
            if res:
                return res
            
            # Option 2: Exclude v
            new_available2 = available & ~v_bit
            res = search_set(new_available2, current)
            if res:
                return res
            
            return None
        
        full_mask = (1 << n) - 1
        return search_set(full_mask, [])

    def _try_add_edge(adj, u, v, n):
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        temp = mask_u
        while temp:
            w_bit = temp & (-temp)
            w = w_bit.bit_length() - 1
            temp -= w_bit
            
            if mask_v & w_bit:
                continue
            
            if adj[w] & mask_v:
                return False
        
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    # Try to construct for target_n
    for n in range(target_n, 30, -1):
        if time.time() > deadline - 10:
            break
        
        for seed in range(200):
            if time.time() > deadline - 5:
                break
            
            adj = [0] * n
            rng = random.Random(seed)
            
            # Random initialization
            for _ in range(500):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v: continue
                if u > v: u, v = v, u
                if not (adj[u] & (1 << v)):
                    _try_add_edge(adj, u, v, n)
            
            # Refinement
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                indep = _find_indep_k_bitset(adj, n, 11)
                if indep is None:
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                
                added = False
                indep_list = list(indep)
                rng.shuffle(indep_list)
                
                for i in range(len(indep_list)):
                    for j in range(i + 1, len(indep_list)):
                        u = indep_list[i]
                        v = indep_list[j]
                        if _try_add_edge(adj, u, v, n):
                            added = True
                            break
                    if added:
                        break
                
                if not added:
                    break
    
    # Fallback to 31 if all else fails
    return {'n': 31, 'edges': []}

# EVOLVE-BLOCK-END
