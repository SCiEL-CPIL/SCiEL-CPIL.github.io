import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try n from 22 down to 20
    # If we find a valid one, return it.
    # If time runs out, return the seed.
    
    for n in [22, 21, 20]:
        if time.time() > deadline:
            break
            
        # Try multiple random graphs
        # We limit attempts to ensure we don't get stuck in one n for too long
        # if valid graphs are rare.
        # But we also want to give it a chance.
        
        for attempt in range(500):
            if time.time() > deadline:
                break
                
            random.seed(attempt * 1000 + n)
            
            adj = [0] * n
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        edges.append([i, j])
                        
            if is_valid_graph(adj, n):
                return {'n': n, 'edges': edges}
                
    return {'n': 5, 'edges': []}

def is_valid_graph(adj, n):
    def is_regular_subset(mask, size):
        if size < 6:
            return False
        
        first_v = None
        m = mask
        while m:
            lsb = m & -m
            first_v = lsb.bit_length() - 1
            break
            
        if first_v is None:
            return False
            
        target_deg = (adj[first_v] & mask).bit_count()
        
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if (adj[v] & mask).bit_count() != target_deg:
                return False
            m ^= lsb
            
        return True

    for k in range(6, n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if is_regular_subset(mask, k):
                return False
    return True
