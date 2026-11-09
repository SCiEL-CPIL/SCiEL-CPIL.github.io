import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115

    for target_n in [20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() > deadline - 10:
            break
            
        # Try to find a valid graph for this n
        result = _find_valid_graph(target_n, deadline)
        if result:
            return result

    return {'n': 5, 'edges': []}

def _find_valid_graph(n, deadline):
    # Try multiple restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            return None
            
        # Random init
        adj = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and random.random() < 0.5:
                    mask |= (1 << j)
            adj[i] = mask
        
        # Local search to eliminate bad size-6 subsets
        bad_count = _count_bad_size(adj, n, 6)
        
        # If we already have 0 bad size-6, check full validity
        if bad_count == 0:
            if _is_fully_valid(adj, n):
                return _adj_to_graph(adj, n)
        
        # Local search
        no_improve = 0
        for step in range(500):
            if time.time() > deadline - 5:
                return None
                
            # Pick a random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            # Flip
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_bad_count = _count_bad_size(adj, n, 6)
            
            if new_bad_count <= bad_count:
                bad_count = new_bad_count
                no_improve = 0
                
                # If we reached 0, check if fully valid
                if bad_count == 0:
                    if _is_fully_valid(adj, n):
                        return _adj_to_graph(adj, n)
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                no_improve += 1
                
            if no_improve > 50:
                break

    return None

def _count_bad_size(adj, n, size):
    count = 0
    for subset in combinations(range(n), size):
        if _is_regular_induced_fast(adj, subset):
            count += 1
    return count

def _is_regular_induced_fast(adj, subset):
    s = len(subset)
    if s < 2:
        return True
    
    subset_mask = 0
    for v in subset:
        subset_mask |= (1 << v)
        
    d0 = None
    for v in subset:
        d = (adj[v] & subset_mask).bit_count()
        if d0 is None:
            d0 = d
        elif d != d0:
            return False
    return True

def _is_fully_valid(adj, n):
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_induced_fast(adj, subset):
                return False
    return True

def _adj_to_graph(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
