import time
import random
import itertools

_START = time.time()
_DEADLINE = 140  # seconds, leave margin for the 150s limit

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    """Check if graph has a K4. Returns the K4 vertices if found, else None."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        return [i, j, k, l]
    return None

def _has_ind_6(adj, n):
    """Check if graph has an independent set of size 6. Returns the set if found, else None."""
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        # Pick vertex with fewest neighbors in candidates
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            cnt = (adj[v] & candidates).bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return None
        
        candidates &= ~(1 << best_v)
        
        # Branch 1: Include best_v
        new_candidates = candidates & ~adj[best_v]
        res = backtrack(new_candidates, current_set + [best_v])
        if res:
            return res
        
        # Branch 2: Exclude best_v
        res = backtrack(candidates, current_set)
        if res:
            return res
        
        return None

    return backtrack((1 << n) - 1, [])

def _count_k4s(adj, n):
    """Count the number of K4s in the graph."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        count += 1
    return count

def _count_ind_6s(adj, n):
    """Count the number of independent sets of size 6 in the graph."""
    def backtrack(candidates, current_set, count):
        if len(current_set) == 6:
            return count + 1
        if candidates.bit_count() < 6 - len(current_set):
            return count
        
        # Pick vertex with fewest neighbors in candidates
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            cnt = (adj[v] & candidates).bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return count
        
        candidates &= ~(1 << best_v)
        
        # Branch 1: Include best_v
        new_candidates = candidates & ~adj[best_v]
        count = backtrack(new_candidates, current_set + [best_v], count)
        
        # Branch 2: Exclude best_v
        count = backtrack(candidates, current_set, count)
        
        return count

    return backtrack((1 << n) - 1, [], 0)

def _paley_edges(p):
    """Generate edges of the Paley graph of order p."""
    squares = {x * x % p for x in range(1, p)}
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return edges

def _adj_from_edges(n, edges):
    """Build adjacency bitmasks from edge list."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _edges_from_adj(n, adj):
    """Build edge list from adjacency bitmasks."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _try_local_search(n, time_limit, seed=None):
    """Try to find a valid graph on n vertices using targeted local search."""
    if seed is not None:
        random.seed(seed)
    
    start = time.time()
    
    # Initialize with random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_adj = adj[:]
    best_k4s = _count_k4s(adj, n)
    best_ind6s = _count_ind_6s(adj, n)
    best_score = best_k4s * 1000 + best_ind6s
    
    while time.time() - start < time_limit:
        # Check for K4
        k4 = _has_k4(adj, n)
        if k4:
            # Remove a random edge from the K4
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        # Check for Ind 6
        ind6 = _has_ind_6(adj, n)
        if ind6:
            # Add a random edge between two vertices in the independent set
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        
        # If no K4 and no Ind 6, we found a valid graph
        return edges_from_adj(n, adj)
    
    return None

def _try_extend_from_paley(base_n, target_n, time_limit):
    """Try to extend a Paley graph to target_n vertices."""
    start = time.time()
    
    # Start with Paley graph
    base_edges = _paley_edges(base_n)
    adj = [0] * target_n
    for u, v in base_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Add new vertices with random connections to base vertices
    for i in range(base_n, target_n):
        for j in range(i):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    while time.time() - start < time_limit:
        # Check for K4
        k4 = _has_k4(adj, target_n)
        if k4:
            # Remove a random edge from the K4
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        # Check for Ind 6
        ind6 = _has_ind_6(adj, target_n)
        if ind6:
            # Add a random edge between two vertices in the independent set
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        
        # If no K4 and no Ind 6, we found a valid graph
        return _edges_from_adj(target_n, adj)
    
    return None

def construct():
    # Base case: Paley 17
    p = 17
    best_edges = _paley_edges(p)
    best_n = p
    
    # Try to find larger graphs
    # We iterate from 18 up to 35
    # We allocate time based on remaining budget
    
    # To maximize the chance of finding a large n, we should prioritize larger n?
    # No, if we can't find 35, we might find 34. 
    # If we iterate 18->35, we might waste time on 18.
    # But 18 is easy. 35 is hard.
    # Let's try a few large ones first?
    # Actually, the inspirations scored 1.41 (n=24). 
    # Let's just iterate 18..35 and give each a time slice.
    
    # Total time ~140s.
    # 18 to 35 is 18 values.
    # 140 / 18 ~ 7.8s each.
    
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 2:
            break
        
        # Give more time to larger n? Or uniform?
        # Larger n is harder, so maybe give more time?
        # But if we fail on 35, we want to have tried 34.
        # Let's give uniform time slices, but shrink if time is running out.
        
        # Remaining time
        rem = _DEADLINE - _elapsed()
        # Number of steps left (including current)
        steps_left = 36 - n
        
        # Heuristic: allocate proportional to remaining steps, but with a cap
        # to ensure we don't spend too much on one n
        time_slice = min(12.0, rem / steps_left)
        
        # Ensure at least 0.5s if possible
        if time_slice < 0.5:
            time_slice = 0.5
            
        # Try local search with a few different seeds
        found = False
        for seed in [None, 42, 123, 456]:
            if _elapsed() > _DEADLINE - 2:
                break
            if found:
                break
            edges = _try_local_search(n, time_slice / 4, seed)
            if edges is not None:
                best_n = n
                best_edges = edges
                found = True
        
        # If not found, try extending from Paley
        if not found and _elapsed() < _DEADLINE - 2:
            edges = _try_extend_from_paley(17, n, time_slice / 2)
            if edges is not None:
                best_n = n
                best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}
