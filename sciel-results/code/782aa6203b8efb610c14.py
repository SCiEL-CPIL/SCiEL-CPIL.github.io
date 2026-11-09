import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def has_c4(n, adj):
        for u in range(n):
            for v in range(u+1, n):
                if len(adj[u] & adj[v]) >= 2:
                    return True
        return False

    def has_independent_set_size_k(n, adj, k):
        """Backtracking with pruning to check if independent set of size k exists."""
        # Sort vertices by degree (ascending) for better pruning
        order = sorted(range(n), key=lambda v: len(adj[v]))
        # Build a bitset representation for speed
        # adj_bits[v] is a bitmask of neighbors of v
        adj_bits = [0] * n
        for v in range(n):
            mask = 0
            for u in adj[v]:
                mask |= (1 << u)
            adj_bits[v] = mask
        
        all_mask = (1 << n) - 1
        
        def backtrack(available_mask, chosen_count):
            if chosen_count == k:
                return True
            if available_mask == 0:
                return False
            # Pruning: count bits in available_mask
            # Python 3.10+ has int.bit_count()
            if available_mask.bit_count() < k - chosen_count:
                return False
            
            # Pick a vertex with min degree in available subgraph
            best_v = -1
            best_deg = n + 1
            temp_mask = available_mask
            while temp_mask:
                lsb = temp_mask & (-temp_mask)
                v = lsb.bit_length() - 1
                deg = (adj_bits[v] & available_mask).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp_mask ^= lsb
            
            if best_v == -1:
                return False
            
            # Branch 1: include best_v
            new_available = available_mask & ~(adj_bits[best_v] | (1 << best_v))
            if backtrack(new_available, chosen_count + 1):
                return True
            
            # Branch 2: exclude best_v
            new_available2 = available_mask & ~(1 << best_v)
            if backtrack(new_available2, chosen_count):
                return True
            
            return False
        
        return backtrack(all_mask, 0)

    def can_add_edge(n, adj, adj_bits, u, v):
        # Adding (u,v) creates C4 iff exists w~u, x~v with w~x, w!=v, x!=u
        # Use bitsets
        nu = adj_bits[u] & ~(1 << v)
        nv = adj_bits[v] & ~(1 << u)
        # For each w in N(u)\{v}, check if N(w) intersects N(v)\{u}
        temp = nu
        while temp:
            lsb = temp & (-temp)
            w = lsb.bit_length() - 1
            if adj_bits[w] & nv:
                return False
            temp ^= lsb
        return True

    n = 43
    random.seed(42)
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        adj = [set() for _ in range(n)]
        adj_bits = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if can_add_edge(n, adj, adj_bits, u, v):
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        
        if has_c4(n, adj):
            continue
        
        if not has_independent_set_size_k(n, adj, 11):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    # Fallback to 39
    n_target = 39
    edges = []
    for i in range(5):
        for j in range(10):
            u = i * 10 + j
            v = i * 10 + ((j + 1) % 10)
            if u > v:
                u, v = v, u
            if u < n_target and v < n_target:
                edges.append([u, v])
    offsets = [1, 2, 4, 8, 6]
    for i in range(5):
        next_i = (i + 1) % 5
        d = offsets[i]
        for j in range(10):
            u = i * 10 + j
            v = next_i * 10 + ((j + d) % 10)
            if u > v:
                u, v = v, u
            if u < n_target and v < n_target:
                edges.append([u, v])
    return {'n': n_target, 'edges': edges}
