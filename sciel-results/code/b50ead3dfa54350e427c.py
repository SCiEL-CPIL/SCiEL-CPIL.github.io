import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    def is_triangle_free(adj, n):
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    if common:
                        return False
        return True

    def max_independent_set_size(adj, n, limit=10):
        """Branch and bound to find if independent set of size >= limit exists."""
        # Returns the size of the largest independent set found (up to limit)
        # Or limit if one of size limit is found
        
        best = [0]
        
        def search(current_size, candidates):
            if current_size >= limit:
                return True
            if not candidates:
                return False
            if current_size + candidates.bit_count() <= best[0]:
                return False
            
            # Pick vertex with minimum degree in candidate subgraph (greedy heuristic)
            # Actually for finding large independent sets, pick vertex that removes fewest candidates
            # i.e., vertex with minimum degree in the induced subgraph on candidates
            
            c = candidates
            min_deg = n + 1
            min_v = -1
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                c ^= lsb
                # Degree of v in candidates
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            
            if min_v == -1:
                return False
            
            # Option 1: include min_v
            new_candidates = candidates & ~adj[min_v] & ~(1 << min_v)
            if search(current_size + 1, new_candidates):
                return True
            
            # Option 2: exclude min_v
            new_candidates = candidates & ~(1 << min_v)
            if search(current_size, new_candidates):
                return True
            
            return False
        
        full = (1 << n) - 1
        if search(0, full):
            return limit
        return best[0]

    def build_turan_graph(n, k):
        """Build Turan graph T(n,k): complete k-partite graph with parts as equal as possible."""
        adj = [0] * n
        # Assign vertices to parts
        part_size = n // k
        remainder = n % k
        # First 'remainder' parts have part_size+1 vertices, rest have part_size
        part_of_vertex = []
        idx = 0
        for p in range(k):
            size = part_size + (1 if p < remainder else 0)
            for _ in range(size):
                part_of_vertex.append(p)
                idx += 1
        
        # Connect vertices in different parts
        for i in range(n):
            for j in range(i+1, n):
                if part_of_vertex[i] != part_of_vertex[j]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Try deterministic Turan constructions for n from 40 down to 26
    for n_try in range(40, 25, -1):
        if time.time() > deadline:
            break
        
        # We need alpha(G) < 10, so we need enough parts
        # For Turan graph T(n,k), alpha = ceil(n/k)
        # Need ceil(n/k) <= 9, so k >= ceil(n/9)
        min_k = (n_try + 8) // 9  # ceil(n/9)
        
        for k in range(min_k, n_try):
            if time.time() > deadline:
                break
            
            adj = build_turan_graph(n_try, k)
            
            # Verify triangle-free
            if not is_triangle_free(adj, n_try):
                continue
            
            # Verify no independent set of size 10
            alpha = max_independent_set_size(adj, n_try, 10)
            if alpha < 10:
                edges = [[i, j] for i in range(n_try) for j in range(i+1, n_try) if (adj[i] >> j) & 1]
                return {'n': n_try, 'edges': edges}
    
    # Fallback: seed graph
    n_seed = 26
    adj_seed = [0] * n_seed
    for i in range(n_seed):
        for j in range(i+1, n_seed):
            if (j - i) % n_seed % 3 == 1:
                adj_seed[i] |= (1 << j)
                adj_seed[j] |= (1 << i)
    edges = [[i, j] for i in range(n_seed) for j in range(i+1, n_seed) if (adj_seed[i] >> j) & 1]
    return {'n': n_seed, 'edges': edges}
