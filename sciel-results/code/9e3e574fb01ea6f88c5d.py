import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    # Fallback: Seed program (n=30)
    def get_seed_graph():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # Helper: Check if graph is C4-free
    # adj: list of int bitmasks
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common.bit_count() > 1:
                    return False
        return True

    # Helper: Find a C4 and return one edge to remove
    def find_c4_edge(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common.bit_count() > 1:
                    # Find two bits in common
                    bits = []
                    mask = common
                    while mask:
                        lsb = mask & -mask
                        bits.append(lsb.bit_length() - 1)
                        mask ^= lsb
                        if len(bits) == 2:
                            break
                    # u, v, bits[0], bits[1] form a C4
                    # Remove edge (u, bits[0]) or (v, bits[1]) etc.
                    # Let's remove (u, bits[0])
                    return (u, bits[0])
        return None

    # Helper: Check if alpha < 11 (i.e., no independent set of size 11)
    # We check for clique of size 11 in complement
    def has_independent_set_11(adj, n):
        # Use a simple backtracking for clique in complement
        # Complement adjacency: ~adj[i] & ((1<<n)-1) & ~((1<<i)-1) & ~(1<<i) ...
        # Actually, non-neighbors of i: 
        # full_mask = (1<<n) - 1
        # non_neighbors[i] = full_mask ^ adj[i] ^ (1<<i)
        
        full_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            non_neighbors[i] = full_mask ^ adj[i] ^ (1 << i)
            
        # Find clique of size 11 in graph with edges (i,j) if j in non_neighbors[i]
        # This is equivalent to finding independent set in original.
        
        def backtrack(candidates, size):
            if size == 11:
                return True
            if len(candidates) < 11 - size:
                return False
            
            # Pick a vertex
            # Heuristic: pick vertex with max degree in subgraph?
            # For simplicity, just pick first
            v = candidates[0]
            rest = candidates[1:]
            
            # Option 1: Include v
            # New candidates must be neighbors of v in complement (i.e., non-neighbors in original)
            new_candidates = [u for u in rest if (non_neighbors[v] >> u) & 1]
            if backtrack(new_candidates, size + 1):
                return True
            
            # Option 2: Exclude v
            if backtrack(rest, size):
                return True
                
            return False
            
        return backtrack(list(range(n)), 0)

    # Try to construct for n from 39 down to 31
    for n in range(39, 30, -1):
        if time.time() > deadline:
            break
            
        for trial in range(50):
            if time.time() > deadline:
                break
                
            # Generate random graph
            # Start with moderate density
            p = 0.3 + 0.2 * random.random()
            adj = [0] * n
            edges = []
            
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.append([u, v])
                        
            # Remove C4s
            # Limit iterations to avoid infinite loop
            for _ in range(100):
                c4_edge = find_c4_edge(adj, n)
                if c4_edge is None:
                    break
                u, v = c4_edge
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                # Remove from edges list (inefficient, but n is small)
                # We will rebuild edges at the end
                pass
            
            # Rebuild edges list
            edges = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    mask ^= lsb
            
            # Check C4 free again (should be)
            if not is_c4_free(adj, n):
                continue
                
            # Check Independent Set
            if not has_independent_set_11(adj, n):
                return {'n': n, 'edges': edges}
                
            # If alpha >= 11, we could try to add edges, but it's complex.
            # Just try next trial.

    # If no graph found for n > 30, return seed
    return get_seed_graph()
