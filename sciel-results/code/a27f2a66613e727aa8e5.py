import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Initial best solution: each vertex in its own part (n parts)
    # We represent parts as a list of bitmasks
    best_parts = [1 << i for i in range(n)]
    best_count = n
    
    start_time = time.time()
    timeout = 45.0  # Safety margin before 90s limit
    
    # We will use a DFS to find the minimum number of parts.
    # State: unassigned (bitmask), current_parts (list of bitmasks)
    # To improve search, we sort vertices by degree descending?
    # Let's stick to 0..n-1 order for simplicity, but we can pick the 
    # vertex with highest degree in the remaining graph to prune faster.
    
    def get_vertex(unassigned):
        # Pick vertex with highest degree in the induced subgraph of unassigned
        # Or just the first one? 
        # Heuristic: Highest degree in original graph is a good static proxy.
        # Let's precompute degree order.
        pass

    # Static order: 0 to n-1
    # Dynamic order might be better but static is safer to implement correctly.
    
    def dfs(unassigned, current_parts):
        nonlocal best_parts, best_count
        
        if time.time() - start_time > timeout:
            return
            
        if not unassigned:
            if len(current_parts) < best_count:
                best_count = len(current_parts)
                best_parts = current_parts[:]
            return
            
        # Pruning: if current number of parts is already >= best, no need to continue
        if len(current_parts) >= best_count:
            return
            
        # Pick a vertex to assign
        # We pick the first available vertex for determinism
        # v = (unassigned & -unassigned).bit_length() - 1
        # Let's try picking the vertex with the highest degree in the remaining set
        # to fail faster.
        v = -1
        max_deg = -1
        temp_u = unassigned
        while temp_u:
            lsb = temp_u & -temp_u
            idx = lsb.bit_length() - 1
            # Calculate degree of idx in the subgraph induced by unassigned
            deg = bin(adj_mask[idx] & unassigned).count('1')
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp_u ^= lsb
            
        if v == -1:
            return
            
        v_bit = 1 << v
        new_unassigned = unassigned & ~v_bit
        
        # Option 1: Try adding v to existing parts
        for i, p in enumerate(current_parts):
            # Check if adding v to p keeps it a clique
            is_clique = (adj_mask[v] & p) == p
            # Check if adding v to p keeps it an independent set
            is_indep = (adj_mask[v] & p) == 0
            
            if is_clique or is_indep:
                new_parts = current_parts[:]
                new_parts[i] = p | v_bit
                dfs(new_unassigned, new_parts)
                
        # Option 2: Start a new part with v
        new_parts = current_parts + [v_bit]
        dfs(new_unassigned, new_parts)

    # Initial unassigned: all vertices
    initial_unassigned = (1 << n) - 1
    dfs(initial_unassigned, [])
    
    # Convert best_parts (list of bitmasks) to list of lists of ints
    result = []
    for mask in best_parts:
        part = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            part.append(idx)
            m ^= lsb
        result.append(part)
        
    # Ensure types are correct: list of lists of ints
    # The problem requires integers to be Python int, sequences list or tuple.
    # Our construction uses Python ints and lists.
    
    return result
# EVOLVE-BLOCK-END
