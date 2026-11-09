"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 70.0  # safety margin before 90s kill
    N = 256
    
    # 1. Build conflict graph using bitmasks
    # conflict_mask[i] is a bitmask where bit j is 1 if i and j conflict
    # Indices 1..N map to bits 0..N-1
    conflict_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                # Set bit (j-1) in mask[i] and bit (i-1) in mask[j]
                conflict_mask[i] |= (1 << (j - 1))
                conflict_mask[j] |= (1 << (i - 1))
                
    # Initial candidates: all nodes 1..N
    # Bit i-1 represents node i
    all_nodes_mask = (1 << N) - 1
    
    best_size = 0
    best_set_mask = 0
    
    # Tomita's Branch and Bound
    # We use a recursive function. To avoid deep recursion limits, we rely on the fact that N=256 is small enough for Python's recursion limit (default 1000) if we prune well, or we can increase it.
    import sys
    sys.setrecursionlimit(3000)
    
    def greedy_coloring(candidates):
        """
        Assigns colors to vertices in 'candidates' such that adjacent vertices have different colors.
        Returns a list of colors for each vertex in candidates (ordered by bit position).
        Actually, we just need the max color count to prune.
        Tomita's algorithm typically sorts vertices by degree or uses a specific ordering.
        Here we use a simple greedy coloring on the induced subgraph.
        """
        colors = {}
        max_color = 0
        # Iterate through bits in candidates
        c = candidates
        while c:
            # Get lowest set bit
            lsb = c & -c
            idx = lsb.bit_length() - 1
            node = idx + 1
            
            # Find available color
            used_colors = set()
            # Check neighbors in candidates
            neighbors_in_c = conflict_mask[node] & candidates
            n = neighbors_in_c
            while n:
                n_lsb = n & -n
                n_idx = n_lsb.bit_length() - 1
                n_node = n_idx + 1
                if n_node in colors:
                    used_colors.add(colors[n_node])
                n ^= n_lsb
                
            # Assign smallest available color
            color = 0
            while color in used_colors:
                color += 1
            colors[node] = color
            if color > max_color:
                max_color = color
            
            c ^= lsb
        return max_color

    def branch_and_bound(candidates, current_size, current_set):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return
            
        # Pruning
        if current_size + candidates.bit_count() <= best_size:
            return
            
        # If no candidates, update best
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_set
            return
        
        # Coloring bound
        max_c = greedy_coloring(candidates)
        if current_size + max_c <= best_size:
            return
            
        # Pick a vertex to branch on.
        # Heuristic: Pick vertex with highest degree in the induced subgraph?
        # Or just pick the first one. Tomita's original uses a specific ordering.
        # Let's pick the vertex with the most neighbors in the candidate set to reduce the 'include' branch size.
        
        c = candidates
        best_node = -1
        max_deg = -1
        
        while c:
            lsb = c & -c
            idx = lsb.bit_length() - 1
            node = idx + 1
            deg = (conflict_mask[node] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_node = node
            c ^= lsb
            
        if best_node == -1:
            return
            
        # Branch 1: Include best_node
        # New candidates = candidates \ {best_node} \ neighbors(best_node)
        # Note: best_node is in candidates.
        # neighbors in candidates = conflict_mask[best_node] & candidates
        # We remove best_node and its neighbors.
        # Since best_node is a neighbor of itself? No, graph has no self-loops.
        # So new_c = candidates & ~(conflict_mask[best_node] | (1 << (best_node - 1)))
        
        new_candidates = candidates & ~(conflict_mask[best_node] | (1 << (best_node - 1)))
        branch_and_bound(new_candidates, current_size + 1, current_set | (1 << (best_node - 1)))
        
        # Branch 2: Exclude best_node
        # New candidates = candidates \ {best_node}
        new_candidates_excl = candidates & ~(1 << (best_node - 1))
        branch_and_bound(new_candidates_excl, current_size, current_set)

    # Start search
    branch_and_bound(all_nodes_mask, 0, 0)
    
    # Convert best_set_mask to list
    result = []
    m = best_set_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    if not result:
        # Fallback: just return [1] which is always valid
        return [1]
        
    return result
# EVOLVE-BLOCK-END
