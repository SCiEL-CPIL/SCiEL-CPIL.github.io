import math

def construct():
    N = 256
    weights = [i + 1 for i in range(N)]
    
    # Precompute conflict neighbors (gcd > 1)
    conflict_neighbors = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if i != j:
                vj = j + 1
                if math.gcd(vi, vj) > 1:
                    mask |= (1 << j)
        conflict_neighbors[i] = mask

    # Precompute list of indices sorted by weight descending
    # indices[i] is the vertex index with the (i+1)-th largest weight
    # Since weights are 1..256, index i has weight i+1.
    # So sorted by weight descending is just N-1 down to 0.
    sorted_indices = list(range(N - 1, -1, -1))

    best_sum = 0
    best_indices = []

    def calc_bound(candidates):
        """
        Greedy Clique Cover Bound.
        Partitions 'candidates' into cliques of the conflict graph.
        Returns sum of max weights of each clique.
        """
        total_bound = 0
        U = candidates
        
        while U:
            # Pick vertex v with max weight in U
            v_idx = -1
            for idx in sorted_indices:
                if (U >> idx) & 1:
                    v_idx = idx
                    break
            
            if v_idx == -1:
                break
            
            # Remove v from U
            U &= ~(1 << v_idx)
            
            # Current clique contains v
            # Max weight in this clique so far
            clique_max = weights[v_idx]
            
            # Vertices that can be added to this clique:
            # Must be in U AND non-coprime to all vertices currently in the clique.
            # Start with neighbors of v.
            valid = U & conflict_neighbors[v_idx]
            
            while valid:
                # Pick vertex u with max weight in valid
                u_idx = -1
                for idx in sorted_indices:
                    if (valid >> idx) & 1:
                        u_idx = idx
                        break
                
                if u_idx == -1:
                    break
                
                # Add u to clique
                # Update clique max
                if weights[u_idx] > clique_max:
                    clique_max = weights[u_idx]
                
                # Remove u from U and valid
                U &= ~(1 << u_idx)
                valid &= ~(1 << u_idx)
                
                # Update valid: u must be non-coprime to all in clique.
                # Since we built the clique greedily, u is non-coprime to v.
                # We need u to be non-coprime to the *next* candidate?
                # Actually, `valid` currently contains vertices non-coprime to v.
                # When we pick u, the *next* vertex w must be non-coprime to v AND u.
                # So we intersect `valid` with `conflict_neighbors[u_idx]`.
                valid &= conflict_neighbors[u_idx]
            
            total_bound += clique_max
            
        return total_bound

    def solve(candidates, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if candidates == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices[:]
            return

        # Calculate Upper Bound
        # Optimization: If current_sum + sum of all weights in candidates <= best_sum, prune.
        # calc_bound is tighter but slower.
        # Let's use calc_bound directly.
        ub = current_sum + calc_bound(candidates)
        if ub <= best_sum:
            return

        # Select vertex v to branch on
        # Heuristic: Vertex with highest degree in the induced conflict subgraph
        # Tie-breaker: Highest weight
        best_v = -1
        best_deg = -1
        best_w = -1
        
        # Iterate through candidates
        # To be efficient, we can iterate through sorted_indices and check if in candidates
        for idx in sorted_indices:
            if (candidates >> idx) & 1:
                deg = (conflict_neighbors[idx] & candidates).bit_count()
                w = weights[idx]
                # Prefer higher degree, then higher weight
                if deg > best_deg or (deg == best_deg and w > best_w):
                    best_deg = deg
                    best_w = w
                    best_v = idx
        
        v = best_v
        
        # Branch 1: Include v
        # Remove v and its conflicts from candidates
        mask_remove = conflict_neighbors[v] | (1 << v)
        new_cands = candidates & ~mask_remove
        solve(new_cands, current_sum + weights[v], current_indices + [v])
        
        # Branch 2: Exclude v
        new_cands_ex = candidates & ~(1 << v)
        solve(new_cands_ex, current_sum, current_indices)

    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    result = [i + 1 for i in best_indices]
    return [int(x) for x in result]
