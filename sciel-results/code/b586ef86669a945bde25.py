import time

def construct():
    start_time = time.time()
    deadline = 90.0  # seconds
    
    # 1. Setup Graph
    n = 14
    edges = P['edges']
    # Adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Complement adjacency
    comp_adj = [0] * n
    for i in range(n):
        # All bits set except self and neighbors
        # (1 << n) - 1 is all 1s
        # ~adj[i] & ((1 << n) - 1) & ~(1 << i)
        comp_adj[i] = (~adj[i]) & ((1 << n) - 1) & ~(1 << i)

    # 2. Find Maximal Cliques and Maximal Independent Sets
    # Iterate through all subsets 1 to (1<<n)-1
    # A subset S is a clique if for all v in S, (S without v) is subset of adj[v]
    # Or simpler: for all u,v in S, u in adj[v]
    
    valid_sets = []
    
    # Helper to check if mask is clique
    def is_clique(mask):
        # If mask has 0 or 1 bits, it's a clique
        if mask & (mask - 1) == 0:
            return True
        v_bit = mask & -mask
        v = v_bit.bit_length() - 1
        rest = mask ^ v_bit
        # All vertices in rest must be adjacent to v
        # adj[v] contains neighbors of v
        # We need rest to be a subset of adj[v]
        return (rest & ~adj[v]) == 0
        
    def is_maximal_clique(mask, adj_matrix):
        if not is_clique(mask):
            return False
        # Check if any vertex outside can be added
        # For each v not in mask, if (mask & ~adj[v]) == 0, then v can be added?
        # No, v can be added if v is adjacent to ALL vertices in mask.
        # i.e., mask is subset of adj[v].
        # i.e., (mask & ~adj[v]) == 0
        for v in range(n):
            if not (mask & (1 << v)):
                if (mask & ~adj_matrix[v]) == 0:
                    return False
        return True

    # This is O(2^n * n^2) roughly. 16384 * 196 ~ 3M ops. Very fast.
    
    max_cliques = []
    max_indep_sets = []
    
    for mask in range(1, 1 << n):
        # Check maximal clique
        if is_maximal_clique(mask, adj):
            max_cliques.append(mask)
        # Check maximal independent set (maximal clique in complement)
        if is_maximal_clique(mask, comp_adj):
            max_indep_sets.append(mask)
            
    # Combine and deduplicate
    candidates = list(set(max_cliques + max_indep_sets))
    
    # Sort by size descending to find good solutions early
    candidates.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    # 3. Branch and Bound for Minimum Set Cover
    target_mask = (1 << n) - 1
    best_count = n + 1
    best_cover = []
    
    # Precompute: for each vertex, list of candidate indices that cover it
    # This helps in branching on the "most constrained" vertex
    vertex_to_sets = [[] for _ in range(n)]
    for i, s in enumerate(candidates):
        for v in range(n):
            if s & (1 << v):
                vertex_to_sets[v].append(i)
                
    # Order vertices by number of covering sets (ascending) to prune earlier
    # But we need to pick the first uncovered one dynamically.
    
    def backtrack(covered_mask, current_sets, depth):
        nonlocal best_count, best_cover
        
        if time.time() > deadline - 1.0:
            return
            
        if covered_mask == target_mask:
            if depth < best_count:
                best_count = depth
                best_cover = current_sets[:]
            return
            
        if depth >= best_count:
            return
            
        # Find first uncovered vertex
        rem = target_mask ^ covered_mask
        # Find the vertex with fewest covering sets among uncovered ones?
        # Or just the first one?
        # Let's pick the uncovered vertex with the fewest candidate sets covering it.
        # This is a heuristic for faster pruning.
        
        min_options = float('inf')
        chosen_v = -1
        chosen_set_indices = []
        
        # Iterate through uncovered bits
        r = rem
        while r:
            v_bit = r & -r
            v = v_bit.bit_length() - 1
            # Count how many candidates cover v AND are not already fully covered?
            # Actually, we just need candidates that cover v.
            # We can filter candidates that add at least one new vertex?
            # For now, just count total candidates covering v.
            # We can refine: only consider candidates that intersect with rem.
            count = 0
            for idx in vertex_to_sets[v]:
                # Check if this set adds any new vertex
                if candidates[idx] & rem:
                    count += 1
            if count < min_options:
                min_options = count
                chosen_v = v
                chosen_set_indices = [idx for idx in vertex_to_sets[v] if candidates[idx] & rem]
            if count == 0:
                # No set covers this vertex? Impossible if we have all maximal sets.
                return
            r ^= v_bit
            
        if chosen_v == -1:
            return
            
        # Sort these set indices by size of intersection with rem (larger first)
        # This helps cover more vertices per step
        chosen_set_indices.sort(key=lambda idx: bin(candidates[idx] & rem).count('1'), reverse=True)
        
        for idx in chosen_set_indices:
            s = candidates[idx]
            new_covered = covered_mask | s
            
            current_sets.append(idx)
            backtrack(new_covered, current_sets, depth + 1)
            current_sets.pop()
            
            # Pruning: if we found a solution, and the lower bound is >= best, stop
            # Lower bound: depth + 1
            # If depth + 1 >= best_count, we can't improve.
            # But we already check depth >= best_count at start.
            # A tighter bound: depth + ceil(remaining_vertices / max_set_size)
            # But simple depth check is usually sufficient for n=14.

    backtrack(0, [], 0)
    
    if best_count == n + 1:
        # Fallback to greedy if B&B failed or timed out (unlikely)
        # Greedy: pick largest set, remove covered, repeat
        covered = 0
        parts = []
        while covered != target_mask:
            best_idx = -1
            best_gain = -1
            for i, s in enumerate(candidates):
                gain = bin(s & ~covered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            parts.append(best_idx)
            covered |= candidates[best_idx]
        best_cover = parts
        best_count = len(parts)
        
    # 4. Construct Partition
    partition = []
    assigned = 0
    for idx in best_cover:
        s = candidates[idx]
        new_part_mask = s & ~assigned
        if new_part_mask:
            # Convert mask to list of vertices
            part_vertices = []
            m = new_part_mask
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                part_vertices.append(v)
                m ^= v_bit
            partition.append(part_vertices)
            assigned |= new_part_mask
            
    # Ensure all vertices are covered
    if assigned != target_mask:
        # Should not happen, but add remaining as singletons
        for v in range(n):
            if not (assigned & (1 << v)):
                partition.append([v])
                assigned |= (1 << v)
                
    return partition

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
