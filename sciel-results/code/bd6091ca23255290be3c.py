"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = set()
    for e in P['edges']:
        edges.add((e[0], e[1]))
        edges.add((e[1], e[0]))
    
    # Precompute adjacency as bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute: for each subset (bitmask), is it a clique? is it an independent set?
    # A subset is a clique if for every pair, there's an edge
    # A subset is an independent set if for every pair, there's no edge
    
    all_cliques = [False] * (1 << n)
    all_indep = [False] * (1 << n)
    
    # Empty set is both
    all_cliques[0] = True
    all_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        # Check clique: v must be adjacent to all in rest, and rest must be clique
        if all_cliques[rest]:
            # v must connect to all vertices in rest
            if (adj[v] & rest) == rest:
                all_cliques[mask] = True
        
        # Check independent set: v must NOT be adjacent to any in rest, and rest must be independent
        if all_indep[rest]:
            # v must have no edges to rest
            if (adj[v] & rest) == 0:
                all_indep[mask] = True
    
    # List of all valid subsets (clique or independent), excluding empty
    valid_subsets = []
    for mask in range(1, 1 << n):
        if all_cliques[mask] or all_indep[mask]:
            valid_subsets.append(mask)
    
    # Branch and bound with memoization
    from functools import lru_cache
    
    full_mask = (1 << n) - 1
    
    @lru_cache(maxsize=None)
    def solve(remaining):
        if remaining == 0:
            return 0
        
        # Lower bound: at least 1
        best = n  # worst case: n parts of size 1
        
        # Try each valid subset that is a subset of remaining
        # To avoid duplicates, we can fix that the smallest vertex in remaining must be in the first part
        # Find the lowest set bit in remaining
        lb = remaining & (-remaining)
        v = lb.bit_length() - 1
        
        # Only consider valid subsets that contain v and are subset of remaining
        for mask in valid_subsets:
            if not (mask & remaining) == mask:
                continue
            if not (mask & lb):
                continue
            # This subset is valid and subset of remaining and contains v
            new_remaining = remaining ^ mask
            count = 1 + solve(new_remaining)
            if count < best:
                best = count
            if best == 1:
                break
        
        return best
    
    min_parts = solve(full_mask)
    
    # Now reconstruct the actual partition
    def reconstruct(remaining, current_parts):
        if remaining == 0:
            return current_parts
        
        lb = remaining & (-remaining)
        v = lb.bit_length() - 1
        
        for mask in valid_subsets:
            if not (mask & remaining) == mask:
                continue
            if not (mask & lb):
                continue
            new_remaining = remaining ^ mask
            if 1 + solve(new_remaining) == solve(remaining):
                # Extract vertices from mask
                vertices = []
                for i in range(n):
                    if mask & (1 << i):
                        vertices.append(i)
                return reconstruct(new_remaining, current_parts + [vertices])
        
        return current_parts
    
    partition = reconstruct(full_mask, [])
    
    # Convert to list of lists of ints (ensure Python int, not numpy)
    result = []
    for part in partition:
        result.append([int(v) for v in part])
    
    return result

# EVOLVE-BLOCK-END
