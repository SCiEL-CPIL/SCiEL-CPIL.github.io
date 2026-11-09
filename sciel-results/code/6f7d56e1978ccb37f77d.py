"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute: for each subset (as bitmask), whether it's a clique or independent set
    # n=14, so 2^14 = 16384 subsets, manageable
    
    # Efficient clique check: a set is a clique iff for every vertex v in the set,
    # all other vertices in the set are neighbors of v.
    # Equivalently: for all v in mask, (mask & ~(1<<v)) is a subset of adj_mask[v]
    
    # Efficient independent set check: a set is independent iff for every vertex v in the set,
    # no other vertex in the set is a neighbor of v.
    # Equivalently: for all v in mask, (mask & adj_mask[v]) == 0 (only self if included)
    
    is_clique_arr = [False] * (1 << n)
    is_indep_arr = [False] * (1 << n)
    
    # Single vertices are both cliques and independent sets
    for v in range(n):
        is_clique_arr[1 << v] = True
        is_indep_arr[1 << v] = True
    
    # Build up: a set S is a clique iff for some v in S, S\{v} is a clique and 
    # v is adjacent to all vertices in S\{v}
    # A set S is independent iff for some v in S, S\{v} is independent and 
    # v is not adjacent to any vertex in S\{v}
    
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue  # single vertex, already handled
        
        # Pick a vertex from mask
        v = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << v)
        
        if is_clique_arr[rest] and (rest & ~adj_mask[v]) == 0:
            is_clique_arr[mask] = True
        
        if is_indep_arr[rest] and (rest & adj_mask[v]) == 0:
            is_indep_arr[mask] = True
    
    # Find all valid subsets (cliques or independent sets)
    valid_masks = []
    for mask in range(1, 1 << n):
        if is_clique_arr[mask] or is_indep_arr[mask]:
            valid_masks.append(mask)
    
    # Sort by size descending to prefer larger parts
    valid_masks.sort(key=lambda m: -m.bit_count())
    
    # Precompute which valid masks are subsets of which masks for faster lookup
    # Group valid masks by their "signature" - the set of vertices they contain
    
    full_mask = (1 << n) - 1
    
    # For each vertex, precompute the list of valid masks that contain it
    masks_containing = [[] for _ in range(n)]
    for mask in valid_masks:
        for v in range(n):
            if mask & (1 << v):
                masks_containing[v].append(mask)
    
    # Sort each list by size descending
    for v in range(n):
        masks_containing[v].sort(key=lambda m: -m.bit_count())
    
    # Lower bound: for remaining vertices, what's the minimum number of parts needed?
    # Simple bound: at least ceil(popcount(remaining) / max_part_size)
    # Better: use a greedy coloring-like bound
    
    max_part_size = max(m.bit_count() for m in valid_masks) if valid_masks else 1
    
    def lower_bound(remaining_mask):
        """Lower bound on number of parts needed to cover remaining_mask."""
        count = remaining_mask.bit_count()
        if count == 0:
            return 0
        # Simple bound
        return (count + max_part_size - 1) // max_part_size
    
    best_parts = None
    best_count = n + 1
    
    def solve(remaining_mask, current_parts):
        nonlocal best_parts, best_count
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if len(current_parts) < best_count:
                best_count = len(current_parts)
                best_parts = list(current_parts)
            return
        
        # Pruning: if current_parts already >= best_count, no point continuing
        if len(current_parts) >= best_count:
            return
        
        # Pruning: lower bound
        if len(current_parts) + lower_bound(remaining_mask) >= best_count:
            return
        
        # Pick the vertex with the fewest valid mask options (most constrained)
        # This is a heuristic to reduce branching
        min_options = float('inf')
        chosen_vertex = -1
        for v in range(n):
            if remaining_mask & (1 << v):
                # Count how many valid masks contain v and are subsets of remaining_mask
                count = 0
                for mask in masks_containing[v]:
                    if mask & ~remaining_mask == 0:
                        count += 1
                if count < min_options:
                    min_options = count
                    chosen_vertex = v
                if count == 1:
                    break
        
        if chosen_vertex == -1:
            return
        
        # Try all valid masks that contain chosen_vertex and are subsets of remaining_mask
        for mask in masks_containing[chosen_vertex]:
            if mask & ~remaining_mask:
                continue
            
            current_parts.append(mask)
            solve(remaining_mask & ~mask, current_parts)
            current_parts.pop()
            
            if time.time() > deadline:
                return
    
    # Start with full_mask
    solve(full_mask, [])
    
    if best_parts is not None:
        # Convert bitmask parts to vertex lists
        result = []
        for part_mask in best_parts:
            vertices = [i for i in range(n) if part_mask & (1 << i)]
            result.append(vertices)
        return result
    
    # Fallback: greedy construction if search didn't find anything
    parts = []
    used_mask = 0
    for v in range(n):
        if used_mask & (1 << v):
            continue
        # Find largest valid subset containing v and only unused vertices
        best_mask = 1 << v
        best_size = 1
        for mask in masks_containing[v]:
            if mask & used_mask:
                continue
            size = mask.bit_count()
            if size > best_size:
                best_mask = mask
                best_size = size
        used_mask |= best_mask
        parts.append([i for i in range(n) if best_mask & (1 << i)])
    return parts

# EVOLVE-BLOCK-END
