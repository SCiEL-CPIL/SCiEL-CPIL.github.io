"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    base_edges_set = set()
    for u, v in P['edges']:
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # Base adjacency as bitmasks for speed
    # n=24 fits in an integer
    base_adj = [0] * n
    for u, v in P['edges']:
        base_adj[u] |= (1 << v)
        base_adj[v] |= (1 << u)
        
    # Generate candidate edges
    # We want to minimize edges. 
    # Short edges (span 3) are likely inefficient for global diameter.
    # Long edges (span >= 5 or 6) are better.
    # Let's generate all non-base, non-triangle-forming edges.
    # Triangle check: (u,v) forms triangle if common neighbor exists in base.
    # In a path, common neighbor exists iff dist(u,v) == 2.
    # So we exclude span 2.
    # Span 1 is base.
    # So candidates are span >= 3.
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges_set:
                continue
            span = v - u
            if span < 3:
                continue
            # Check triangle with base graph
            # Common neighbors of u and v in base graph
            # In a path, neighbors are u-1, u+1, v-1, v+1.
            # Common neighbor exists if {u-1, u+1} intersects {v-1, v+1}
            # This happens if v-1 == u+1 => v-u=2 (excluded)
            # or v+1 == u-1 (impossible for u<v)
            # etc.
            # So for span >= 3, no triangle with base.
            candidates.append((u, v))
            
    # Heuristic: Prioritize longer edges.
    # Sort by span descending.
    candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    # Prune candidates to a manageable size for search.
    # If we try to find k=3, we need C(M, 3) to be small.
    # If M=30, C(30,3) = 4050. Very fast.
    # If M=40, C(40,3) = 9880. Fast.
    # If M=50, C(50,3) = 19600. Fast.
    # Let's take top 40 longest edges.
    # Also, we might need to ensure coverage of both ends.
    # The longest edges are (0,23), (0,22), (1,23), etc.
    
    pool_size = 40
    pool = candidates[:pool_size]
    
    def get_diameter(adj_masks):
        """
        adj_masks: list of int bitmasks representing adjacency.
        Returns diameter.
        """
        max_d = 0
        for i in range(n):
            # BFS using bitmasks
            visited = 1 << i
            dist = 0
            frontier = visited
            
            while frontier:
                # Expand frontier
                # neighbors = union of adj[x] for x in frontier
                # We can iterate through set bits in frontier
                next_frontier = 0
                f = frontier
                while f:
                    # Get lowest set bit
                    lsb = f & -f
                    u = lsb.bit_length() - 1
                    f -= lsb
                    
                    # Add neighbors of u to next_frontier
                    # Mask out already visited
                    next_frontier |= adj_masks[u]
                
                # Remove visited from next_frontier
                next_frontier &= ~visited
                
                if not next_frontier:
                    break
                    
                visited |= next_frontier
                frontier = next_frontier
                dist += 1
                
                if dist > max_d:
                    max_d = dist
                    # Early exit if we already know it's > 4?
                    # We want to check if diameter <= 4.
                    # If dist > 4, we can return early to save time.
                    if dist > 4:
                        return dist
                        
        return max_d

    # Search for minimum k
    # Start with k=3. If not found, k=4, etc.
    
    # To speed up k=3, we can just iterate combinations.
    # To speed up k=4, we might need pruning, but C(40,4) ~ 100k is fine.
    
    best_solution = None
    
    # Try k=3
    for combo in combinations(pool, 3):
        if time.time() > deadline:
            break
            
        # Construct adj masks
        adj_masks = base_adj[:]
        for u, v in combo:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        if get_diameter(adj_masks) <= 4:
            best_solution = list(combo)
            break
            
    # If k=3 failed, try k=4
    if best_solution is None and time.time() < deadline:
        for combo in combinations(pool, 4):
            if time.time() > deadline:
                break
                
            adj_masks = base_adj[:]
            for u, v in combo:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
            if get_diameter(adj_masks) <= 4:
                best_solution = list(combo)
                break
                
    # If k=4 failed, try k=5
    if best_solution is None and time.time() < deadline:
        # C(40,5) is ~ 650k. Might be slow with Python loops.
        # Let's optimize the check or reduce pool.
        # If we didn't find it in top 40, maybe we need more candidates?
        # Or maybe k=5 is not the answer?
        # Let's try a greedy fallback if combinatorial search is too slow or fails.
        
        # Actually, let's just try k=5 on a slightly smaller pool to be safe, 
        # or just break and use greedy.
        # Given the path structure, k=3 or 4 is highly likely.
        # Let's do a quick greedy to ensure we have a valid answer if search fails.
        
        current_adj = base_adj[:]
        current_edges = []
        
        # Greedy: pick edge that reduces diameter most
        # Re-evaluate candidates dynamically? 
        # For simplicity and speed, just pick from pool sequentially?
        # No, greedy should pick best.
        
        # Let's just return empty if we fail, but we should not fail.
        # To be safe, let's run a simple greedy on the full candidate list.
        
        # Re-generate full candidate list for greedy
        all_candidates = candidates[:]
        
        while True:
            if time.time() > deadline:
                break
                
            best_edge = None
            best_diam = 24 # Max possible
            
            # Sort candidates by span desc for tie-breaking in greedy
            # We already have them sorted.
            
            for u, v in all_candidates:
                # Check if edge already added
                if (current_adj[u] >> v) & 1:
                    continue
                    
                # Check triangle
                # Common neighbors in current graph
                common = current_adj[u] & current_adj[v]
                if common:
                    continue
                    
                # Tentatively add
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
                d = get_diameter(current_adj)
                
                # Undo
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                if d < best_diam:
                    best_diam = d
                    best_edge = (u, v)
                    
            if best_edge is None:
                break
                
            u, v = best_edge
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            current_edges.append((u, v))
            
            if best_diam <= 4:
                best_solution = current_edges
                break
                
    if best_solution:
        # Convert to list of lists of ints
        final_added = [[int(u), int(v)] for u, v in best_solution]
        return final_added
    else:
        return []

# EVOLVE-BLOCK-END
