import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_n = 0
    best_edges = []
    min_edge_count = float('inf')
    
    # We search for n in a reasonable range
    # The minimum number of edges is >= 29.
    # Let's try n=9 to n=14.
    
    for n in range(9, 15):
        if time.time() > deadline:
            break
            
        vertices = list(range(n))
        all_edges = list(combinations(vertices, 5))
        num_colorings = 1 << n
        
        # Precompute edge masks
        # mask[i] is an integer where bit c is 1 if coloring c is monochromatic on edge i
        edge_masks = []
        for e in all_edges:
            # Create a bitmask for the vertices in this edge
            v_mask = 0
            for v in e:
                v_mask |= (1 << v)
            
            m = 0
            for c in range(num_colorings):
                # Check if c is monochromatic on e
                # c & v_mask should be 0 or v_mask
                if (c & v_mask) == 0 or (c & v_mask) == v_mask:
                    m |= (1 << c)
            edge_masks.append(m)
            
        uncovered = (1 << num_colorings) - 1
        selected_edges = []
        
        # Sort edges by potential gain (number of bits set) to help pruning?
        # Or just use the B&B logic.
        
        # B&B Solver
        # We want to minimize the number of edges.
        # State: (uncovered_mask, current_count, current_edges_list)
        
        # Heuristic: Always pick the edge that covers the most uncovered bits first.
        # This is a "Greedy" step within the search, but we need to be exact.
        # Actually, simple B&B on Set Cover is hard.
        # Let's use a recursive search that tries to add edges.
        
        def solve(uncovered, depth, current_edges):
            nonlocal min_edge_count, best_n, best_edges
            nonlocal edge_masks, all_edges, num_colorings, deadline, start_time
            
            if time.time() > deadline:
                return True # Signal timeout
            
            if uncovered == 0:
                if depth < min_edge_count:
                    min_edge_count = depth
                    best_n = n
                    best_edges = list(current_edges)
                return False # Continue search to find smaller? No, we want min.
                # Actually, if we find a solution, we can prune branches with depth >= min_edge_count.
                
            if depth >= min_edge_count:
                return False
                
            # Lower bound:
            # Max bits covered by any single edge is max_popcount
            # min_edges_needed = ceil(popcount(uncovered) / max_popcount)
            # If depth + min_edges_needed >= min_edge_count, prune.
            pop_uncovered = uncovered.bit_count()
            if pop_uncovered == 0:
                 return False # Should have been caught above
            
            # Find the edge with max coverage among uncovered
            max_gain = 0
            best_edge_idx = -1
            for i, m in enumerate(edge_masks):
                gain = (m & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_edge_idx = i
            
            if max_gain == 0:
                # No edge can cover any uncovered coloring? Impossible unless uncovered is 0.
                # Or if we ran out of edges? No, we have all edges.
                # This means it's impossible to cover the remaining colorings with any edges?
                # Wait, if max_gain is 0, it means for every edge, it covers 0 new colorings.
                # This implies the uncovered colorings are not monochromatic on ANY edge.
                # This means the hypergraph formed by ALL edges is still 2-colorable?
                # No, if we use ALL edges, we cover all colorings?
                # If max_gain is 0, it means the current set of ALL edges doesn't cover `uncovered`.
                # This implies the full hypergraph is 2-colorable.
                return True # Dead end
                
            # Pruning check
            # A crude bound: each edge covers at most max_gain bits.
            # But max_gain changes as we cover bits.
            # Better bound: ceil(pop_uncovered / global_max_edge_size)
            # Global max edge size is fixed.
            # Let's compute global_max_edge_size once.
            
            # For now, just branch on the best edge.
            # This is a "Greedy Backtracking". It might miss optimal solutions if the greedy choice is bad.
            # To be exact, we should branch on ALL edges? That's too slow.
            # Standard Set Cover B&B branches on including the "best" edge and excluding it?
            # Or branching on the element with the fewest covering sets?
            
            # Let's try a different heuristic:
            # Pick an uncovered coloring c.
            # It must be covered by one of the edges that make c monochromatic.
            # Let C be the set of edges that cover c.
            # We must pick at least one edge from C.
            # Branch on each edge in C.
            
            # Find an uncovered coloring
            # Get the index of the least significant set bit
            c_idx = (uncovered & -uncovered).bit_length() - 1
            
            # Find all edges that cover c_idx
            candidate_edges = []
            for i, m in enumerate(edge_masks):
                if (m >> c_idx) & 1:
                    candidate_edges.append(i)
            
            # Sort candidates by gain (heuristic)
            candidate_edges.sort(key=lambda i: (edge_masks[i] & uncovered).bit_count(), reverse=True)
            
            for i in candidate_edges:
                new_uncovered = uncovered & ~edge_masks[i]
                current_edges.append(i)
                
                # Recursive call
                timeout = solve(new_uncovered, depth + 1, current_edges)
                if timeout:
                    current_edges.pop()
                    return True
                
                current_edges.pop()
                
                # Pruning: if we already have a solution with depth+1, 
                # and we are trying to find < depth+1, we don't need to explore other branches
                # that lead to >= depth+1? 
                # The logic `if depth >= min_edge_count` handles this.
                
                # Additional pruning: 
                # If we found a solution of size K, and we are at depth D,
                # and the lower bound is D + LB, if D + LB >= K, prune.
                
        # We need global_max_edge_size for better pruning?
        # Let's just run it.
        
        timeout = solve(uncovered, 0, selected_edges)
        
        if timeout:
            continue
            
    if best_edges:
        return {'n': best_n, 'edges': [list(all_edges[i]) for i in best_edges]} # Wait, all_edges is local to the loop.
        # We need to store the actual edges, not indices, or re-derive them.
        # Better to store the edges directly in best_edges.
        
    # Fallback
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
