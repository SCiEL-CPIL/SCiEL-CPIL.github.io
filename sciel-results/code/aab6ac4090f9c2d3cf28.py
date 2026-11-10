"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring (no Property B).
    Strategy:
    1. Use a fast 2-colorability checker based on iterating all 2^n colorings with bitmask checks.
    2. Start with the complete 5-uniform hypergraph K^5_n for small n (9, 10, 11).
    3. Minimize the edge set by greedily removing edges that do not restore 2-colorability.
    4. Return the hypergraph with the fewest edges found.
    """
    import random
    import time
    from itertools import combinations

    def is_2_colorable(n, edges):
        """
        Check if a 5-uniform hypergraph on n vertices is 2-colorable.
        Uses bitmask iteration over all 2^n possible colorings.
        Returns True if a valid 2-coloring exists, False otherwise.
        """
        if not edges:
            return True
        
        # Precompute bitmask for each edge
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Iterate through all 2^n colorings
        # mask represents the coloring: bit v is 1 if vertex v is color 1, 0 if color 0
        for coloring in range(1 << n):
            is_valid = True
            for emask in edge_masks:
                # Check if the edge is monochromatic
                # bits_in_edge = coloring & emask
                # If bits_in_edge == 0, all vertices in edge are 0 (monochromatic)
                # If bits_in_edge == emask, all vertices in edge are 1 (monochromatic)
                bits_in_edge = coloring & emask
                if bits_in_edge == 0 or bits_in_edge == emask:
                    is_valid = False
                    break
            if is_valid:
                return True
        return False

    def minimize_hypergraph(n):
        """
        Start with K^5_n and remove edges while maintaining non-2-colorability.
        """
        all_edges = [list(e) for e in combinations(range(n), 5)]
        # Start with all edges
        current_edges = all_edges[:]
        
        # We will try to remove edges one by one.
        # To be efficient, we shuffle the edge list and try to remove each one.
        # If removal keeps the hypergraph non-2-colorable, we keep it removed.
        # We repeat this process until no more edges can be removed.
        
        # Optimization: Limit the number of iterations to avoid infinite loops or excessive time
        # Although theoretically it terminates, we add a time check.
        start_time = time.time()
        time_limit = 50  # seconds per n
        
        while time.time() - start_time < time_limit:
            # Shuffle current edges to try different removal orders
            random.shuffle(current_edges)
            found_removal = False
            i = 0
            while i < len(current_edges):
                if time.time() - start_time > time_limit:
                    break
                
                edge_to_remove = current_edges[i]
                # Try removing edge_to_remove
                # Create a new list without this edge
                # Instead of copying the whole list, we can check with a flag
                # But is_2_colorable takes a list. 
                # For n=9, list is ~126 items. Copying is fast.
                
                # Optimization: We can just pass a filtered view? No, is_2_colorable iterates.
                # Let's just create a new list.
                temp_edges = []
                for e in current_edges:
                    if e != edge_to_remove:
                        temp_edges.append(e)
                
                # Check if temp_edges is 2-colorable
                if not is_2_colorable(n, temp_edges):
                    # It is still non-2-colorable, so we can remove the edge
                    current_edges = temp_edges
                    found_removal = True
                    # Break inner loop to restart with new list
                    break
                else:
                    # It became 2-colorable, so we must keep the edge
                    i += 1
            
            if not found_removal:
                # No more edges can be removed
                break
        
        return current_edges

    best_edges = None
    best_n = 0
    
    # Try different n values
    for n in [9, 10, 11]:
        edges = minimize_hypergraph(n)
        if best_edges is None or len(edges) < len(best_edges):
            best_edges = edges
            best_n = n
    
    # If we didn't find anything (shouldn't happen), fallback to K^5_9
    if best_edges is None:
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
