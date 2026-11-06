import time
import random
from itertools import combinations

def is_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    i.e., For every 2-coloring, there exists a monochromatic edge.
    """
    if not edges:
        return False # Empty hypergraph is 2-colorable
    
    # Precompute edge bitmasks for faster checking?
    # Or just iterate.
    
    for mask in range(1 << n):
        # Check if this coloring has any monochromatic edge
        has_mono = False
        for e in edges:
            # Check if all vertices in e have same color
            # Get colors
            # If all 0 or all 1, it's mono
            # Optimization: 
            # color of v is (mask >> v) & 1
            # If (mask & e_mask) == 0 or (mask & e_mask) == e_mask, it's mono
            # Precompute e_mask for each edge
            pass
            
    # Let's precompute edge masks
    edge_masks = [0] * len(edges)
    for i, e in enumerate(edges):
        m = 0
        for v in e:
            m |= (1 << v)
        edge_masks[i] = m
        
    for mask in range(1 << n):
        is_mono_found = False
        for i in range(len(edges)):
            em = edge_masks[i]
            # Check if all bits in em are 0 or all are 1 in mask
            if (mask & em) == 0 or (mask & em) == em:
                is_mono_found = True
                break
        if not is_mono_found:
            # Found a coloring with no monochromatic edges
            # So it IS 2-colorable
            return False
            
    # No such coloring found
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    # Start with K_9^(5)
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Verify K_9 is non-2-colorable (sanity check)
    if not is_non_2_colorable(n, all_edges):
        # This should not happen
        return {'n': n, 'edges': all_edges}
        
    best_edges = all_edges[:]
    
    # Greedy removal loop
    # We will try to remove edges to minimize the count
    # We run multiple trials with different random orders
    
    while time.time() < deadline:
        current_edges = best_edges[:]
        random.shuffle(current_edges)
        
        # Try to remove each edge
        # We iterate through the shuffled list
        # If removing an edge keeps the hypergraph non-2-colorable, we remove it
        # We do this in-place on a temporary list to test
        # Actually, it's better to maintain the set of edges and test removal
        
        # To be efficient, we maintain the list of edges currently in the hypergraph
        # and try to remove one by one from the shuffled candidates
        # If removal is successful, we actually remove it from the main list
        
        # Let's use a list of indices or just the edges
        # Since edges are small lists, we can use them as keys? No, just check membership.
        
        # A better way:
        # Start with a copy of best_edges
        # For each edge in the shuffled list:
        #    If edge is in current_set:
        #       Try removing it
        #       If still non-2-colorable, remove it
        
        # This is O(E^2) if we check membership linearly.
        # Use a set of tuples for fast lookup?
        
        current_set = set(tuple(e) for e in best_edges)
        changed = False
        
        for e in current_edges:
            if time.time() > deadline:
                break
                
            te = tuple(e)
            if te not in current_set:
                continue
                
            # Try removing te
            new_set = current_set - {te}
            if not new_set:
                continue
                
            # Check if new_set is non-2-colorable
            # Convert back to list of lists for is_non_2_colorable
            # Or modify is_non_2_colorable to accept a set of tuples
            
            # Let's optimize is_non_2_colorable to accept a list of tuples or masks
            # I'll stick to the list of lists for simplicity in the inner loop if it's fast enough
            # 512 * 125 ops is small.
            
            edges_list = [list(x) for x in new_set]
            if is_non_2_colorable(n, edges_list):
                current_set = new_set
                changed = True
                
        if changed:
            new_best = [list(x) for x in current_set]
            if len(new_best) < len(best_edges):
                best_edges = new_best
                # If we reach a very small number, we can stop early?
                # Theoretical min is 29.
                # If we hit 51, we might still find 50?
                # Keep going until deadline.
        
        # If no change in a trial, we might be stuck in a local minimum for this order.
        # The next trial with different shuffle might escape.
        
    return {'n': n, 'edges': best_edges}
