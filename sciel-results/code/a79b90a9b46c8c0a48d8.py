import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n = 9
    all_edges = list(itertools.combinations(range(n), 5))
    
    # Precompute masks for edges
    edge_masks = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # Map edge tuple to index for fast lookup if needed, but we'll just use indices
    # We maintain a list of active edge indices
    active_edges = list(range(len(all_edges)))
    
    def is_not_2_colorable(active_indices):
        """
        Checks if the hypergraph defined by active_indices is NOT 2-colorable.
        Returns True if NOT 2-colorable.
        """
        if not active_indices:
            return False # Empty is colorable
        
        # For n=9, we can iterate all 2^9 = 512 masks
        # Optimization: If an edge is monochromatic in a coloring, it's invalid.
        # We want to know if there EXISTS a coloring where NO edge is monochromatic.
        # If such a coloring exists, return False (it IS colorable).
        # If no such coloring exists, return True (it is NOT colorable).
        
        for c in range(1, (1 << n) - 1):
            # Skip all-0 and all-1 as they are always mono for any edge of size > 0
            # Actually, if n=9, range(1, 511) covers all non-trivial colorings.
            # Wait, c=0 is all 0s, c=511 is all 1s.
            # An edge is mono if (c & em) == 0 or (c & em) == em.
            # If we find a c where NO edge is mono, then it is 2-colorable.
            
            is_valid_coloring = True
            for idx in active_indices:
                em = edge_masks[idx]
                if (c & em) == 0 or (c & em) == em:
                    is_valid_coloring = False
                    break
            
            if is_valid_coloring:
                return False # Found a valid coloring, so it IS 2-colorable
                
        return True # No valid coloring found, so it is NOT 2-colorable

    best_edges_count = len(all_edges)
    best_active = active_edges[:]

    # Greedy reduction
    # We try to remove edges one by one.
    # We repeat this process with different random orders to escape local minima.
    
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        # Start with a full copy or a previously reduced set?
        # Starting from full is safer for finding a global minimum via random restarts.
        # But starting from the best found so far might be faster?
        # Let's start from full for the first few, then from best.
        current_active = active_edges[:]
        
        # Shuffle to randomize the order of removal
        random.shuffle(current_active)
        
        # We iterate through the shuffled list.
        # We need to check if removing the edge keeps the property.
        # If we remove it, we must check the new set.
        
        # To be efficient, we can just iterate through the current set and try to remove each.
        # However, removing one edge changes the set for the next check.
        # Standard greedy:
        # for i in range(len(current_active)):
        #    try to remove current_active[i]
        #    if still not 2-colorable, remove it.
        #    else keep it.
        
        # But the list changes size.
        # Let's use a while loop or iterate over a copy and filter.
        
        # Efficient approach:
        # We have a list of indices. We want to minimize its size.
        # We iterate through the list.
        # If we decide to remove an element, we just don't include it in the next list.
        
        new_active = []
        for idx in current_active:
            # Try to remove idx
            # The candidate set is new_active + [idx] (if we keep it) vs new_active (if we remove it)
            # We want to remove it if possible.
            # So we check if new_active is NOT 2-colorable.
            # If new_active is NOT 2-colorable, then idx is redundant, so we skip adding it.
            # If new_active IS 2-colorable, then idx is needed (or at least, removing it broke the property), so we add it.
            
            # Check new_active
            if is_not_2_colorable(new_active):
                # It's still not 2-colorable without idx, so idx is redundant.
                pass
            else:
                # It became 2-colorable without idx, so we need idx.
                new_active.append(idx)
                
        # After processing all, new_active is a minimal set (with respect to the order).
        if len(new_active) < best_edges_count:
            best_edges_count = len(new_active)
            best_active = new_active[:]
            
        # Optional: Use best_active as the starting point for the next trial?
        # This might get stuck. Random restarts from full are better for exploration.
        # But we can mix.
        if trial % 5 == 4:
            # Every 5th trial, start from the best found so far to polish
            current_active = best_active[:]
            random.shuffle(current_active)
            # ... run the same greedy logic on current_active ...
            # For simplicity, let's stick to random restarts from full for now, 
            # as 126 edges is small enough.
            
    # Convert best_active indices back to edges
    best_edges = [list(all_edges[i]) for i in best_active]
    
    return {'n': n, 'edges': best_edges}
