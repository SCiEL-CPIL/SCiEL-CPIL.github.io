import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def is_not_2_colorable(n, edge_masks):
        """
        Returns True if the hypergraph is NOT 2-colorable.
        i.e., For every 2-coloring, there exists a monochromatic edge.
        Uses bitmask iteration over all 2^n colorings.
        """
        if not edge_masks:
            return False
        
        # Iterate through all possible colorings (bitmasks from 0 to 2^n - 1)
        # Note: 0 and (1<<n)-1 are all-one-color, so they will always have mono edges if E>0.
        # We can skip them or include them, doesn't matter much for correctness, 
        # but skipping saves 2 iterations.
        
        for mask in range(1 << n):
            # Check if this coloring has NO monochromatic edges
            # If we find one, it IS 2-colorable, so return False.
            has_mono = False
            for em in edge_masks:
                # Mono if (mask & em) is 0 (all 0s) or em (all 1s)
                if (mask & em) == 0 or (mask & em) == em:
                    has_mono = True
                    break
            if not has_mono:
                return False
        return True

    def greedy_remove(n, edge_masks):
        """
        Greedily removes edges while maintaining non-2-colorability.
        Returns the reduced list of edge masks.
        """
        current_masks = list(edge_masks)
        if not current_masks:
            return current_masks
            
        # We will try to remove edges in random order multiple times
        # to escape local minima.
        # For efficiency, we do a few passes.
        
        # Optimization: If we have many edges, checking every single removal
        # is expensive. 
        # Cost of one check: 2^n * E.
        # Total cost of one pass: E * (2^n * E) = E^2 * 2^n.
        # For n=10, E=100: 100^2 * 1024 = 10M ops. ~1s.
        # For n=12, E=100: 100^2 * 4096 = 40M ops. ~4s.
        
        # We'll do a single pass with random shuffle first.
        indices = list(range(len(current_masks)))
        
        # We can't just remove if it fails, we need to be careful.
        # Standard greedy: 
        # for i in shuffled_indices:
        #    if remove(current_masks[i]) is still non-2-colorable:
        #        remove it
        
        # To speed up, we can check time.
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            indices = list(range(len(current_masks)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5:
                    break
                
                # Try removing current_masks[idx]
                # Create a new list without this index
                # Optimization: Instead of creating a new list every time,
                # we can just pass a flag or modify in place if we backtrack.
                # But creating a list of size E-1 is fast for E<200.
                
                test_masks = current_masks[:idx] + current_masks[idx+1:]
                
                if is_not_2_colorable(n, test_masks):
                    current_masks = test_masks
                    changed = True
                    # If we removed one, indices are now shifted?
                    # Actually, we are iterating over a fixed 'indices' list from the start of the pass.
                    # If we remove an element, the indices in 'indices' that are > idx are now invalid.
                    # This is a problem with the simple loop.
                    
                    # Better approach for the pass:
                    # Iterate from end to start? No, random order.
                    # If we remove an item, we should just break the pass and restart with a new shuffle?
                    # Or handle index shifts.
                    # Simpler: Just break the inner loop and restart the while loop if we removed anything?
                    # That might be slow if we remove many.
                    
                    # Let's restart the pass if we remove an edge to keep logic simple and avoid index bugs.
                    break
            
        return current_masks

    def generate_random_non_2_colorable(n, max_edges=200):
        """
        Generates a random hypergraph that is not 2-colorable by adding edges until it is.
        """
        all_edges = list(itertools.combinations(range(n), 5))
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Shuffle all edges
        random.shuffle(edge_masks)
        
        current_masks = []
        # Add edges one by one
        for em in edge_masks:
            current_masks.append(em)
            # Check if it's now non-2-colorable
            # Optimization: Only check if len(current_masks) is large enough?
            # Min edges for non-2-colorable is 29.
            if len(current_masks) >= 29:
                if is_not_2_colorable(n, current_masks):
                    return current_masks
            if time.time() > deadline - 10:
                return None
            if len(current_masks) > max_edges:
                return None
                
        return None

    best_edges = None
    best_m = 200
    
    # Strategy 1: Start with K_9^(5)
    if time.time() < deadline - 20:
        n = 9
        all_edges = list(itertools.combinations(range(n), 5))
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        # Verify K9 is non-2-colorable (it is)
        if is_not_2_colorable(n, edge_masks):
            reduced = greedy_remove(n, edge_masks)
            if reduced:
                m = len(reduced)
                if m < best_m:
                    best_m = m
                    # Convert back to list of lists for output
                    # We need to map masks back to edges. 
                    # Since we started with all edges, we can just use the original list and filter.
                    # But greedy_remove returns masks. We need the actual vertex lists.
                    # Let's store (mask, edge_tuple) pairs.
                    
    # To make output easy, let's restructure to keep (mask, edge) pairs.
    
    def greedy_remove_pairs(n, pairs):
        """
        pairs: list of (mask, edge_tuple)
        Returns list of edge_tuples.
        """
        current_pairs = list(pairs)
        if not current_pairs:
            return []
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            indices = list(range(len(current_pairs)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5:
                    break
                
                # Try removing current_pairs[idx]
                test_masks = [p[0] for p in current_pairs[:idx]] + [p[0] for p in current_pairs[idx+1:]]
                
                if is_not_2_colorable(n, test_masks):
                    # Remove it
                    current_pairs.pop(idx)
                    changed = True
                    break # Restart pass to be safe with indices
                    
        return [p[1] for p in current_pairs]

    def generate_random_non_2_colorable_pairs(n, max_edges=200):
        all_edges = list(itertools.combinations(range(n), 5))
        pairs = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            pairs.append((mask, e))
        
        random.shuffle(pairs)
        
        current_pairs = []
        for p in pairs:
            current_pairs.append(p)
            if len(current_pairs) >= 29:
                masks = [x[0] for x in current_pairs]
                if is_not_2_colorable(n, masks):
                    return [x[1] for x in current_pairs]
            if time.time() > deadline - 10:
                return None
            if len(current_pairs) > max_edges:
                return None
        return None

    # Reset best
    best_edges = None
    best_m = 200

    # 1. Try n=9 with K9
    if time.time() < deadline - 15:
        n = 9
        pairs = []
        for e in itertools.combinations(range(n), 5):
            mask = 0
            for v in e:
                mask |= (1 << v)
            pairs.append((mask, e))
        
        if is_not_2_colorable(n, [p[0] for p in pairs]):
            reduced_edges = greedy_remove_pairs(n, pairs)
            if reduced_edges and len(reduced_edges) < best_m:
                best_m = len(reduced_edges)
                best_edges = [list(e) for e in reduced_edges]

    # 2. Try n=10 with random build
    if time.time() < deadline - 15 and best_m > 51:
        n = 10
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            edges = generate_random_non_2_colorable_pairs(n)
            if edges:
                # Greedy remove
                pairs = []
                for e in edges:
                    mask = 0
                    for v in e:
                        mask |= (1 << v)
                    pairs.append((mask, e))
                
                reduced_edges = greedy_remove_pairs(n, pairs)
                if reduced_edges and len(reduced_edges) < best_m:
                    best_m = len(reduced_edges)
                    best_edges = [list(e) for e in reduced_edges]
                    if best_m <= 51:
                        break

    # 3. Try n=11 with random build
    if time.time() < deadline - 15 and best_m > 51:
        n = 11
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            edges = generate_random_non_2_colorable_pairs(n)
            if edges:
                pairs = []
                for e in edges:
                    mask = 0
                    for v in e:
                        mask |= (1 << v)
                    pairs.append((mask, e))
                
                reduced_edges = greedy_remove_pairs(n, pairs)
                if reduced_edges and len(reduced_edges) < best_m:
                    best_m = len(reduced_edges)
                    best_edges = [list(e) for e in reduced_edges]
                    if best_m <= 51:
                        break

    # Fallback if nothing found (should not happen)
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_m = len(best_edges)

    return {'n': n, 'edges': best_edges}
