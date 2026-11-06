import random
from itertools import combinations

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges.
    
    Approach: 
    1. Fix n to a small value (e.g., 11).
    2. Use a greedy algorithm that iteratively adds the edge which invalidates the most 
       currently valid 2-colorings.
    3. A coloring is valid if no edge is monochromatic.
    4. We represent colorings as bitmasks.
    """
    # Try different n values, starting from small ones where computation is fast.
    # We want to minimize edges. Smaller n usually allows fewer edges to force unsat?
    # Or larger n allows more flexibility?
    # Actually, for fixed uniformity k, the minimum edges for non-2-colorability 
    # (m(k)) is related to n. 
    # The task says "best known construction has 51 edges".
    # Let's try n=11 first.
    
    def get_all_colorings(n):
        return list(range(1 << n))
        
    def is_monochromatic(coloring, edge):
        # Check if all vertices in edge have the same bit in coloring
        # edge is a tuple of vertex indices
        # We can extract the bits and check if they are all 0 or all 1
        mask = 0
        for v in edge:
            mask |= (coloring >> v) & 1
        # If mask is 0, all bits are 0.
        # If mask has all bits set to 1 corresponding to the edge size?
        # Better: check if all bits are equal.
        # Extract bits:
        bits = [(coloring >> v) & 1 for v in edge]
        return all(b == bits[0] for b in bits)
        
    def count_invalidations(valid_colorings, edge):
        count = 0
        for c in valid_colorings:
            if is_monochromatic(c, edge):
                count += 1
        return count

    def construct_for_n(n):
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges) # Randomize to avoid bias? Or just iterate all.
        
        # We will iterate through ALL edges to find the best one each time.
        # This is O(N_edges * 2^n * |E_current|) which might be heavy if we re-count.
        # But we can just count for each edge against the current valid_colorings.
        
        valid_colorings = get_all_colorings(n)
        selected_edges = []
        
        # Optimization: Precompute bitmasks for edges?
        # For an edge e, monochromatic means:
        # (coloring & mask_e) == 0 OR (coloring & mask_e) == mask_e
        # where mask_e has 1s at positions in e.
        
        edge_masks = {}
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks[e] = mask
            
        def is_mono_fast(coloring, mask):
            return (coloring & mask) == 0 or (coloring & mask) == mask

        while valid_colorings:
            best_edge = None
            best_count = 0
            # Find the edge that invalidates the most current valid colorings
            for e in all_edges:
                if e in selected_edges:
                    continue
                mask = edge_masks[e]
                count = 0
                for c in valid_colorings:
                    if is_mono_fast(c, mask):
                        count += 1
                if count > best_count:
                    best_count = count
                    best_edge = e
                    
            if best_edge is None or best_count == 0:
                # No edge can reduce the solution space further?
                # This implies we are stuck in a satisfiable state.
                # Break to avoid infinite loop.
                break
                
            selected_edges.append(best_edge)
            mask = edge_masks[best_edge]
            # Update valid colorings: remove those that are monochromatic on best_edge
            valid_colorings = [c for c in valid_colorings if not is_mono_fast(c, mask)]
            
        if not valid_colorings:
            return {'n': n, 'edges': [list(e) for e in selected_edges]}
        else:
            return None

    # Try n=11
    res = construct_for_n(11)
    if res:
        return res
        
    # Try n=10
    res = construct_for_n(10)
    if res:
        return res
        
    # Try n=12 (slower)
    res = construct_for_n(12)
    if res:
        return res

    # Fallback: return a known valid structure or empty if failed (should not happen)
    # A complete 5-uniform hypergraph on 9 vertices is definitely non-2-colorable?
    # Actually, for n=9, K_9^5 is not 2-colorable?
    # Property B is not guaranteed for small n?
    # Wait, the task says "best known construction has 51 edges".
    # If we fail, we return a valid one.
    # Let's use n=9, all edges.
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
