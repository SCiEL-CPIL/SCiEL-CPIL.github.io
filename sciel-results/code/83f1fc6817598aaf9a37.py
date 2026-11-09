import time
import sys
from itertools import combinations

def _greedy_set_cover(n, deadline):
    """
    Finds a small set of 5-edges on n vertices that covers all 2-colorings.
    Covers all colorings means: for every coloring, there is a monochromatic edge.
    """
    num_colorings = 1 << n
    full_mask = (1 << num_colorings) - 1
    
    # Generate all 5-edges
    edges = list(combinations(range(n), 5))
    
    # Calculate mask for each edge
    # Mask is an integer of num_colorings bits.
    # Bit c is 1 if edge is monochromatic in coloring c.
    edge_masks = []
    for e in edges:
        mask = 0
        # Iterate over all colorings? No, that's 2^n.
        # Better: Iterate over subsets of the edge?
        # An edge is monochromatic if all 5 vertices are 0 or all 5 are 1.
        # Let vs = list(e).
        # Coloring c is determined by bits of c.
        # Condition: (c & (1<<v) == 0 for all v in e) OR (c & (1<<v) != 0 for all v in e).
        # We can iterate over all 2^n colorings and check.
        # n=9 -> 512 colorings. 126 edges. 126 * 512 = 64k checks. Fast.
        # n=10 -> 1024 colorings. 252 edges. 252 * 1024 = 258k checks. Fast.
        
        # Optimization:
        # The colorings where e is all 0:
        # The bits for vertices in e must be 0. The other n-5 bits can be anything.
        # There are 2^(n-5) such colorings.
        # Similarly for all 1.
        
        vs = list(e)
        vs_set = set(vs)
        rest = [v for v in range(n) if v not in vs_set]
        
        m = 0
        # All 0 case
        for p in range(1 << (n - 5)):
            # Construct coloring index where vs are 0
            idx = 0
            for i, v in enumerate(rest):
                if (p >> i) & 1:
                    idx |= (1 << v)
            m |= (1 << idx)
            
            # All 1 case
            idx1 = 0
            for v in vs:
                idx1 |= (1 << v)
            for i, v in enumerate(rest):
                if (p >> i) & 1:
                    idx1 |= (1 << v)
            m |= (1 << idx1)
        
        edge_masks.append(m)
    
    # Greedy Set Cover
    uncovered = full_mask
    chosen_indices = []
    remaining = list(range(len(edges)))
    
    while uncovered:
        if time.time() > deadline:
            break
        best_idx = -1
        best_count = 0
        for idx in remaining:
            c = (uncovered & edge_masks[idx]).bit_count()
            if c > best_count:
                best_count = c
                best_idx = idx
                if best_count == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        chosen_indices.append(best_idx)
        uncovered &= ~edge_masks[best_idx]
        remaining.remove(best_idx)
        
    # Local Pruning
    # Maintain cover counts
    cover_counts = [0] * num_colorings
    for idx in chosen_indices:
        m = edge_masks[idx]
        # Add to counts
        # Efficient addition of bits?
        # Just iterate over set bits?
        # Or use a helper to add mask to array of counts.
        # Since num_colorings is small (<= 1024), we can just loop.
        for c in range(num_colorings):
            if (m >> c) & 1:
                cover_counts[c] += 1
                
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(chosen_indices)):
            idx = chosen_indices[i]
            m = edge_masks[idx]
            # Check if removing idx leaves any uncovered
            # i.e., if any cover_counts[c] becomes 0 where m has bit c
            # We can check: if (uncovered_by_idx & uncovered_after_removal) != 0?
            # Simpler: 
            # Temporarily decrement
            can_remove = True
            for c in range(num_colorings):
                if (m >> c) & 1:
                    if cover_counts[c] == 1:
                        can_remove = False
                        break
            if can_remove:
                # Remove it
                for c in range(num_colorings):
                    if (m >> c) & 1:
                        cover_counts[c] -= 1
                chosen_indices.pop(i)
                improved = True
                break # Restart loop
        
    if uncovered == 0:
        return [list(edges[i]) for i in chosen_indices]
    return None

def construct():
    start = time.time()
    deadline = start + 130.0 # Safety margin
    
    best_edges = None
    best_count = float('inf')
    
    # Try n=9
    res9 = _greedy_set_cover(9, deadline)
    if res9 is not None:
        if len(res9) < best_count:
            best_count = len(res9)
            best_edges = res9
            best_n = 9
            
    # Try n=10
    res10 = _greedy_set_cover(10, deadline)
    if res10 is not None:
        if len(res10) < best_count:
            best_count = len(res10)
            best_edges = res10
            best_n = 10
            
    # Fallback to K9 if greedy fails or returns too many
    if best_edges is None or best_count > 100:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)
        
    return {'n': best_n, 'edges': best_edges}
