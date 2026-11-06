import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Start with n=8
    n = 8
    all_edges = [list(e) for e in combinations(range(n), 5)]
    num_edges = len(all_edges) # 56
    
    # Precompute edge masks
    edge_masks = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Precompute for each coloring c, which edges are monochromatic?
    # bad_masks[c] is a bitmask of edges that are monochromatic under coloring c.
    # If (bad_masks[c] & active_mask) == 0, then coloring c is valid for the active hypergraph.
    # So if there exists c such that (bad_masks[c] & active_mask) == 0, the hypergraph is 2-colorable.
    
    bad_masks = [0] * (1 << n)
    for c in range(1 << n):
        bm = 0
        for i in range(num_edges):
            mask_e = edge_masks[i]
            inter = c & mask_e
            if inter == 0 or inter == mask_e:
                bm |= (1 << i)
        bad_masks[c] = bm

    def is_non_2_colorable(active_mask):
        # Check if any coloring c has no monochromatic edges in active_mask
        for c in range(1 << n):
            if (bad_masks[c] & active_mask) == 0:
                return False # Found a valid coloring, so it IS 2-colorable
        return True # No valid coloring found, so it is NON-2-colorable

    # Initial active mask: all edges
    active_mask = (1 << num_edges) - 1
    
    # Check if K_8^5 is non-2-colorable
    if not is_non_2_colorable(active_mask):
        # If it is 2-colorable, fall back to n=9
        n = 9
        all_edges = [list(e) for e in combinations(range(n), 5)]
        num_edges = len(all_edges)
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        bad_masks = [0] * (1 << n)
        for c in range(1 << n):
            bm = 0
            for i in range(num_edges):
                mask_e = edge_masks[i]
                inter = c & mask_e
                if inter == 0 or inter == mask_e:
                    bm |= (1 << i)
            bad_masks[c] = bm
            
        def is_non_2_colorable(active_mask):
            for c in range(1 << n):
                if (bad_masks[c] & active_mask) == 0:
                    return False
            return True

        active_mask = (1 << num_edges) - 1
        # Assume K_9^5 is non-2-colorable. If not, we are in trouble, but it is known to be.

    best_mask = active_mask
    
    # Greedy deletion
    # We try to remove edges.
    # To speed up, we can check edges in random order.
    
    for _ in range(50): # Passes
        if time.time() > deadline:
            break
        
        changed = False
        # Get list of active edges
        active_indices = [i for i in range(num_edges) if best_mask & (1 << i)]
        random.shuffle(active_indices)
        
        for i in active_indices:
            if time.time() > deadline:
                break
            
            # Try removing i
            new_mask = best_mask & ~(1 << i)
            
            if is_non_2_colorable(new_mask):
                # Still non-2-colorable, so remove i
                best_mask = new_mask
                changed = True
        
        if not changed:
            break
            
        if bin(best_mask).count('1') <= 51:
            break

    final_edges = [all_edges[i] for i in range(num_edges) if best_mask & (1 << i)]
    
    return {'n': n, 'edges': final_edges}
