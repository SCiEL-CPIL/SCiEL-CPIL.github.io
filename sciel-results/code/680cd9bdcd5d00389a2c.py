import time
import random
from itertools import combinations

def _edge_kills_mask(n, edge):
    """
    Return a bitmask of length 2^n where bit i is set if coloring i
    (represented as an n-bit integer) makes the given edge monochromatic.
    """
    mask = 0
    edge_set = set(edge)
    other_verts = [v for v in range(n) if v not in edge_set]
    
    for color_val in [0, 1]:
        for other_mask in range(1 << len(other_verts)):
            c = 0
            for v in edge:
                c |= (color_val << v)
            for j, v in enumerate(other_verts):
                if (other_mask >> j) & 1:
                    c |= (1 << v)
            mask |= (1 << c)
    return mask

def construct():
    deadline = time.time() + 145
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Try different n values
    # n=9 is 126 edges, n=10 is 252, n=11 is 462, n=12 is 792, n=13 is 1287, n=14 is 2002
    for n in range(9, 15):
        if time.time() > deadline - 20:
            break
        
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        
        # Precompute kill masks
        kill_masks = []
        for e in all_edges:
            kill_masks.append(_edge_kills_mask(n, e))
        
        # Run multiple greedy set cover attempts
        # We want to find the minimum number of edges
        # Greedy is approximate, so random restarts help
        
        best_n_cover = None
        best_n_count = float('inf')
        
        # Number of restarts depends on time
        # For small n, we can do more
        restarts = 5 if n <= 11 else 2
        
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
            
            # Randomize edge order
            indices = list(range(num_edges))
            random.shuffle(indices)
            
            uncovered = (1 << (1 << n)) - 1
            chosen = []
            used = [False] * num_edges
            
            while uncovered:
                if time.time() > deadline - 5:
                    break
                best_idx = -1
                best_count_new = -1
                for idx in indices:
                    if used[idx]:
                        continue
                    new_cover = kill_masks[idx] & uncovered
                    cnt = new_cover.bit_count()
                    if cnt > best_count_new:
                        best_count_new = cnt
                        best_idx = idx
                
                if best_idx == -1:
                    break
                
                chosen.append(best_idx)
                used[best_idx] = True
                uncovered &= ~kill_masks[best_idx]
            
            if not uncovered:
                current_count = len(chosen)
                if current_count < best_n_count:
                    best_n_count = current_count
                    best_n_cover = chosen[:]
        
        if best_n_cover is not None:
            # Local search: remove redundant edges
            current_chosen = best_n_cover[:]
            current_set = set(best_n_cover)
            
            # Recompute uncovered mask for current set?
            # Or just check if removing an edge leaves uncovered=0?
            # We need the mask of the current set.
            # Let's maintain the mask.
            
            # Actually, we can just try to remove each edge and check if the union of the rest covers all.
            # This is expensive if done naively.
            # Better: Maintain `current_mask`.
            current_mask = 0
            for idx in current_chosen:
                current_mask |= kill_masks[idx]
            
            changed = True
            while changed and time.time() < deadline - 2:
                changed = False
                for i in range(len(current_chosen)):
                    idx = current_chosen[i]
                    # Try removing idx
                    # new_mask = current_mask ^ kill_masks[idx]? No, kill_masks might overlap.
                    # We need to check if (current_mask & ~kill_masks[idx]) == current_mask?
                    # No, we need to check if the union of the OTHERS covers all.
                    # This is hard to do incrementally with just one mask if overlaps exist.
                    
                    # Alternative: Just recompute the mask for the subset?
                    # If size is ~50, recomputing is 50 ORs of big ints. Fast.
                    
                    temp_chosen = current_chosen[:i] + current_chosen[i+1:]
                    temp_mask = 0
                    for j in temp_chosen:
                        temp_mask |= kill_masks[j]
                    
                    if temp_mask == (1 << (1 << n)) - 1:
                        # Can remove
                        current_chosen = temp_chosen
                        current_mask = temp_mask
                        changed = True
                        break # Restart loop with new list
            
            if len(current_chosen) < best_count:
                best_count = len(current_chosen)
                best_n = n
                best_edges = [list(all_edges[i]) for i in current_chosen]
        
    # Fallback
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
        
    return {'n': best_n, 'edges': best_edges}
