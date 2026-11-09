import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 11
    num_colorings = 1 << n
    full_mask = (1 << num_colorings) - 1
    
    # Generate all edges
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute masks for each edge
    # mask[e] is a bitset of colorings where edge e is monochromatic
    edge_masks = [0] * num_edges
    for i, e in enumerate(all_edges):
        mask_vs = 0
        for v in e:
            mask_vs |= (1 << v)
        mask = 0
        for c in range(num_colorings):
            if (c & mask_vs) == 0 or (c & mask_vs) == mask_vs:
                mask |= (1 << c)
        edge_masks[i] = mask
        
    best_edges_count = num_edges
    best_chosen_indices = None
    
    # Run multiple attempts with randomization
    # We want to find a cover of size 51
    # Greedy set cover + pruning is a good heuristic
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Shuffle edges to break symmetry in greedy choice
        indices = list(range(num_edges))
        random.shuffle(indices)
        
        # Greedy Set Cover
        uncovered = full_mask
        chosen_indices = []
        remaining = set(indices)
        
        while uncovered:
            best_idx = -1
            best_count = -1
            # To add randomness, we can pick from top K candidates
            # But for speed, just pick best, or random among top few
            candidates = []
            for idx in remaining:
                c = (uncovered & edge_masks[idx]).bit_count()
                if c > 0:
                    candidates.append((c, idx))
            
            if not candidates:
                break
                
            # Sort by count descending
            candidates.sort(reverse=True)
            
            # Pick randomly from top 5 to vary the solution
            top_k = min(5, len(candidates))
            best_c, best_idx = random.choice(candidates[:top_k])
            
            chosen_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            remaining.remove(best_idx)
        
        # Pruning - more systematic local search
        if not chosen_indices:
            continue
            
        # Maintain counts of how many times each coloring is covered
        counts = [0] * num_colorings
        for idx in chosen_indices:
            m = edge_masks[idx]
            # Iterate over set bits
            temp_m = m
            while temp_m:
                lsb = temp_m & -temp_m
                c_idx = lsb.bit_length() - 1
                counts[c_idx] += 1
                temp_m ^= lsb
                
        # Try multiple rounds of removal with different orderings
        improved = True
        max_rounds = 50
        for _ in range(max_rounds):
            if not improved:
                break
            if time.time() > deadline:
                break
            improved = False
            
            # Try removing each edge in the current set
            for i in range(len(chosen_indices)):
                if time.time() > deadline:
                    break
                idx = chosen_indices[i]
                m = edge_masks[idx]
                can_remove = True
                temp_m = m
                while temp_m:
                    lsb = temp_m & -temp_m
                    c_idx = lsb.bit_length() - 1
                    if counts[c_idx] == 1:
                        can_remove = False
                        break
                    temp_m ^= lsb
                    
                if can_remove:
                    # Remove it
                    temp_m = m
                    while temp_m:
                        lsb = temp_m & -temp_m
                        c_idx = lsb.bit_length() - 1
                        counts[c_idx] -= 1
                        temp_m ^= lsb
                        
                    chosen_indices.pop(i)
                    improved = True
                    break  # Restart loop with updated list
        
        # Additional local search: try swapping one edge for another to reduce count
        if time.time() < deadline:
            for _ in range(20):  # Limited swaps
                if time.time() > deadline:
                    break
                
                # Try to find an edge that can be removed and replaced with a better one
                # This is more complex, so we'll just do simple removal attempts
                pass
        
        if len(chosen_indices) < best_edges_count:
            best_edges_count = len(chosen_indices)
            best_chosen_indices = chosen_indices[:]
            if best_edges_count <= 51:
                break
                
    if best_chosen_indices is None:
        # Fallback: return a known valid structure or the first greedy result
        # If we didn't find anything, we must return something valid.
        # The greedy cover is always valid (covers all colorings).
        # So we should keep the first valid greedy result.
        # Let's re-run one deterministic greedy to be safe if best is None
        indices = list(range(num_edges))
        uncovered = full_mask
        chosen_indices = []
        remaining = set(indices)
        while uncovered:
            best_idx = -1
            best_count = -1
            for idx in remaining:
                c = (uncovered & edge_masks[idx]).bit_count()
                if c > best_count:
                    best_count = c
                    best_idx = idx
            if best_idx == -1: break
            chosen_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            remaining.remove(best_idx)
        best_chosen_indices = chosen_indices
        
    result_edges = [list(all_edges[i]) for i in best_chosen_indices]
    return {'n': n, 'edges': result_edges}
