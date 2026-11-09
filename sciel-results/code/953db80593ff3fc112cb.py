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
            
        # Pruning
        if not chosen_indices:
            continue
            
        # Maintain counts of how many times each coloring is covered
        counts = [0] * num_colorings
        for idx in chosen_indices:
            m = edge_masks[idx]
            # Iterate over set bits
            # Since m is a large int, we can iterate over the 2048 bits
            # Or use bit manipulation
            temp_m = m
            while temp_m:
                # Get index of lowest set bit
                lsb = temp_m & -temp_m
                c_idx = lsb.bit_length() - 1
                counts[c_idx] += 1
                temp_m ^= lsb
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle chosen indices to try removal in different orders
            chosen_indices_copy = chosen_indices[:]
            random.shuffle(chosen_indices_copy)
            
            for i in range(len(chosen_indices_copy)):
                if time.time() > deadline:
                    break
                idx = chosen_indices_copy[i]
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
                        
                    chosen_indices.remove(idx)
                    improved = True
                    break # Restart loop
                    
        # Try 1-for-2 swaps: remove 2 edges, add 1 edge to cover the gap
        if time.time() < deadline and len(chosen_indices) > 51:
            chosen_set = set(chosen_indices)
            for i in range(len(chosen_indices)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(chosen_indices)):
                    if time.time() > deadline:
                        break
                    # Try removing edges i and j
                    m_i = edge_masks[chosen_indices[i]]
                    m_j = edge_masks[chosen_indices[j]]
                    # Check if any coloring is only covered by these two
                    # If so, we need an edge that covers all such colorings
                    # Count coverings without these two
                    # This is expensive, so only do it for promising pairs
                    # For now, skip this optimization due to time constraints
                    
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
