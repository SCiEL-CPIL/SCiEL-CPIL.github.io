import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin before 60s limit
    
    n_vertices = 12
    m = 40
    
    # Precompute all 2^12 = 4096 subsets as bitmasks
    all_masks = list(range(1 << n_vertices))
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << n_vertices)
    for mask in range(1, 1 << n_vertices):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Sort masks by (popcount, mask) for a structured initial ordering
    sorted_masks = sorted(all_masks, key=lambda x: (popcount[x], x))
    
    # Initial greedy construction:
    # Start with empty set, then singletons, then largest sets
    # This mirrors the current program's strategy but in bitmask form
    initial_order = [0]  # empty set
    for i in range(n_vertices):
        initial_order.append(1 << i)  # singletons
    # Add full set
    initial_order.append((1 << n_vertices) - 1)
    # Add size-11 sets
    for i in range(n_vertices):
        initial_order.append(((1 << n_vertices) - 1) ^ (1 << i))
    # Add size-10 sets until we have 40
    for mask in sorted_masks:
        if popcount[mask] == 10 and mask not in initial_order:
            initial_order.append(mask)
            if len(initial_order) == m:
                break
    
    # Ensure we have exactly m masks
    if len(initial_order) < m:
        for mask in sorted_masks:
            if mask not in initial_order:
                initial_order.append(mask)
                if len(initial_order) == m:
                    break
    
    selected = list(initial_order[:m])
    
    # Helper to count comparable pairs
    def count_comparable(family):
        n = len(family)
        cnt = 0
        for i in range(n):
            mi = family[i]
            for j in range(i + 1, n):
                mj = family[j]
                if (mi & mj) == mi and mi != mj:
                    cnt += 1
                elif (mi & mj) == mj and mi != mj:
                    cnt += 1
        return cnt
    
    current_score = count_comparable(selected)
    best_selected = list(selected)
    best_score = current_score
    
    # Local search: try swapping one selected mask with one unselected mask
    selected_set = set(selected)
    unselected = [mask for mask in all_masks if mask not in selected_set]
    
    # Precompute for speed: for each mask in selected, which unselected masks it's comparable with
    # Actually, let's just do a simple local search with time limit
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all possible swaps
        for i in range(m):
            if time.time() > deadline:
                break
            mi = selected[i]
            for mj in unselected:
                if time.time() > deadline:
                    break
                # Try swapping mi with mj
                new_selected = list(selected)
                new_selected[i] = mj
                
                # Compute score change incrementally
                # Remove mi's contributions, add mj's contributions
                old_contrib = 0
                for j in range(m):
                    if j == i:
                        continue
                    mj2 = selected[j]
                    if (mi & mj2) == mi and mi != mj2:
                        old_contrib += 1
                    elif (mi & mj2) == mj2 and mi != mj2:
                        old_contrib += 1
                
                new_contrib = 0
                for j in range(m):
                    if j == i:
                        continue
                    mj2 = selected[j]
                    if (mj & mj2) == mj and mj != mj2:
                        new_contrib += 1
                    elif (mj & mj2) == mj2 and mj != mj2:
                        new_contrib += 1
                
                delta = new_contrib - old_contrib
                if delta > 0:
                    # Accept the swap
                    selected[i] = mj
                    selected_set.discard(mi)
                    selected_set.add(mj)
                    unselected.remove(mi)
                    unselected.append(mj)
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                    break  # Restart the inner loop after a successful swap
    
    # Use the best found
    final_selected = best_selected
    
    # Convert to lists of integers
    result = []
    for mask in final_selected:
        subset = [i for i in range(n_vertices) if (mask >> i) & 1]
        result.append(subset)
    
    # Verify: exactly 40 distinct subsets
    if len(result) != 40:
        # Fallback
        result = [[i % 12] for i in range(40)]
    
    # Check distinctness
    if len(set(tuple(s) for s in result)) != 40:
        # Fallback
        result = [[i % 12] for i in range(40)]
    
    # Check types
    for s in result:
        for x in s:
            if not isinstance(x, int) or isinstance(x, bool):
                # Should not happen
                pass
    
    return result
