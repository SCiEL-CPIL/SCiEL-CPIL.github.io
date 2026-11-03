import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin

    # 1. Precomputation
    # Generate all t-subsets and map to index
    all_t_subsets = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # Generate all k-blocks
    all_blocks = [tuple(c) for c in itertools.combinations(range(v), k)]
    num_blocks = len(all_blocks)
    
    # Precompute for each block: the indices of t-subsets it covers
    # Shape: (num_blocks, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    block_subsets = np.empty((num_blocks, n_t_in_k), dtype=np.int32)
    
    # Precompute t_to_blocks: list of sets of block indices for each t-subset
    t_to_blocks = [set() for _ in range(num_subsets)]
    
    for i, block in enumerate(all_blocks):
        for j, lc in enumerate(t_in_k):
            ts = tuple(block[p] for p in lc)
            idx = sub_index[ts]
            block_subsets[i, j] = idx
            t_to_blocks[idx].add(i)
            
    # Convert t_to_blocks to lists for faster iteration if needed, 
    # but sets are better for intersection. 
    # We will keep them as sets.
    
    # 2. Greedy Set Cover
    # Heuristic: Pick the uncovered subset with the fewest covering blocks.
    # Then pick the block covering it that covers the most other uncovered subsets.
    
    covered_mask = np.zeros(num_subsets, dtype=bool)
    selected_indices = []
    selected_set = set()
    
    # To speed up "fewest covering blocks" check, we can maintain counts of uncovered blocks per subset?
    # Or just compute on the fly. 
    # num_subsets is ~11k. Checking all is fast.
    # But we need to find the block with max gain among those covering the chosen subset.
    
    while not np.all(covered_mask):
        if time.time() > deadline - 10.0:
            break
            
        # Find uncovered subset with min degree
        uncovered = np.where(~covered_mask)[0]
        if len(uncovered) == 0:
            break
            
        # Find the one with fewest blocks covering it
        # We can precompute static degrees, but dynamic is better.
        # However, calculating dynamic degree for all uncovered subsets is O(N_subsets * avg_degree).
        # Let's just pick a random uncovered subset if we are slow, 
        # or the one with min static degree among uncovered.
        # Static degree is precomputed from t_to_blocks lengths.
        
        # Optimization: Precompute static degrees
        static_degrees = np.array([len(s) for s in t_to_blocks], dtype=np.int32)
        
        # Find uncovered subset with min static degree
        # We can't easily filter static_degrees by covered_mask without a loop or fancy numpy.
        # Let's just iterate a few random uncovered subsets and pick the best?
        # Or use a heap?
        # For 11k subsets, a linear scan to find min degree among uncovered is ~11k ops.
        # 300 blocks * 11k ops = 3.3M ops. Acceptable.
        
        min_deg = float('inf')
        best_sub_idx = -1
        
        # Optimization: If we have many uncovered, checking all is fine.
        for s_idx in uncovered:
            deg = static_degrees[s_idx]
            if deg < min_deg:
                min_deg = deg
                best_sub_idx = s_idx
                if deg == 1: # Can't get better
                    break

        if best_sub_idx == -1:
            break

        # Find best block covering best_sub_idx
        candidates = t_to_blocks[best_sub_idx]
        best_block_idx = -1
        best_gain = -1
        
        # We want to maximize the number of *other* uncovered subsets covered by the block.
        # Gain = count of uncovered subsets in block_subsets[b]
        
        for b_idx in candidates:
            if b_idx in selected_set:
                continue
                
            # Calculate gain
            subs = block_subsets[b_idx]
            # Count how many are not covered
            gain = np.sum(~covered_mask[subs])
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            # Fallback: pick any candidate
            if candidates:
                best_block_idx = next(iter(candidates))
            else:
                break
                
        # Select block
        selected_indices.append(best_block_idx)
        selected_set.add(best_block_idx)
        covered_mask[block_subsets[best_block_idx]] = True

    if time.time() > deadline - 5.0:
        return [list(all_blocks[i]) for i in selected_indices]

    # 3. Local Search
    # State: selected_set, coverage_count (int array)
    coverage_count = np.zeros(num_subsets, dtype=np.int32)
    for idx in selected_set:
        coverage_count[block_subsets[idx]] += 1
        
    def _is_redundant(block_idx):
        subs = block_subsets[block_idx]
        return np.all(coverage_count[subs] > 1)

    def _remove_block(block_idx):
        selected_set.remove(block_idx)
        subs = block_subsets[block_idx]
        coverage_count[subs] -= 1

    def _add_block(block_idx):
        selected_set.add(block_idx)
        subs = block_subsets[block_idx]
        coverage_count[subs] += 1

    rng = random.Random(42)
    improved = True
    
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Deletion
        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            # Iterate over a copy of the list to allow modification
            for idx in list(selected_set):
                if _is_redundant(idx):
                    _remove_block(idx)
                    improved = True
                    changed = True
                    break # Restart deletion loop as counts changed

        if time.time() > deadline - 2.0:
            break

        # Phase 2: 2-for-1 Swap
        # Try to replace two blocks with one
        # Sample random pairs
        num_selected = len(selected_set)
        if num_selected >= 2:
            pairs_to_try = min(50, num_selected * (num_selected - 1) // 2)
            for _ in range(pairs_to_try):
                if time.time() > deadline - 2.0:
                    break
                    
                # Pick two distinct random blocks
                b1, b2 = rng.sample(list(selected_set), 2)
                
                subs1 = block_subsets[b1]
                subs2 = block_subsets[b2]
                
                # Find unique subsets for b1 and b2
                # Unique to b1: in subs1, count==1, not in subs2
                # Unique to b2: in subs2, count==1, not in subs1
                
                # Efficient check:
                # Create a mask for subs2
                mask2 = np.zeros(num_subsets, dtype=bool)
                mask2[subs2] = True
                
                unique1_mask = (coverage_count[subs1] == 1) & ~mask2[subs1]
                unique1 = subs1[unique1_mask]
                
                mask1 = np.zeros(num_subsets, dtype=bool)
                mask1[subs1] = True
                
                unique2_mask = (coverage_count[subs2] == 1) & ~mask1[subs2]
                unique2 = subs2[unique2_mask]
                
                # Combine unique subsets
                all_unique = np.concatenate([unique1, unique2])
                
                if len(all_unique) == 0:
                    # One is redundant? Should have been caught.
                    continue
                
                # Find a block that covers all all_unique
                # Start with the smallest candidate set
                # Sort all_unique by size of t_to_blocks[subset]
                # This is expensive to sort every time. 
                # Just pick the first one, if candidate set is empty, try next?
                # Better: pick the one with smallest degree.
                
                # To be fast, just iterate all_unique and intersect
                # Start with a large set? No, start with smallest.
                # Let's just pick the first one and intersect with others.
                # If it fails, we might miss a solution, but it's a heuristic.
                
                # Optimization: pick the subset in all_unique with min degree
                min_deg_sub = -1
                min_deg = float('inf')
                for s in all_unique:
                    d = len(t_to_blocks[s])
                    if d < min_deg:
                        min_deg = d
                        min_deg_sub = s
                        
                candidates = set(t_to_blocks[min_deg_sub])
                
                for s in all_unique:
                    if s == min_deg_sub:
                        continue
                    candidates.intersection_update(t_to_blocks[s])
                    if not candidates:
                        break
                        
                # Exclude b1 and b2
                candidates.discard(b1)
                candidates.discard(b2)
                
                if candidates:
                    # Found a candidate!
                    # Pick one (random or best)
                    new_b = rng.choice(list(candidates))
                    
                    # Perform swap
                    _remove_block(b1)
                    _remove_block(b2)
                    _add_block(new_b)
                    improved = True
                    break # Restart main loop

        if time.time() > deadline - 2.0:
            break

        # Phase 3: 1-for-1 Swap (Shuffling)
        # Try to replace one block with another to escape local minima
        if len(selected_set) > 0:
            swap_attempts = 10
            for _ in range(swap_attempts):
                if time.time() > deadline - 2.0:
                    break
                    
                b_idx = rng.choice(list(selected_set))
                subs = block_subsets[b_idx]
                
                # Critical subsets: count == 1
                critical_mask = (coverage_count[subs] == 1)
                critical_subs = subs[critical_mask]
                
                if len(critical_subs) == 0:
                    # Redundant, should have been deleted
                    continue
                    
                # Find candidates covering all critical_subs
                # Pick min degree critical sub
                min_deg_sub = -1
                min_deg = float('inf')
                for s in critical_subs:
                    d = len(t_to_blocks[s])
                    if d < min_deg:
                        min_deg = d
                        min_deg_sub = s
                        
                candidates = set(t_to_blocks[min_deg_sub])
                
                for s in critical_subs:
                    if s == min_deg_sub:
                        continue
                    candidates.intersection_update(t_to_blocks[s])
                    if not candidates:
                        break
                        
                candidates.discard(b_idx)
                
                if candidates:
                    # Pick a candidate
                    # Heuristic: Pick one that covers most "redundant" subsets?
                    # Or just random.
                    new_b = rng.choice(list(candidates))
                    
                    _remove_block(b_idx)
                    _add_block(new_b)
                    improved = True
                    break # Restart main loop

    # Final check and return
    # Verify coverage
    final_coverage = np.zeros(num_subsets, dtype=bool)
    for idx in selected_set:
        final_coverage[block_subsets[idx]] = True
        
    if not np.all(final_coverage):
        # Fallback to greedy if local search broke something (shouldn't happen)
        return [list(all_blocks[i]) for i in selected_indices]
        
    return [list(all_blocks[i]) for i in selected_set]
