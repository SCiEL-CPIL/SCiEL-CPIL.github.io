import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Precomputation
    # Generate all t-subsets and map to masks
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_masks = []
    for sub in all_t_subsets:
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_subset_masks.append(mask)
    
    initial_uncovered_set = set(t_subset_masks)
    initial_uncovered_list = t_subset_masks[:]

    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    
    # Precompute covers for each k-subset as a set of t-subset masks
    block_covers = []
    for block in all_k_subsets:
        covered_masks = set()
        for t_sub in itertools.combinations(block, t):
            mask = 0
            for p in t_sub:
                mask |= (1 << p)
            covered_masks.add(mask)
        block_covers.append(covered_masks)
        
    # Precompute for each t-subset mask, the list of k-block indices that contain it
    t_mask_to_block_indices = {mask: [] for mask in t_subset_masks}
    
    for i, block in enumerate(all_k_subsets):
        for t_sub in itertools.combinations(block, t):
            mask = 0
            for p in t_sub:
                mask |= (1 << p)
            t_mask_to_block_indices[mask].append(i)
            
    # 2. Greedy with restarts
    best_blocks = None
    best_count = float('inf')
    
    # Deterministic seed for reproducibility in local testing, but random restarts are key
    # We don't set a global seed to allow different runs to explore different paths if the evaluator runs multiple times,
    # but within one call, we use the random module.
    
    while time.time() < deadline:
        uncovered_set = set(initial_uncovered_set)
        uncovered_list = initial_uncovered_list[:]
        blocks = []
        
        while uncovered_set:
            # Pick a random uncovered t-subset to focus on
            # This provides diversification compared to always picking min()
            target_mask = random.choice(uncovered_list)
            
            # Get candidate blocks that cover this target_mask
            candidates = t_mask_to_block_indices[target_mask]
            
            if not candidates:
                # Should not happen if covering number exists, but break to avoid infinite loop
                break
                
            # Calculate gain for each candidate
            max_gain = 0
            best_candidates = []
            
            # Optimization: We can short-circuit if max_gain reaches the size of uncovered_set?
            # No, because we want to find ALL best candidates to randomize among them.
            # But if we just want ONE, we can stop early if we find a block that covers everything.
            # However, randomizing among ties is better for escaping local optima.
            
            for idx in candidates:
                cover = block_covers[idx]
                # Intersection of two sets of integers
                gain = len(uncovered_set & cover)
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [idx]
                elif gain == max_gain:
                    best_candidates.append(idx)
                    
            if not best_candidates:
                break
                
            # Randomly select from best candidates to add diversity
            chosen_idx = random.choice(best_candidates)
            blocks.append(all_k_subsets[chosen_idx])
            
            # Update uncovered
            uncovered_set.difference_update(block_covers[chosen_idx])
            if uncovered_set:
                # Rebuild list for next random choice
                # This is O(N) where N is number of uncovered subsets
                uncovered_list = list(uncovered_set)
            else:
                uncovered_list = []

        if uncovered_set:
            # Incomplete covering in this restart
            continue
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    if best_blocks is None:
        # Fallback: Deterministic greedy (pick min uncovered, pick max gain, tie-break by first)
        uncovered_set = set(initial_uncovered_set)
        uncovered_list = initial_uncovered_list[:]
        blocks = []
        while uncovered_set:
            target_mask = min(uncovered_list)
            candidates = t_mask_to_block_indices[target_mask]
            
            max_gain = 0
            best_idx = candidates[0]
            
            for idx in candidates:
                gain = len(uncovered_set & block_covers[idx])
                if gain > max_gain:
                    max_gain = gain
                    best_idx = idx
            
            blocks.append(all_k_subsets[best_idx])
            uncovered_set.difference_update(block_covers[best_idx])
            if uncovered_set:
                uncovered_list = list(uncovered_set)
            else:
                break
                
        return [list(b) for b in blocks]
        
    return best_blocks
