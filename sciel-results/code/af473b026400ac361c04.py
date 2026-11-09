import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    all_blocks = []
    block_masks = []
    
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(all_blocks)
    best_solution = None
    best_size = float('inf')
    
    # Helper to count bits
    def count_bits(x):
        return x.bit_count()

    def greedy_cover(random_seed):
        rng = random.Random(random_seed)
        current_uncovered = (1 << num_t_subsets) - 1
        chosen_indices = []
        used = [False] * num_blocks
        
        while current_uncovered != 0:
            if time.time() > deadline:
                return None
            
            best_gain = -1
            best_block_idx = -1
            # Collect top candidates to allow random tie-breaking
            candidates = []
            
            for i in range(num_blocks):
                if used[i]:
                    continue
                intersection = block_masks[i] & current_uncovered
                gain = count_bits(intersection)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if best_block_idx == -1:
                break
                
            # Randomly pick from top candidates to diversify
            chosen_idx = rng.choice(candidates)
            chosen_indices.append(chosen_idx)
            used[chosen_idx] = True
            current_uncovered &= ~block_masks[chosen_idx]
            
        return chosen_indices

    def reduce_cover(chosen_indices):
        if not chosen_indices:
            return []
            
        # Count coverage for each subset
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Helper to update counts
        def update_counts(idx, delta):
            mask = block_masks[idx]
            # Extract set bits
            # Since mask is a Python int, we can iterate through set bits
            # Or use numpy if we converted, but int is fine for 35 bits
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                idx_bit = lsb.bit_length() - 1
                cover_count[idx_bit] += delta
                m ^= lsb
                
        # Initial counts
        for idx in chosen_indices:
            update_counts(idx, 1)
            
        kept_indices = []
        # Try to remove each block
        # Note: Order of removal matters? Not really, we just check redundancy.
        # We iterate through the list. If we remove one, it might make another redundant.
        # So we should probably iterate multiple times or just check once.
        # Checking once is usually sufficient for greedy results.
        
        for idx in chosen_indices:
            # Check if idx is redundant
            # It is redundant if all subsets in mask[idx] have count > 1
            mask = block_masks[idx]
            m = mask
            is_redundant = True
            while m:
                lsb = m & -m
                idx_bit = lsb.bit_length() - 1
                if cover_count[idx_bit] <= 1:
                    is_redundant = False
                    break
                m ^= lsb
                
            if is_redundant:
                # Remove it
                update_counts(idx, -1)
            else:
                kept_indices.append(idx)
                
        return kept_indices

    # Run multiple restarts
    seed = 0
    while time.time() < deadline:
        indices = greedy_cover(seed)
        if indices is None:
            break
            
        reduced_indices = reduce_cover(indices)
        
        if len(reduced_indices) < best_size:
            best_size = len(reduced_indices)
            best_solution = [list(all_blocks[i]) for i in reduced_indices]
            
        seed += 1

    if best_solution is None:
        # Fallback: simple greedy without reduction
        indices = greedy_cover(0)
        if indices:
            best_solution = [list(all_blocks[i]) for i in indices]
        else:
            return []
            
    return best_solution
