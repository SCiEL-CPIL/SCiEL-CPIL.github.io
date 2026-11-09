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
            
            # Pick a random uncovered t-subset to anchor the search
            # Find an uncovered subset index
            uncovered_list = []
            m = current_uncovered
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                uncovered_list.append(idx)
                m ^= lsb
            
            if not uncovered_list:
                break
            
            # Randomly select an uncovered subset
            target_idx = uncovered_list[rng.randrange(len(uncovered_list))]
            target_sub = all_t_subsets[target_idx]
            
            # Find all blocks that cover this target subset
            # A block covers target_sub if all elements of target_sub are in the block
            target_set = set(target_sub)
            candidates = []
            
            for i in range(num_blocks):
                if used[i]:
                    continue
                block = all_blocks[i]
                # Check if block covers target_sub
                if all(x in block for x in target_set):
                    intersection = block_masks[i] & current_uncovered
                    gain = count_bits(intersection)
                    candidates.append((gain, i))
            
            if not candidates:
                break
            
            # Sort by gain descending, pick from top
            candidates.sort(key=lambda x: -x[0])
            max_gain = candidates[0][0]
            
            # Pick randomly from top-gain candidates (diversification)
            top_candidates = [c[1] for c in candidates if c[0] == max_gain]
            # Limit to a reasonable number to avoid too much randomness
            if len(top_candidates) > 20:
                top_candidates = rng.sample(top_candidates, 20)
            
            chosen_idx = rng.choice(top_candidates)
            chosen_indices.append(chosen_idx)
            used[chosen_idx] = True
            current_uncovered &= ~block_masks[chosen_idx]
            
        return chosen_indices

    def reduce_cover(chosen_indices):
        if not chosen_indices:
            return []
            
        # Count coverage for each subset using numpy for speed
        # Convert block_masks to a list of lists of indices for faster iteration
        # Pre-compute the list of covered indices for each block
        mask_indices_list = []
        for mask in block_masks:
            indices_in_mask = []
            m = mask
            while m:
                lsb = m & -m
                idx_bit = lsb.bit_length() - 1
                indices_in_mask.append(idx_bit)
                m ^= lsb
            mask_indices_list.append(indices_in_mask)
        
        # Count coverage for each subset
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Initial counts
        for idx in chosen_indices:
            for idx_bit in mask_indices_list[idx]:
                cover_count[idx_bit] += 1
        
        kept_indices = []
        
        for idx in chosen_indices:
            # Check if idx is redundant
            # It is redundant if all subsets in mask[idx] have count > 1
            is_redundant = True
            for idx_bit in mask_indices_list[idx]:
                if cover_count[idx_bit] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                for idx_bit in mask_indices_list[idx]:
                    cover_count[idx_bit] -= 1
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
        # Also try a few extra seeds with different randomization strategies
        if seed % 50 == 0 and time.time() < deadline:
            # Try with slightly different random seeds for diversity
            for extra_seed in range(5):
                if time.time() > deadline:
                    break
                indices2 = greedy_cover(seed * 1000 + extra_seed)
                if indices2 is None:
                    continue
                reduced2 = reduce_cover(indices2)
                if len(reduced2) < best_size:
                    best_size = len(reduced2)
                    best_solution = [list(all_blocks[i]) for i in reduced2]

    if best_solution is None:
        # Fallback: simple greedy without reduction
        indices = greedy_cover(0)
        if indices:
            best_solution = [list(all_blocks[i]) for i in indices]
        else:
            return []
            
    return best_solution
