import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

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

    def get_bit_index(lsb):
        return lsb.bit_length() - 1

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
                # If no block covers any new subset, we are stuck (shouldn't happen with valid parameters)
                # but let's return what we have if we can't proceed
                return chosen_indices if chosen_indices else None
                
            # Randomly pick from top candidates to diversify
            chosen_idx = rng.choice(candidates)
            chosen_indices.append(chosen_idx)
            used[chosen_idx] = True
            current_uncovered &= ~block_masks[chosen_idx]
            
        return chosen_indices

    def reduce_cover(chosen_indices):
        if not chosen_indices:
            return []
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in chosen_indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                cover_count[get_bit_index(lsb)] += 1
                m ^= lsb
                
        kept_indices = []
        for idx in chosen_indices:
            m = block_masks[idx]
            is_redundant = True
            while m:
                lsb = m & -m
                if cover_count[get_bit_index(lsb)] <= 1:
                    is_redundant = False
                    break
                m ^= lsb
            if is_redundant:
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    cover_count[get_bit_index(lsb)] -= 1
                    m ^= lsb
            else:
                kept_indices.append(idx)
        return kept_indices

    def local_search_swap(chosen_indices, random_seed, max_iterations=50):
        """Try to improve by swapping blocks: remove one, add another if it helps."""
        if not chosen_indices:
            return chosen_indices
        
        rng = random.Random(random_seed)
        chosen_set = set(chosen_indices)
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in chosen_indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                cover_count[get_bit_index(lsb)] += 1
                m ^= lsb
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iterations += 1
            
            for i in range(len(chosen_indices)):
                if time.time() > deadline:
                    break
                remove_idx = chosen_indices[i]
                remove_mask = block_masks[remove_idx]
                
                m = remove_mask
                can_remove = True
                while m:
                    lsb = m & -m
                    if cover_count[get_bit_index(lsb)] <= 1:
                        can_remove = False
                        break
                    m ^= lsb
                
                if not can_remove:
                    continue
                
                m = remove_mask
                while m:
                    lsb = m & -m
                    cover_count[get_bit_index(lsb)] -= 1
                    m ^= lsb
                
                uncovered_mask = 0
                m = remove_mask
                while m:
                    lsb = m & -m
                    if cover_count[get_bit_index(lsb)] == 0:
                        uncovered_mask |= lsb
                    m ^= lsb
                
                if uncovered_mask == 0:
                    chosen_indices.pop(i)
                    chosen_set.discard(remove_idx)
                    improved = True
                    break
                
                best_gain = -1
                best_add_idx = -1
                for j in range(num_blocks):
                    if j in chosen_set:
                        continue
                    intersection = block_masks[j] & uncovered_mask
                    gain = count_bits(intersection)
                    if gain > best_gain:
                        best_gain = gain
                        best_add_idx = j
                
                if best_add_idx != -1 and best_gain > 0:
                    chosen_indices.append(best_add_idx)
                    chosen_set.add(best_add_idx)
                    m = block_masks[best_add_idx]
                    while m:
                        lsb = m & -m
                        cover_count[get_bit_index(lsb)] += 1
                        m ^= lsb
                    improved = True
                else:
                    m = remove_mask
                    while m:
                        lsb = m & -m
                        cover_count[get_bit_index(lsb)] += 1
                        m ^= lsb
                    chosen_indices.insert(i, remove_idx)
                    chosen_set.add(remove_idx)
        
        return chosen_indices

    def local_search_2for1(chosen_indices, random_seed, max_iterations=20):
        """Try to remove two blocks and add one that covers all newly uncovered subsets."""
        if len(chosen_indices) < 3:
            return chosen_indices
        
        rng = random.Random(random_seed)
        chosen_set = set(chosen_indices)
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in chosen_indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                cover_count[get_bit_index(lsb)] += 1
                m ^= lsb
        
        n = len(chosen_indices)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iterations += 1
            n = len(chosen_indices)
            
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    remove1 = chosen_indices[i]
                    remove2 = chosen_indices[j]
                    mask1 = block_masks[remove1]
                    mask2 = block_masks[remove2]
                    
                    # Check if both can be removed
                    m = mask1 | mask2
                    can_remove = True
                    while m:
                        lsb = m & -m
                        if cover_count[get_bit_index(lsb)] < 2:
                            can_remove = False
                            break
                        m ^= lsb
                    
                    if not can_remove:
                        continue
                    
                    # Temporarily remove both
                    m = mask1
                    while m:
                        lsb = m & -m
                        cover_count[get_bit_index(lsb)] -= 1
                        m ^= lsb
                    m = mask2
                    while m:
                        lsb = m & -m
                        cover_count[get_bit_index(lsb)] -= 1
                        m ^= lsb
                    
                    # Find uncovered subsets
                    uncovered_mask = 0
                    m = mask1 | mask2
                    while m:
                        lsb = m & -m
                        if cover_count[get_bit_index(lsb)] == 0:
                            uncovered_mask |= lsb
                        m ^= lsb
                    
                    if uncovered_mask == 0:
                        # Both removable!
                        chosen_indices.pop(j)
                        chosen_indices.pop(i)
                        chosen_set.discard(remove1)
                        chosen_set.discard(remove2)
                        improved = True
                        break
                    
                    # Find a single block that covers all uncovered subsets
                    best_add_idx = -1
                    for add_idx in range(num_blocks):
                        if add_idx in chosen_set:
                            continue
                        if (block_masks[add_idx] & uncovered_mask) == uncovered_mask:
                            best_add_idx = add_idx
                            break
                    
                    if best_add_idx != -1:
                        chosen_indices.append(best_add_idx)
                        chosen_set.add(best_add_idx)
                        m = block_masks[best_add_idx]
                        while m:
                            lsb = m & -m
                            cover_count[get_bit_index(lsb)] += 1
                            m ^= lsb
                        # Remove the two blocks
                        chosen_indices.pop(j)
                        chosen_indices.pop(i)
                        chosen_set.discard(remove1)
                        chosen_set.discard(remove2)
                        improved = True
                        break
                    else:
                        # Restore
                        m = mask1
                        while m:
                            lsb = m & -m
                            cover_count[get_bit_index(lsb)] += 1
                            m ^= lsb
                        m = mask2
                        while m:
                            lsb = m & -m
                            cover_count[get_bit_index(lsb)] += 1
                            m ^= lsb
        
        return chosen_indices

    # Run multiple restarts
    seed = 0
    while time.time() < deadline:
        indices = greedy_cover(seed)
        if indices is None:
            break
            
        reduced_indices = reduce_cover(indices)
        
        # Apply local search swap
        swapped_indices = local_search_swap(reduced_indices, seed, max_iterations=50)
        
        # Apply 2-for-1 local search
        final_indices = local_search_2for1(swapped_indices, seed, max_iterations=20)
        
        if len(final_indices) < best_size:
            best_size = len(final_indices)
            best_solution = [list(all_blocks[i]) for i in final_indices]
        elif len(swapped_indices) < best_size:
            best_size = len(swapped_indices)
            best_solution = [list(all_blocks[i]) for i in swapped_indices]
        elif len(reduced_indices) < best_size:
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
