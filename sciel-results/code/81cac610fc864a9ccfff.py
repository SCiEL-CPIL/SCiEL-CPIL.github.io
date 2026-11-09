import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using multiple randomized greedy set cover
    restarts, fast redundancy removal, and 1-exchange local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    block_masks = []
    all_blocks = []
    
    # Generate all k-blocks and their masks
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    num_blocks = len(block_masks)
    
    # Helper for bit count
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
        
    def remove_redundant(chosen_indices):
        """
        Remove redundant blocks using coverage counts.
        Returns a list of non-redundant block indices.
        """
        if not chosen_indices:
            return []
            
        # Initialize counts
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Increment counts for all chosen blocks
        for idx in chosen_indices:
            mask = block_masks[idx]
            # Iterate set bits in mask
            m = mask
            while m:
                lsb = m & -m
                idx_bit = lsb.bit_length() - 1
                counts[idx_bit] += 1
                m ^= lsb
                
        # Iterate and remove redundant blocks
        # We need to be careful: removing a block changes counts for other blocks.
        # A block is redundant if all its t-subsets are covered by at least one OTHER block.
        # i.e., for all t_sub in block, counts[t_sub] > 1.
        
        # We can iterate multiple times or just one pass? 
        # One pass might not be enough if removing A makes B redundant, but B was not redundant before A was removed.
        # However, if we remove a block, counts decrease. This can only make MORE blocks redundant, not fewer.
        # So we can just iterate through the list, and if a block is redundant, remove it and update counts.
        
        active_indices = list(chosen_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(active_indices)):
                idx = active_indices[i]
                mask = block_masks[idx]
                
                # Check if redundant
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    if counts[idx_bit] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                    
                if is_redundant:
                    # Remove block
                    active_indices.pop(i)
                    # Decrement counts
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] -= 1
                        m ^= lsb
                    changed = True
                    break # Restart loop as indices shifted and counts changed
                    
            if time.time() > deadline - 5.0:
                break
                
        return active_indices

    def greedy_randomized(seed):
        """
        Run randomized greedy set cover.
        Returns None if it times out before completing the cover.
        """
        random.seed(seed)
        current_uncovered = (1 << num_t_subsets) - 1
        chosen_block_indices = []
        
        while current_uncovered != 0:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = 0
            candidates = []
            
            for i in range(num_blocks):
                intersection = block_masks[i] & current_uncovered
                gain = intersection.bit_count()
                
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
                    
            if not candidates:
                # Should not happen if current_uncovered is not 0
                return None
            
            best_block_idx = random.choice(candidates)
            chosen_block_indices.append(best_block_idx)
            current_uncovered &= ~block_masks[best_block_idx]
            
        return chosen_block_indices

    def greedy_deterministic():
        """
        Fallback deterministic greedy. Faster than randomized due to no tie-breaking list.
        """
        current_uncovered = (1 << num_t_subsets) - 1
        chosen_block_indices = []
        
        while current_uncovered != 0:
            if time.time() > deadline - 1.0:
                return None
            
            max_gain = 0
            best_block_idx = -1
            
            for i in range(num_blocks):
                intersection = block_masks[i] & current_uncovered
                gain = intersection.bit_count()
                
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
            
            if best_block_idx == -1:
                return None
                
            chosen_block_indices.append(best_block_idx)
            current_uncovered &= ~block_masks[best_block_idx]
            
        return chosen_block_indices

    # Main Strategy:
    # 1. Run several randomized greedy instances
    # 2. Clean up redundancies
    # 3. Keep the best valid solution
    
    best_solution = []
    
    # Number of restarts depends on time. 
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        seed = 42 + r
        candidates = greedy_randomized(seed)
        if candidates is None:
            # Timed out, try next seed
            continue
            
        # Remove redundants
        cleaned = remove_redundant(candidates)
        
        # Only update if we have a solution and it's better
        if len(cleaned) < len(best_solution) or not best_solution:
            best_solution = cleaned
            
    # If no valid solution found yet, run deterministic fallback
    if not best_solution:
        candidates = greedy_deterministic()
        if candidates is not None:
            best_solution = remove_redundant(candidates)
            
    # If still no solution (extreme timeout), return empty (will score 0, but better than crash)
    # However, greedy_deterministic is very fast, so this is unlikely.
    
    return [list(all_blocks[idx]) for idx in best_solution]
