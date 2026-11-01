import itertools
import time
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a randomized greedy set cover 
    approach with bitset optimization and local search for redundancy removal.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # 1. Pre-compute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Initialize full mask (all bits set)
    full_mask = (1 << num_t_subsets) - 1
    
    # 2. Pre-compute masks for all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    block_masks = [0] * num_k_blocks
    block_points = all_k_blocks # Store the actual points for return
    
    # Map each k-block to a bitmask of the t-subsets it contains
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask

    def greedy_construct(seed):
        """
        Runs a randomized greedy set cover.
        Returns a list of block indices.
        """
        random.seed(seed)
        uncovered_mask = full_mask
        selected_indices = []
        
        while uncovered_mask:
            # Find the block with maximum gain (newly covered t-subsets)
            max_gain = -1
            best_candidates = []
            
            for i in range(num_k_blocks):
                # Bitwise AND to find intersection with uncovered
                # bit_count() counts the number of 1s
                gain = (uncovered_mask & block_masks[i]).bit_count()
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain:
                    best_candidates.append(i)
            
            if not best_candidates:
                break # Should not happen if uncovered_mask is not 0
                
            # Pick a random candidate from the best ones to encourage diversity
            chosen_idx = random.choice(best_candidates)
            selected_indices.append(chosen_idx)
            
            # Update uncovered mask
            uncovered_mask &= ~block_masks[chosen_idx]
            
            # Time check
            if time.time() > deadline:
                break
                
        return selected_indices

    def remove_redundant(block_indices):
        """
        Removes redundant blocks from the covering.
        A block is redundant if all its t-subsets are covered by other blocks.
        """
        if not block_indices:
            return []
            
        # Count how many times each t-subset is covered
        cover_counts = [0] * num_t_subsets
        
        # Initialize counts
        for idx in block_indices:
            mask = block_masks[idx]
            # Iterate through set bits
            # Efficient way to iterate set bits in a Python int:
            # while m:
            #   lsb = m & -m
            #   idx = lsb.bit_length() - 1
            #   m ^= lsb
            m = mask
            while m:
                lsb = m & -m
                t_idx = lsb.bit_length() - 1
                cover_counts[t_idx] += 1
                m ^= lsb
                
        current_indices = list(block_indices)
        
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to handle removals easily, or just scan and mark
            # We scan and if we find a redundant one, we remove it and restart scan
            # This is O(N^2 * cost_check) in worst case, but N is small (~150)
            for i in range(len(current_indices)):
                idx = current_indices[i]
                mask = block_masks[idx]
                
                # Check if any t-subset in this block has count == 1
                # If all have count > 1, it's redundant
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    t_idx = lsb.bit_length() - 1
                    if cover_counts[t_idx] == 1:
                        is_redundant = False
                        break
                    m ^= lsb
                    
                if is_redundant:
                    # Remove this block
                    # Decrement counts
                    m = mask
                    while m:
                        lsb = m & -m
                        t_idx = lsb.bit_length() - 1
                        cover_counts[t_idx] -= 1
                        m ^= lsb
                        
                    current_indices.pop(i)
                    changed = True
                    break # Restart the loop as counts changed
                    
        return current_indices

    best_result = None
    best_len = float('inf')
    
    # Run multiple randomized greedy searches
    # The number of runs can be adaptive based on time
    num_runs = 20
    
    for i in range(num_runs):
        if time.time() > deadline - 5: # Leave 5s for finalization
            break
            
        indices = greedy_construct(i)
        if not indices:
            continue
            
        # Local search: remove redundancy
        indices = remove_redundant(indices)
        
        if len(indices) < best_len:
            best_len = len(indices)
            best_result = indices
            
        # If we hit the theoretical lower bound or a very good number, we could stop earlier
        # but let's try a few more if time permits
        
    if best_result is None:
        # Fallback: return empty or a trivial cover if something went wrong
        # But greedy should always find a cover
        return []
        
    # Convert indices back to lists of points
    final_blocks = [list(block_points[i]) for i in best_result]
    
    return final_blocks
