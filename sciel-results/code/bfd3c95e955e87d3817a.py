import itertools

def construct(v, k, t):
    # Generate all t-subsets and map them to indices for fast lookup
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Set of uncovered t-subset indices
    uncovered = set(range(len(all_t_subs)))
    
    blocks = []
    
    while uncovered:
        # Select the uncovered t-subset with the smallest index to ensure determinism
        s_idx = min(uncovered)
        s = all_t_subs[s_idx]
        s_set = set(s)
        
        # Points not in s
        remaining = [x for x in range(v) if x not in s_set]
        
        best_block = None
        best_gain = -1
        best_covered = []
        
        # We need to add k - t points to s to form a block of size k
        num_extra = k - t
        
        # Iterate over all combinations of extra points to complete the block
        for ext in itertools.combinations(remaining, num_extra):
            # Form the candidate block
            block_tuple = tuple(sorted(s + ext))
            
            # Calculate gain: number of uncovered t-subsets contained in this block
            gain = 0
            covered_in_block = []
            
            # Check all t-subsets of this candidate block
            for t_sub in itertools.combinations(block_tuple, t):
                idx = t_sub_index[t_sub]
                if idx in uncovered:
                    gain += 1
                    covered_in_block.append(idx)
            
            # Update best block if this one covers more uncovered subsets
            if gain > best_gain:
                best_gain = gain
                best_block = block_tuple
                best_covered = covered_in_block
            
            # Early exit if we found a block covering all its t-subsets (max possible)
            # Max possible is C(k, t). If we hit that, we can't do better.
            if gain == itertools.comb(k, t):
                break
        
        # Add the best block to the solution
        blocks.append(list(best_block))
        
        # Remove the covered t-subsets from the uncovered set
        uncovered.difference_update(best_covered)
        
    return blocks
