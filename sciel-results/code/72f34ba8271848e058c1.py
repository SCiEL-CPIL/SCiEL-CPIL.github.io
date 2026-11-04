import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    # Use a deadline slightly below the 120s limit to be safe
    deadline = 110.0
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # 2. Generate all k-subsets and precompute their coverage masks
    k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(k_subsets)
    
    masks = [0] * num_k
    for i, k_sub in enumerate(k_subsets):
        mask = 0
        # Iterate through all t-subsets of this k-subset
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # 3. Greedy Set Cover
    uncovered = (1 << num_t) - 1
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_block_idx = -1
        max_covered = 0
        current_uncovered_count = uncovered.bit_count()
        
        for i in range(num_k):
            # Calculate how many currently uncovered t-subsets are covered by block i
            covered = (uncovered & masks[i]).bit_count()
            if covered > max_covered:
                max_covered = covered
                best_block_idx = i
                if max_covered == current_uncovered_count:
                    # Found a block that covers all remaining uncovered subsets
                    break
                    
        if best_block_idx == -1:
            # No block covers any new subset (should not happen if uncovered > 0)
            break
            
        blocks.append(list(k_subsets[best_block_idx]))
        uncovered &= ~masks[best_block_idx]
        
    return blocks
