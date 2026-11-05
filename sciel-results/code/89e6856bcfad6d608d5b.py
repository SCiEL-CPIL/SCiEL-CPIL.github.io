import itertools
import time
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Simulated Annealing
    on the set of blocks to minimize the number of blocks.
    """
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute masks for each k-subset
    # mask[i] is a set of t-subset indices contained in k_subsets[i]
    # Using a list of sets for easier iteration in SA, or bitmasks for speed?
    # Bitmasks are faster for intersection, but checking "is uncovered" requires counting bits.
    # For SA, we need to know if a swap is valid.
    # Validity: For every t-subset in Block_A, if it was covered ONLY by Block_A, then Block_B must also cover it.
    # This is hard to check with just bitmasks without maintaining coverage counts.
    
    # Let's maintain coverage_counts: array of length n_t.
    
    # Greedy Initialization
    uncovered_mask = (1 << n_t) - 1
    blocks_indices = []
    
    # We need block_masks for greedy
    block_masks = [0] * n_k
    for i, k_sub in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask

    # Greedy Phase
    while uncovered_mask:
        best_idx = -1
        max_cov = 0
        # Randomized tie-breaking to diversify
        for i in range(n_k):
            cov = (uncovered_mask & block_masks[i]).bit_count()
            if cov > max_cov:
                max_cov = cov
                best_idx = i
            elif cov == max_cov and cov > 0:
                # Randomly pick one of the best to break symmetry
                if random.random() < 0.5:
                    best_idx = i
        if max_cov == 0:
            break
        blocks_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # Initialize coverage counts for the greedy solution
    coverage_counts = [0] * n_t
    for b_idx in blocks_indices:
        mask = block_masks[b_idx]
        # Iterate set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage_counts[idx] += 1
            m ^= lsb
            
    best_solution = list(blocks_indices)
    best_size = len(best_solution)
    
    # Simulated Annealing
    start_time = time.time()
    timeout = 110 # Leave 10s margin
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    current_blocks = set(best_solution)
    
    while time.time() - start_time < timeout and T > min_T:
        # Choose a move
        move_type = random.random()
        if len(current_blocks) > 0:
            # Try Remove (30%)
            if move_type < 0.3:
                b_out = random.choice(list(current_blocks))
                # Check if b_out is redundant
                # It is redundant if for all t_sub in b_out, coverage_counts[t_sub] > 1
                redundant = True
                mask = block_masks[b_out]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if coverage_counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    # Accept
                    current_blocks.remove(b_out)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_counts[idx] -= 1
                        m ^= lsb
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_solution = list(current_blocks)
            # Try Swap (70%)
            else:
                b_out = random.choice(list(current_blocks))
                b_in = random.randint(0, n_k - 1)
                if b_in in current_blocks:
                    continue
                
                # Check validity of swap
                # For every t_sub in b_out, if coverage_counts[t_sub] == 1, then b_in MUST contain t_sub.
                valid = True
                mask_out = block_masks[b_out]
                mask_in = block_masks[b_in]
                m = mask_out
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if coverage_counts[idx] == 1:
                        # b_in must cover this
                        if not (mask_in & (1 << idx)):
                            valid = False
                            break
                    m ^= lsb
                
                if valid:
                    # Accept swap
                    current_blocks.remove(b_out)
                    current_blocks.add(b_in)
                    
                    # Update counts
                    m = mask_out
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_counts[idx] -= 1
                        m ^= lsb
                        
                    m = mask_in
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_counts[idx] += 1
                        m ^= lsb
                        
                    # After swap, try to remove redundant blocks immediately?
                    # This helps to shrink the solution.
                    changed = True
                    while changed:
                        changed = False
                        # Randomly pick a block to check for redundancy
                        # To be fast, just check a few random ones or all?
                        # Checking all is O(N * C(k,t)). N~150. C(7,4)=35. 5250 ops. Fast enough.
                        for b_check in list(current_blocks):
                            redundant = True
                            mask_c = block_masks[b_check]
                            m = mask_c
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                if coverage_counts[idx] == 1:
                                    redundant = False
                                    break
                                m ^= lsb
                            
                            if redundant:
                                current_blocks.remove(b_check)
                                m = mask_c
                                while m:
                                    lsb = m & -m
                                    idx = lsb.bit_length() - 1
                                    coverage_counts[idx] -= 1
                                    m ^= lsb
                                changed = True
                                if len(current_blocks) < best_size:
                                    best_size = len(current_blocks)
                                    best_solution = list(current_blocks)
                                break # Restart loop as set changed
                else:
                    # Invalid swap, reject (or accept with prob? No, keep it simple)
                    pass
        T *= cooling

    return [list(k_subsets[i]) for i in best_solution]
