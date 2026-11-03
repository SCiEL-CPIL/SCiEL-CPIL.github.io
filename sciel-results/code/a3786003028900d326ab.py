import itertools
import time
import numpy as np

# Precomputation for v=19, k=7, t=4
# These can be generalized if v, k, t change, but the task specifies these values.
# We will compute them inside construct to be safe, or cache them.
# Given the time limit, caching is better if possible, but construct is called once.
# Let's just compute them in construct.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    # Total number of t-subsets
    n_t_subsets = len(list(itertools.combinations(range(v), t)))
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Precompute for each point x and each (t-1)-subset s, the index of s U {x}
    # This is used to quickly find which t-subsets are covered by adding x to a block containing s.
    # gain[x][s_idx] = index of the t-subset
    # Actually, we want to know, for a partial block B, and a candidate x, which t-subsets are covered.
    # A t-subset T is covered by B U {x} if T \ {x} is a subset of B.
    # So for each candidate x, we look at all (t-1)-subsets of B. For each such subset s, 
    # the t-subset is s U {x}.
    
    # Let's precompute: for each x, and each (t-1)-subset s, the index of s U {x}.
    # (t-1)-subsets count: C(19, 3) = 969
    # x count: 19
    # Total entries: 19 * 969 ~ 18411. Very small.
    
    t_minus_1_subsets = list(itertools.combinations(range(v), t - 1))
    t_minus_1_to_idx = {sub: i for i, sub in enumerate(t_minus_1_subsets)}
    
    # map[x][s_idx] = index in t_subsets of (t_minus_1_subsets[s_idx] U {x})
    # If x is in s, this is invalid, but we only call it if x not in B.
    # We can store -1 if x is in s.
    map_x_s_to_t_idx = [[-1] * len(t_minus_1_subsets) for _ in range(v)]
    for x in range(v):
        for s_idx, s in enumerate(t_minus_1_subsets):
            if x not in s:
                t_sub = tuple(sorted(s + (x,)))
                map_x_s_to_t_idx[x][s_idx] = subset_to_idx[t_sub]

    # Bitset for uncovered subsets
    # Python int is arbitrary precision, so it can hold 3876 bits easily.
    # Bit i is 1 if t_subsets[i] is uncovered.
    uncovered = (1 << n_t_subsets) - 1
    
    blocks = []
    
    # Helper to get the index of the lowest set bit
    def lowest_set_bit(n):
        if n == 0:
            return -1
        return (n & -n).bit_length() - 1

    # Helper to update uncovered when adding a block
    # We need to clear the bits for all t-subsets in the block.
    # Precompute for each block (represented as frozenset or tuple) the mask?
    # No, blocks are dynamic.
    # We can compute the mask on the fly.
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Find the first uncovered t-subset
        first_idx = lowest_set_bit(uncovered)
        if first_idx == -1:
            break
            
        seed = t_subsets[first_idx]
        block = list(seed)
        
        # Greedily extend the block to size k
        while len(block) < k:
            best_gain = -1
            best_point = -1
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                # For each (t-1)-subset of the current block, check if it forms a covered t-subset with x
                # Wait, the gain is the number of UNCOVERED t-subsets that become covered.
                # A t-subset T is covered by block U {x} if T \ {x} is a subset of block.
                # So we iterate over all (t-1)-subsets s of block.
                # For each s, the t-subset is s U {x}.
                # We check if that t-subset is currently uncovered.
                
                # Optimization: Precompute (t-1)-subsets of the current block?
                # The block size is small (max 6 during extension).
                # C(6, 3) = 20. C(5, 3) = 10. Very fast.
                
                for s in itertools.combinations(block, t - 1):
                    s_idx = t_minus_1_to_idx[s]
                    t_idx = map_x_s_to_t_idx[x][s_idx]
                    if t_idx != -1:
                        # Check if bit t_idx is set in uncovered
                        if (uncovered >> t_idx) & 1:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # Should not happen if uncovered is not empty and v > k
                break
                
            block.append(best_point)
        
        # Sort and add block
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        # Create a mask for this block
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = subset_to_idx[t_sub]
            mask |= (1 << idx)
        
        uncovered &= ~mask
        
    return blocks
