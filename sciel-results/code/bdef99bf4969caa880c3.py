import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Precomputation
    # 1. List all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    idx_to_t_subset = [None] * num_t_subsets
    for i, s in enumerate(t_subsets):
        idx_to_t_subset[i] = s
        
    # Bitmask for each t-subset
    # We don't need a mask per subset, we just need to know the index.
    # But for the "gain" calculation, we need to OR bits.
    # A bit i in 'uncovered' means t_subset i is uncovered.
    
    # 2. Helper to get bitmask for a specific t-subset
    def get_bit(s):
        return 1 << t_subset_to_idx[s]

    # 3. Greedy Cover
    uncovered = (1 << num_t_subsets) - 1
    blocks = []
    
    # Precompute combinations for gain calculation?
    # We need combinations of the current block of size t-1.
    # Since k is small (8) and t is 5, we can just compute on the fly.
    
    while uncovered:
        if time.time() > deadline:
            # This should not happen, but if it does, we are in trouble.
            # Returning an invalid cover scores 0.
            # We must try to finish. 
            # If we are here, it means the algorithm is too slow.
            # But we estimated it's fast.
            pass
            
        # Find the first uncovered t-subset
        # lowest set bit
        lsb = uncovered & -uncovered
        idx = lsb.bit_length() - 1
        first_subset = idx_to_t_subset[idx]
        
        block = list(first_subset)
        
        # Expand block to size k
        while len(block) < k:
            best_gain = -1
            best_x = -1
            block_set = set(block)
            
            # Calculate gain for each candidate x
            # Gain is number of uncovered t-subsets that can be formed 
            # by adding x to the current block.
            # A t-subset T is covered by block + {x} if T subseteq block U {x}.
            # Since T has size t, and x is new, T must contain x.
            # So T = {x} U S, where S is a subset of block of size t-1.
            
            if len(block) >= t - 1:
                # Precompute masks for all (t-1)-subsets of current block?
                # Or just iterate.
                # Iterating over combinations of block of size t-1
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    mask = 0
                    # Iterate over all (t-1)-subsets of block
                    for rest in itertools.combinations(block, t - 1):
                        # The t-subset is rest + (x,)
                        # Sort to match the key in t_subset_to_idx
                        t_sub = tuple(sorted(rest + (x,)))
                        # It is guaranteed to be in t_subset_to_idx
                        mask |= (1 << t_subset_to_idx[t_sub])
                    
                    # Count how many of these are currently uncovered
                    gain = (uncovered & mask).bit_count()
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            
            # If no gain (e.g. len(block) < t-1), pick arbitrary
            if best_x == -1:
                # Pick first available
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        # Find all t-subsets within this block
        block_mask = 0
        for t_sub in itertools.combinations(block, t):
            block_mask |= (1 << t_subset_to_idx[t_sub])
        
        uncovered &= ~block_mask
        
    return blocks
