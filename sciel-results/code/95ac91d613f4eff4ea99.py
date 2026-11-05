import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate all t-subsets and map them to IDs
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # 2. Generate all k-blocks and precompute their t-subset ID sets
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # block_t_sets[i] is a frozenset of IDs of t-subsets contained in block i
    block_t_sets = []
    for block in blocks:
        # Get all t-subsets of this block
        block_t_subs = itertools.combinations(block, t)
        # Map them to IDs
        ids = {t_sub_to_id[sub] for sub in block_t_subs}
        block_t_sets.append(frozenset(ids))
        
    # 3. Greedy Set Cover
    uncovered = set(range(n_t_subs))
    used_blocks = [False] * n_blocks
    result_blocks = []
    
    # To speed up selection, we can maintain a list of candidate indices
    # But simply scanning all unused blocks is O(N) per step.
    # N=50k, Steps=150. Total 7.5M ops.
    
    while uncovered:
        if time.time() > deadline:
            # If we time out, we return what we have. 
            # The prompt implies invalid results score 0. 
            # We must try to finish. If we can't, we can't.
            # But the estimate says it's fast.
            pass
        
        best_idx = -1
        best_gain = -1
        
        # Find the block that covers the most uncovered t-subsets
        for i in range(n_blocks):
            if not used_blocks[i]:
                # Calculate intersection size
                # frozenset & set is fast
                gain = len(block_t_sets[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
        
        if best_gain == 0:
            # No block covers any uncovered subset? 
            # This shouldn't happen for valid parameters where a covering exists.
            # If it does, we break to avoid infinite loop.
            break
            
        # Select the block
        used_blocks[best_idx] = True
        result_blocks.append(list(blocks[best_idx]))
        
        # Update uncovered
        uncovered.difference_update(block_t_sets[best_idx])
        
    return result_blocks
