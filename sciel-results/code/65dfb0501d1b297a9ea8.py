import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Generate all t-subsets and map to IDs
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_id = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    # 2. Generate all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # 3. Pre-compute mappings
    # block_to_subsets[i] = list of t-subset IDs covered by block i
    # subset_to_blocks[j] = list of block indices covering t-subset j
    
    block_to_subsets = [None] * n_k
    subset_to_blocks = [[] for _ in range(n_t)]
    
    for i, b in enumerate(all_k_subs):
        covered_ids = []
        for t_sub in itertools.combinations(b, t):
            j = t_sub_to_id[t_sub]
            covered_ids.append(j)
            subset_to_blocks[j].append(i)
        block_to_subsets[i] = covered_ids
        
    # 4. Greedy Loop
    # gains[i] = number of currently uncovered t-subsets covered by block i
    gains = np.array([len(ids) for ids in block_to_subsets], dtype=np.int32)
    is_covered = np.zeros(n_t, dtype=bool)
    remaining = n_t
    blocks = []
    
    while remaining > 0:
        if time.time() > deadline:
            # If we run out of time, we return what we have. 
            # The task says "invalid result ... scores 0". 
            # But we must return *something*. 
            # Given the speed, we should not hit this.
            # If we do, the result is likely invalid.
            break
        
        best_idx = np.argmax(gains)
        
        # If best gain is 0, we are stuck (should not happen if covering exists)
        if gains[best_idx] == 0:
            break
            
        blocks.append(list(all_k_subs[best_idx]))
        
        # Update coverage and gains
        for sid in block_to_subsets[best_idx]:
            if not is_covered[sid]:
                is_covered[sid] = True
                remaining -= 1
                # Decrement gain for all blocks that cover this subset
                # subset_to_blocks[sid] is a list of indices
                if subset_to_blocks[sid]:
                    gains[subset_to_blocks[sid]] -= 1
                    
    return blocks
