import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Generate t-subsets and map to index
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t = len(t_subs)
    
    # 2. Generate k-subsets and compute masks
    k_subs = list(itertools.combinations(range(v), k))
    num_k = len(k_subs)
    
    masks = [0] * num_k
    for i, block in enumerate(k_subs):
        mask = 0
        # For each t-subset within this block, set the bit
        # A block of size k contains C(k, t) t-subsets
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    # 3. Greedy Selection
    uncovered_mask = (1 << num_t) - 1
    unused_indices = list(range(num_k))
    blocks = []
    
    while uncovered_mask:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_idx = -1
        
        # Find the block that covers the most uncovered t-subsets
        for idx in unused_indices:
            gain = (masks[idx] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                if gain == num_t: # Can't do better than covering all remaining? 
                    # Actually, if gain is large, it's good. 
                    # We can't break early unless gain is max possible (which is popcount(uncovered_mask))
                    # But popcount(uncovered_mask) changes.
                    pass
        
        if best_idx == -1:
            break # No block covers any uncovered subset? Error.
            
        blocks.append(list(k_subs[best_idx]))
        uncovered_mask &= ~masks[best_idx]
        unused_indices.remove(best_idx)
        
        # Optional: Prune unused_indices
        # This might be expensive if done naively. 
        # Let's just keep the list and let the gain check handle it.
        # If gain is 0, it won't be selected.
        
    return blocks
