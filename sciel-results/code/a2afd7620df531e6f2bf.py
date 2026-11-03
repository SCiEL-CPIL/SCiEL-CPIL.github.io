import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # 140s hard limit to be safe
    deadline = 140.0
    
    # Precompute t-subsets index
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Precompute all k-subsets and their masks
    blocks_list = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks_list)
    
    block_masks = [0] * num_blocks
    for i, block in enumerate(blocks_list):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
            
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy instances
    # Estimate time per greedy: ~5s. We can run ~20 instances.
    # But precomputation is one-time.
    
    attempts = 0
    while time.time() < deadline - 20: # Reserve 20s for polishing and final checks
        attempts += 1
        
        uncovered = (1 << num_t_subs) - 1
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline - 20:
                break
                
            best_gain = 0
            candidates = []
            
            # Find best block
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_gain == 0:
                break
                
            # Random tie-breaking
            idx = random.choice(candidates)
            selected_indices.append(idx)
            uncovered &= ~block_masks[idx]
            
        # Polish: Remove redundant blocks
        current_blocks = [blocks_list[i] for i in selected_indices]
        
        # Helper to count coverage
        def polish(blocks):
            # Compute counts
            counts = [0] * num_t_subs
            for b in blocks:
                for t_sub in itertools.combinations(b, t):
                    idx = t_sub_idx[t_sub]
                    counts[idx] += 1
            
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    if time.time() > deadline - 5:
                        break
                    b = blocks[i]
                    redundant = True
                    for t_sub in itertools.combinations(b, t):
                        idx = t_sub_idx[t_sub]
                        if counts[idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        for t_sub in itertools.combinations(b, t):
                            idx = t_sub_idx[t_sub]
                            counts[idx] -= 1
                        blocks.pop(i)
                        changed = True
            
            return blocks

        polished_blocks = polish(current_blocks)
        
        if len(polished_blocks) < best_count:
            best_count = len(polished_blocks)
            best_blocks = polished_blocks
            
        if best_count == 0: # Impossible
            break

    if best_blocks is None:
        # Fallback: return empty or last attempt
        return []
        
    return [list(b) for b in best_blocks]
