import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    # Allow 110 seconds for the whole process to be safe
    deadline = start_time + 110
    
    # 1. Precompute triples and their indices
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {tup: i for i, tup in enumerate(triples)}
    
    # 2. Precompute all blocks and their coverage masks
    all_blocks = []
    all_block_masks = []
    
    # For v=24, k=7, t=3, this loop takes ~2-3 seconds
    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << triple_to_idx[sub])
        all_blocks.append(block)
        all_block_masks.append(mask)
        
    n_blocks = len(all_blocks)
    
    # If precomputation took too long, we might not have time for search
    if time.time() > deadline - 10:
        # Fallback: simple greedy without multiple restarts?
        # Just return the first block? No, must be valid.
        # Do one quick greedy.
        pass

    best_blocks = None
    best_len = float('inf')
    
    restart_id = 0
    
    while time.time() < deadline - 5: # Leave 5s for final polish/return
        restart_id += 1
        rng = random.Random(restart_id)
        
        # Shuffle the order in which we consider blocks
        # This helps escape local optima caused by deterministic tie-breaking
        block_indices = list(range(n_blocks))
        rng.shuffle(block_indices)
        
        uncovered_mask = (1 << n_triples) - 1
        selected_indices = []
        
        # Greedy Selection
        while uncovered_mask:
            best_gain = -1
            best_idx = -1
            
            # Find the block that covers the most uncovered triples
            for idx in block_indices:
                gain = (all_block_masks[idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    # Early termination if we found a block covering all possible new triples
                    # Max gain is C(k, t)
                    if best_gain == 35: # Specific to k=7, t=3. Use general calc?
                        # C(7,3) = 35.
                        # To be general:
                        # max_gain = 1
                        # for i in range(k, t-1, -1): max_gain *= i
                        # for i in range(1, k-t+1): max_gain //= i
                        # But 35 is hardcoded for speed.
                        pass
                    
            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            uncovered_mask &= ~all_block_masks[best_idx]
            
        # Polish: Remove redundant blocks
        current_indices = list(selected_indices)
        changed = True
        while changed:
            changed = False
            total_union = 0
            for idx in current_indices:
                total_union |= all_block_masks[idx]
                
            new_indices = []
            for idx in current_indices:
                m = all_block_masks[idx]
                others_union = 0
                for j in current_indices:
                    if j != idx:
                        others_union |= all_block_masks[j]
                
                if (m & ~others_union) == 0:
                    changed = True
                else:
                    new_indices.append(idx)
                    
            current_indices = new_indices
            
        if len(current_indices) < best_len:
            best_len = len(current_indices)
            best_blocks = [all_blocks[i] for i in current_indices]
            
            if best_len <= 69:
                break
                
    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]
