import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    points = range(v)
    triples = list(itertools.combinations(points, t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Pre-compute bit for each triple
    triple_bits = [1 << i for i in range(n_triples)]
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Compute bitmask for each block
    block_masks = [0] * n_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = triple_to_idx[sub]
            mask |= triple_bits[idx]
        block_masks[b_idx] = mask
        
    full_mask = (1 << n_triples) - 1
    
    best_solution = None
    best_len = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline:
        uncovered = full_mask
        selected_blocks = []
        
        # Greedy Set Cover
        while uncovered:
            best_gain = -1
            best_indices = []
            
            # Find block with max gain
            for i in range(n_blocks):
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain and gain > 0:
                    best_indices.append(i)
            
            if not best_indices:
                break
                
            chosen_idx = random.choice(best_indices)
            selected_blocks.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        # Redundancy Removal
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for i in selected_blocks:
                union_others = 0
                for j in selected_blocks:
                    if j != i:
                        union_others |= block_masks[j]
                
                if (block_masks[i] & ~union_others) == 0:
                    changed = True
                else:
                    new_selected.append(i)
            
            selected_blocks = new_selected
            
        if len(selected_blocks) < best_len:
            best_len = len(selected_blocks)
            best_solution = selected_blocks[:]
            
    if best_solution is None:
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
