import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0 # Leave 5s margin for 150s limit
    
    # 1. Precomputation
    # Generate all t-subsets and map to index
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmask for each block
    # block_masks[i] is an int where bit j is 1 if all_blocks[i] covers all_t_subs[j]
    block_masks = [0] * num_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    def time_left():
        return deadline - time.time()
        
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask &= mask - 1
            
    best_solution = []
    best_score = float('inf')
    
    # Run multiple random restarts
    max_restarts = 100 # Limit by time
    
    for restart in range(max_restarts):
        if time_left() < 10.0:
            break
            
        # Random weights for tie-breaking
        random_weights = [random.random() for _ in range(num_blocks)]
        
        uncovered_mask = (1 << num_t_subs) - 1
        selected_indices = []
        
        # Greedy Construction
        while uncovered_mask:
            if time_left() < 5.0:
                break
                
            best_gain = -1
            best_idx = -1
            best_weight = -1.0
            
            # Find block with max gain (tie-broken by random weight)
            for i in range(num_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain or (gain == best_gain and random_weights[i] > best_weight):
                    best_gain = gain
                    best_idx = i
                    best_weight = random_weights[i]
                    
            if best_gain <= 0:
                break
                
            selected_indices.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
        # Polish: Remove redundant blocks
        if time_left() > 5.0 and selected_indices:
            current_indices = selected_indices[:]
            
            # Calculate coverage counts
            counts = [0] * num_t_subs
            for idx in current_indices:
                for bit in iter_bits(block_masks[idx]):
                    counts[bit] += 1
                        
            # Try to remove blocks
            changed = True
            while changed and time_left() > 5.0:
                changed = False
                for i in range(len(current_indices) - 1, -1, -1):
                    if time_left() < 5.0:
                        break
                    idx = current_indices[i]
                    is_redundant = True
                    for bit in iter_bits(block_masks[idx]):
                        if counts[bit] == 1:
                            is_redundant = False
                            break
                            
                    if is_redundant:
                        for bit in iter_bits(block_masks[idx]):
                            counts[bit] -= 1
                        current_indices.pop(i)
                        changed = True
                        
            if len(current_indices) < best_score:
                best_score = len(current_indices)
                best_solution = [all_blocks[i] for i in current_indices]
                
    if not best_solution:
        # Fallback
        if time_left() > 1.0:
            uncovered_mask = (1 << num_t_subs) - 1
            selected_indices = []
            while uncovered_mask:
                best_gain = -1
                best_idx = -1
                for i in range(num_blocks):
                    gain = (uncovered_mask & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_gain <= 0: break
                selected_indices.append(best_idx)
                uncovered_mask &= ~block_masks[best_idx]
            best_solution = [all_blocks[i] for i in selected_indices]
            
    return [list(b) for b in best_solution]
