import itertools
import random
import time

def construct():
    points = range(12)
    all_blocks = list(itertools.combinations(points, 6))
    num_blocks = len(all_blocks)
    
    # Map 5-subsets to indices 0..791
    all_5_subsets = list(itertools.combinations(points, 5))
    mask_to_idx = {}
    for i, sub in enumerate(all_5_subsets):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        mask_to_idx[mask] = i
        
    # Precompute the indices of the 5-subsets for each block
    block_5_indices = []
    for b in all_blocks:
        idxs = []
        for i in range(6):
            subset = b[:i] + b[i+1:]
            mask = 0
            for p in subset:
                mask |= (1 << p)
            idxs.append(mask_to_idx[mask])
        block_5_indices.append(idxs)
        
    best_indices = []
    best_count = 0
    target_count = 132
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Use a fixed seed for reproducibility? No, random is fine, 
    # but maybe we should try to be deterministic if possible? 
    # The task doesn't require determinism, just the result.
    
    while time.time() < deadline:
        # Create a shuffled list of block indices
        # list(range(num_blocks)) creates a new list each time
        # We can optimize this by shuffling a pre-existing list? 
        # But we need a fresh shuffle each time.
        indices = list(range(num_blocks))
        random.shuffle(indices)
        
        covered = [0] * 792
        current_indices = []
        
        for idx in indices:
            idxs = block_5_indices[idx]
            # Check conflict
            conflict = False
            for s_idx in idxs:
                if covered[s_idx]:
                    conflict = True
                    break
            
            if not conflict:
                current_indices.append(idx)
                for s_idx in idxs:
                    covered[s_idx] = 1
                    
        if len(current_indices) > best_count:
            best_count = len(current_indices)
            best_indices = current_indices
            if best_count == target_count:
                break
                
    # Convert best indices to list of blocks
    return [list(all_blocks[i]) for i in best_indices]
