import itertools
import numpy as np
import time
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Precompute block_triples: list of lists of triple indices for each block
    block_triples = []
    # Precompute triple_to_blocks: list of lists of block indices for each triple
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        b_triples = []
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            b_triples.append(t_idx)
            triple_to_blocks[t_idx].append(i)
        block_triples.append(b_triples)
    
    # Convert triple_to_blocks to numpy arrays for faster indexing
    triple_to_blocks_np = [np.array(b, dtype=np.int32) if b else np.array([], dtype=np.int32) for b in triple_to_blocks]
    
    def greedy_cover_with_prune(random_seed):
        rng = np.random.RandomState(random_seed)
        
        # Initial gain for each block is k_t (all its triples are uncovered)
        gains = np.full(n_blocks, k_t, dtype=np.int32)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_mask = np.zeros(n_blocks, dtype=bool)
        
        selected_indices = []
        remaining = n_triples
        
        while remaining > 0:
            # Check deadline
            if time.time() > deadline:
                break
                
            # Find max gain among unselected blocks
            gains[selected_mask] = -1
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Select block
            selected_mask[best_idx] = True
            selected_indices.append(best_idx)
            
            # Update gains for affected blocks
            newly_covered = []
            for t_idx in block_triples[best_idx]:
                if uncovered_mask[t_idx]:
                    uncovered_mask[t_idx] = False
                    newly_covered.append(t_idx)
                    remaining -= 1
            
            if newly_covered:
                for t_idx in newly_covered:
                    affected = triple_to_blocks_np[t_idx]
                    if len(affected) > 0:
                        gains[affected] -= 1
                        
        # Pruning Phase: remove redundant blocks
        # coverage_count tracks how many selected blocks cover each triple
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in selected_indices:
            for t_idx in block_triples[b_idx]:
                coverage_count[t_idx] += 1
        
        active_blocks = list(selected_indices)
        
        changed = True
        passes = 0
        max_passes = 15
        
        while changed and passes < max_passes:
            if time.time() > deadline:
                break
            changed = False
            passes += 1
            rng.shuffle(active_blocks)
            
            new_active_blocks = []
            for b_idx in active_blocks:
                # Check if any triple in this block has coverage_count == 1
                # If no triple has count == 1, this block is redundant
                block_t_idxs = block_triples[b_idx]
                is_redundant = True
                for t_idx in block_t_idxs:
                    if coverage_count[t_idx] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for t_idx in block_t_idxs:
                        coverage_count[t_idx] -= 1
                    changed = True
                else:
                    new_active_blocks.append(b_idx)
            
            active_blocks = new_active_blocks
        
        # Convert indices to blocks
        return [list(all_blocks[i]) for i in active_blocks]

    # Run multiple times with different seeds
    best_blocks = None
    best_len = float('inf')
    
    # Dynamic number of restarts based on time
    n_restarts = 0
    for i in range(100):  # Max 100 restarts
        if time.time() > deadline - 5:  # Leave 5s buffer
            break
        if i > 0 and time.time() - start_time > 120:  # Hard stop at 120s
            break
            
        blocks = greedy_cover_with_prune(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        n_restarts += 1
    
    # If we didn't get any valid result due to timeout, do a quick greedy without pruning
    if best_blocks is None or best_len == float('inf'):
        # Fallback: simple greedy without pruning
        rng = np.random.RandomState(0)
        gains = np.full(n_blocks, k_t, dtype=np.int32)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_mask = np.zeros(n_blocks, dtype=bool)
        selected_indices = []
        remaining = n_triples
        
        while remaining > 0:
            gains[selected_mask] = -1
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            selected_mask[best_idx] = True
            selected_indices.append(best_idx)
            
            for t_idx in block_triples[best_idx]:
                if uncovered_mask[t_idx]:
                    uncovered_mask[t_idx] = False
                    remaining -= 1
            for t_idx in block_triples[best_idx]:
                if not uncovered_mask[t_idx]:  # was just covered
                    affected = triple_to_blocks_np[t_idx]
                    if len(affected) > 0:
                        gains[affected] -= 1
        
        best_blocks = [list(all_blocks[i]) for i in selected_indices]
    
    return best_blocks
