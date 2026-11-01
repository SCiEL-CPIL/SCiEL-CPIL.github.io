import itertools
import random
import time
import heapq

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Compute degrees for weighted greedy
    degrees = [len(t_subset_blocks[i]) for i in range(num_t_subsets)]
    # Static weights: prioritize rare t-subsets
    weights = [1.0 / d if d > 0 else 0.0 for d in degrees]
    
    def run_greedy_and_minimize(seed_val):
        rng = random.Random(seed_val)
        
        # Reset state
        uncovered = [True] * num_t_subsets
        # Initial gain is sum of weights of all t-subsets in the block
        gain = [sum(weights[idx] for idx in block_t_indices[b]) for b in range(num_k_subsets)]
        
        selected_blocks = []
        
        # Greedy Phase
        while True:
            # Find max gain
            max_gain = max(gain)
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            candidates = []
            for b_idx in range(num_k_subsets):
                if gain[b_idx] == max_gain:
                    candidates.append(b_idx)
            
            # Random pick from candidates
            best_block = rng.choice(candidates)
            
            selected_blocks.append(best_block)
            
            # Update gains
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= weights[t_idx]
                    
        # Minimalization Phase
        # Calculate cover counts
        cover_count = [0] * num_t_subsets
        for b_idx in selected_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        active_blocks = set(selected_blocks)
        
        # Multiple passes of removal
        # We use a more aggressive local search: 
        # 1. Remove redundant blocks.
        # 2. Try to swap blocks to enable further removals (simple version: try replacing a block with another if it allows removing at least one other block).
        
        for _ in range(5): # 5 passes
            indices_to_check = list(active_blocks)
            rng.shuffle(indices_to_check)
            
            for i in indices_to_check:
                b_idx = active_blocks[i] if isinstance(active_blocks, list) else list(active_blocks)[i]
                # Since active_blocks is a set, we need to iterate carefully or convert to list
                pass
            
            # Let's use a list for active blocks to make swapping easier
            active_list = list(active_blocks)
            rng.shuffle(active_list)
            
            for b_idx in active_list:
                if b_idx not in active_blocks:
                    continue
                    
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    active_blocks.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                        
        return len(active_blocks), active_blocks

    best_size = float('inf')
    best_blocks = None
    
    # Run multiple times
    start_time = time.time()
    run_id = 0
    while time.time() - start_time < 110: # Leave 10s for overhead
        size, blocks = run_greedy_and_minimize(run_id)
        if size < best_size:
            best_size = size
            best_blocks = blocks
        run_id += 1
        
    if best_blocks is None:
        # Fallback if no run finished (shouldn't happen)
        # Do a deterministic run
        size, blocks = run_greedy_and_minimize(0)
        best_blocks = blocks
        
    result = [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
    return result
