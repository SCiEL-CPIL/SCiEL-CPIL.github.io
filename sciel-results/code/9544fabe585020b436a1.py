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

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts with random tie-breaking
    num_restarts = 10
    for restart in range(num_restarts):
        uncovered = [True] * num_t_subsets
        gain = [0] * num_k_subsets
        
        for b_idx in range(num_k_subsets):
            gain[b_idx] = len(block_t_indices[b_idx])
            
        # Heap for max gain. Store (-gain, random_priority, block_idx)
        heap = [(-gain[b_idx], random.random(), b_idx) for b_idx in range(num_k_subsets)]
        heapq.heapify(heap)
        
        selected_blocks = []
        
        while True:
            if not heap:
                break
            
            # Pop until we find a valid entry (gain matches current and > 0)
            valid_block = None
            while heap:
                neg_g, _, b_idx = heap[0]
                current_gain = gain[b_idx]
                if -neg_g == current_gain and current_gain > 0:
                    heapq.heappop(heap)
                    valid_block = b_idx
                    break
                else:
                    heapq.heappop(heap)
            
            if valid_block is None:
                break
                
            selected_blocks.append(valid_block)
            
            # Update gains
            for t_idx in block_t_indices[valid_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx2 in t_subset_blocks[t_idx]:
                        gain[b_idx2] -= 1
                        
        # Minimalization: Repeated deletion
        cover_count = [0] * num_t_subsets
        for b_idx in selected_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        active_blocks = set(selected_blocks)
        changed = True
        while changed:
            changed = False
            for b_idx in list(active_blocks):
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        if len(active_blocks) < best_size:
            best_size = len(active_blocks)
            best_solution = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
            
        if best_size == 131:
            break

    return best_solution
