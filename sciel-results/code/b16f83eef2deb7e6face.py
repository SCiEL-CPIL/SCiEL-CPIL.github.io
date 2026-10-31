import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each block, store the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # For each t-subset, store the list of blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Count how many blocks cover each t-subset
    subset_count = [len(blocks) for blocks in t_subset_blocks]
    # Weights for greedy: prioritize rare subsets
    weights = [1.0 / c if c > 0 else 0.0 for c in subset_count]

    def run_greedy_and_local_search():
        uncovered = [True] * num_t_subsets
        gain = [0.0] * num_k_subsets
        for b_idx in range(num_k_subsets):
            g = 0.0
            for t_idx in block_t_indices[b_idx]:
                g += weights[t_idx]
            gain[b_idx] = g
            
        selected_blocks = []
        
        while True:
            max_gain = -1.0
            best_block = -1
            # Add small random noise to break ties and allow exploration
            for b_idx in range(num_k_subsets):
                current_gain = gain[b_idx] + random.uniform(0, 0.01)
                if current_gain > max_gain:
                    max_gain = current_gain
                    best_block = b_idx
                    
            if max_gain <= 1e-9:
                break
                
            selected_blocks.append(best_block)
            
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    w = weights[t_idx]
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= w
                        
        # Local Search: Iterative Deletion
        cover_count = [0] * num_t_subsets
        for b_idx in selected_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        active_blocks = set(selected_blocks)
        
        changed = True
        while changed:
            changed = False
            items = list(active_blocks)
            random.shuffle(items)
            for b_idx in items:
                is_redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    active_blocks.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
                    
        return [list(all_k_subsets[b_idx]) for b_idx in active_blocks]

    best_result = None
    best_size = float('inf')
    
    # Perform multiple restarts
    num_restarts = 10
    for _ in range(num_restarts):
        res = run_greedy_and_local_search()
        if len(res) < best_size:
            best_size = len(res)
            best_result = res
            
    return best_result
