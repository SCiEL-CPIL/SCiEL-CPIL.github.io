import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # Precompute for each block, the indices of t-subsets it covers
    block_t_subset_indices = []
    for block in k_subsets:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_subset_to_idx[t_sub])
        block_t_subset_indices.append(indices)
        
    coverage = [0] * num_t_subsets
    uncovered_count = num_t_subsets
    solution_blocks = []
    
    # Greedy Phase
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        best_block_idx = -1
        max_gain = 0
        
        # Find the block that covers the most uncovered t-subsets
        for i in range(num_k_subsets):
            gain = 0
            # Optimization: if max_gain is already very high, we might skip? No.
            for idx in block_t_subset_indices[i]:
                if coverage[idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block_idx = i
                
        if max_gain == 0:
            # No block covers any uncovered subset? This implies we are stuck, 
            # but for covering designs this shouldn't happen unless v,k,t are invalid.
            break
            
        # Add the block
        solution_blocks.append(best_block_idx)
        for idx in block_t_subset_indices[best_block_idx]:
            if coverage[idx] == 0:
                uncovered_count -= 1
            coverage[idx] += 1
            
    # Reduction Phase
    # Try to remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate in reverse to handle removals
        for i in range(len(solution_blocks) - 1, -1, -1):
            block_idx = solution_blocks[i]
            # Check if this block is critical
            # A block is redundant if all its t-subsets are covered by other blocks
            is_redundant = True
            for idx in block_t_subset_indices[block_idx]:
                if coverage[idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                for idx in block_t_subset_indices[block_idx]:
                    coverage[idx] -= 1
                solution_blocks.pop(i)
                changed = True
                
    # Convert indices back to actual blocks
    final_blocks = [list(k_subsets[i]) for i in solution_blocks]
    
    return final_blocks
