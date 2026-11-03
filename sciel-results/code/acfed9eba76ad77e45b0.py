import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Precomputation
    # Generate all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, find the indices of the t-subsets it covers
    # subsets_in_block[b] = list of t-subset indices
    subsets_in_block = []
    for b in all_k_subsets:
        covered_t_indices = []
        for ts in itertools.combinations(b, t):
            covered_t_indices.append(t_subset_to_idx[ts])
        subsets_in_block.append(covered_t_indices)
        
    # For each t-subset, find the indices of the k-subsets that cover it
    # blocks_covering_subset[s] = list of k-subset indices
    blocks_covering_subset = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(subsets_in_block):
        for s_idx in t_indices:
            blocks_covering_subset[s_idx].append(b_idx)
            
    # 2. Greedy Set Cover
    # gain[b] = number of uncovered t-subsets in block b
    gain = np.full(num_k_subsets, len(subsets_in_block[0]), dtype=np.int32)
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    chosen_blocks = []
    
    while True:
        if time.time() > deadline:
            break
            
        # Find block with max gain
        b_star = np.argmax(gain)
        if gain[b_star] == 0:
            break
            
        chosen_blocks.append(b_star)
        
        # Update gains and cover counts
        for s_idx in subsets_in_block[b_star]:
            if cover_count[s_idx] == 0:
                cover_count[s_idx] += 1
                # Decrement gain for all blocks covering this t-subset
                for b_prime in blocks_covering_subset[s_idx]:
                    gain[b_prime] -= 1
            else:
                cover_count[s_idx] += 1
                
    # 3. Local Search: Remove redundant blocks
    # We iterate and try to remove blocks if they are redundant
    # A block is redundant if every t-subset in it is covered by at least one other block.
    # i.e., cover_count[s] > 1 for all s in block.
    
    # To speed up, we can just do a few passes.
    # Since the greedy order might matter, we can try removing from the end or randomly.
    # Let's do a simple pass: iterate through chosen_blocks, try to remove.
    # If removed, restart the pass (or continue, but restart is safer for finding more).
    
    current_blocks = chosen_blocks[:]
    
    # Optimization: Instead of restarting, we can just loop until no change.
    # But checking redundancy is O(56) per block.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_blocks = []
        for b_idx in current_blocks:
            # Check if b_idx can be removed
            can_remove = True
            for s_idx in subsets_in_block[b_idx]:
                if cover_count[s_idx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove b_idx
                for s_idx in subsets_in_block[b_idx]:
                    cover_count[s_idx] -= 1
                changed = True
            else:
                new_blocks.append(b_idx)
                
        current_blocks = new_blocks
        
    # Convert block indices back to lists of points
    result = [list(all_k_subsets[b_idx]) for b_idx in current_blocks]
    
    return result
