import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy + Local Search.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precomputation
    # 1. Map all t-subsets to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Generate all k-blocks and their coverage bitmasks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    block_masks = [0] * num_k_blocks
    # For local search, we also need to know which t-subsets are in a block for fast deletion checks
    # But bitmask is enough for greedy. For deletion, we can re-derive or store lists.
    # Storing lists of indices for each block is memory heavy? 
    # 75582 blocks * 126 subsets per block (C(8,5)=56) -> ~4M entries. 
    # Let's store the bitmask and also a list of indices for the deletion phase to be safe/fast.
    block_indices = []
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        indices = []
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_indices.append(indices)
        
    # Initial full mask
    full_mask = (1 << num_t_subsets) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            best_gain = 0
            best_block_idx = -1
            # To introduce randomness in ties, we can collect all bests or just pick one randomly from the bests
            # For speed, we just pick the first max, but we can shuffle block order?
            # Shuffling 75k blocks every step is expensive.
            # Instead, we can just pick the first max. The randomness comes from the seed affecting... 
            # Wait, greedy is deterministic unless we add noise.
            # Let's add a small random noise to the gain to break ties randomly.
            
            # Optimization: If we just iterate, it's deterministic.
            # Let's try to randomize the iteration order of blocks slightly or use noise.
            # Given 75k blocks, iterating is fast.
            
            # We will use a simple noise: gain + random.random() * 0.1
            # This ensures different runs might pick different blocks in near-ties.
            
            for i in range(num_k_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain == 0:
                    continue
                # Add noise
                noisy_gain = gain + random.random() * 0.5
                if noisy_gain > best_gain:
                    best_gain = noisy_gain
                    best_block_idx = i
            
            if best_block_idx == -1:
                # Should not happen if uncovered is not 0 and we have all blocks
                break
                
            blocks.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        return blocks

    def local_search(block_indices_list):
        # block_indices_list is a list of block indices (into all_k_blocks)
        # We maintain a count of how many blocks cover each t-subset
        # Since we only have the list of block IDs, we need to rebuild counts or use the precomputed indices
        
        counts = [0] * num_t_subsets
        
        # Initialize counts
        for b_idx in block_indices_list:
            for s_idx in block_indices[b_idx]:
                counts[s_idx] += 1
                
        # Check if valid
        # All counts must be >= 1
        
        # Iterative deletion
        changed = True
        while changed:
            changed = False
            # Shuffle order to avoid bias? Or just iterate.
            # Iterate in reverse to remove easily?
            for i in range(len(block_indices_list) - 1, -1, -1):
                b_idx = block_indices_list[i]
                # Check if any subset in this block has count == 1
                # If so, it's essential.
                essential = False
                for s_idx in block_indices[b_idx]:
                    if counts[s_idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove block
                    for s_idx in block_indices[b_idx]:
                        counts[s_idx] -= 1
                    block_indices_list.pop(i)
                    changed = True
        
        return block_indices_list

    # Main loop
    run_id = 0
    while time.time() < deadline:
        blocks = greedy_cover(run_id)
        blocks = local_search(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        run_id += 1
        
    # Convert block indices back to actual blocks (lists of points)
    if best_blocks is None:
        # Fallback: return empty or something valid? 
        # Should not happen.
        return []
        
    result = [list(all_k_blocks[i]) for i in best_blocks]
    return result

