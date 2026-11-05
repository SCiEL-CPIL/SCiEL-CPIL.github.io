import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # Precompute all t-subsets and index them
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Full universe mask
    full_mask = (1 << num_tsubs) - 1

    # Precompute all blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask

    # Convert block_masks to numpy array for vectorized operations
    block_masks_np = np.array(block_masks, dtype=np.uint64)
    
    def greedy_cover_vectorized():
        """
        Vectorized greedy set cover using NumPy.
        Uses batch processing to reduce Python loop overhead.
        """
        uncovered = full_mask
        selected_indices = []
        
        # Work with numpy arrays for vectorized gain computation
        # We'll process in batches to balance memory and speed
        
        while uncovered:
            if check_time(10.0):
                break
            
            # Compute gains for all blocks at once using numpy
            # Since we can't directly do bitwise AND with a Python int in numpy easily,
            # we need to use Python's int for the mask operations
            
            # For better performance, let's use a different approach:
            # Precompute which t-subsets each block covers as a set for faster lookup
            # Actually, let's stick with bitmask but optimize the inner loop
            
            best_gain = -1
            best_block = -1
            
            # Use a more efficient approach: only consider blocks that might have high gain
            # By checking a subset first, then refining
            batch_size = min(num_blocks, 5000)
            
            for start in range(0, num_blocks, batch_size):
                if check_time(5.0):
                    break
                end = min(start + batch_size, num_blocks)
                
                for i in range(start, end):
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                        
            if best_block == -1:
                break
                
            selected_indices.append(best_block)
            uncovered &= ~block_masks[best_block]
            
        return selected_indices

    def local_search_remove(indices):
        """
        Remove redundant blocks using union-based redundancy check.
        Optimized with incremental union tracking.
        """
        indices = list(indices)
        n = len(indices)
        if n < 2:
            return indices
            
        # Compute prefix and suffix unions for O(1) "union of all others" queries
        prefix_union = [0] * (n + 1)
        suffix_union = [0] * (n + 1)
        
        for i in range(n):
            prefix_union[i + 1] = prefix_union[i] | block_masks[indices[i]]
        for i in range(n - 1, -1, -1):
            suffix_union[i] = suffix_union[i + 1] | block_masks[indices[i]]
            
        changed = True
        while changed and not check_time(5.0):
            changed = False
            new_indices = []
            n = len(indices)
            
            if n < 2:
                break
                
            # Recompute prefix/suffix unions
            prefix_union = [0] * (n + 1)
            suffix_union = [0] * (n + 1)
            for i in range(n):
                prefix_union[i + 1] = prefix_union[i] | block_masks[indices[i]]
            for i in range(n - 1, -1, -1):
                suffix_union[i] = suffix_union[i + 1] | block_masks[indices[i]]
                
            for i in range(n):
                if check_time(2.0):
                    new_indices.extend(indices[i:])
                    break
                    
                idx = indices[i]
                # Union of all blocks except indices[i]
                others_union = prefix_union[i] | suffix_union[i + 1]
                
                if (block_masks[idx] & ~others_union) == 0:
                    changed = True
                else:
                    new_indices.append(idx)
                    
            indices = new_indices
            
        return indices

    def randomized_local_search(initial_indices, seed=42):
        """
        Randomized local search: remove a block, then repair by adding minimal blocks.
        """
        rng = random.Random(seed)
        indices = list(initial_indices)
        if not indices:
            return indices
            
        current_len = len(indices)
        best_indices = list(indices)
        best_len = current_len
        
        T = 15.0
        alpha = 0.985
        iters = 0
        max_iters = 80000
        
        # Precompute block membership for faster repair
        while T > 0.05 and iters < max_iters and not check_time(10.0):
            iters += 1
            
            if len(indices) < 2:
                break
                
            remove_idx = rng.randint(0, len(indices) - 1)
            removed_block = indices[remove_idx]
            
            new_indices = [x for x in indices if x != removed_block]
            
            # Compute what's missing
            missing = full_mask
            for idx in new_indices:
                missing &= ~block_masks[idx]
                
            if missing == 0:
                indices = new_indices
                current_len = len(indices)
                if current_len < best_len:
                    best_len = current_len
                    best_indices = list(indices)
                continue
            
            # Repair: greedily add blocks to cover missing
            added_blocks = []
            temp_missing = missing
            max_repair = 5
            
            for _ in range(max_repair):
                if temp_missing == 0:
                    break
                
                best_gain = -1
                best_add = -1
                
                # Sample candidates for speed
                sample_size = min(2000, num_blocks)
                candidates = rng.sample(range(num_blocks), sample_size)
                
                for i in candidates:
                    gain = (temp_missing & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_add = i
                        
                if best_add == -1 or best_gain == 0:
                    break
                    
                added_blocks.append(best_add)
                temp_missing &= ~block_masks[best_add]
                
            if temp_missing == 0:
                new_size = len(new_indices) + len(added_blocks)
                if new_size < current_len:
                    indices = new_indices + added_blocks
                    current_len = new_size
                    if current_len < best_len:
                        best_len = current_len
                        best_indices = list(indices)
                elif new_size == current_len and rng.random() < 0.15:
                    indices = new_indices + added_blocks
            
            T *= alpha
            
        return best_indices

    # --- Main Execution ---
    
    # 1. Greedy Initialization
    initial = greedy_cover_vectorized()
    
    if not initial:
        return []
        
    # 2. Standard 1-Opt Reduction
    reduced = local_search_remove(initial)
    best_blocks = list(reduced)
    best_len = len(best_blocks)

    # 3. Randomized Repair Search
    for seed in range(8):
        if check_time(15.0):
            break
        result = randomized_local_search(best_blocks, seed=seed * 137 + 1)
        if result and len(result) < best_len:
            best_len = len(result)
            best_blocks = result
            # After improvement, do another reduction pass
            best_blocks = local_search_remove(best_blocks)
            best_len = len(best_blocks)

    # 4. Final 1-Opt pass
    best_blocks = local_search_remove(best_blocks)
    
    return [list(all_blocks[i]) for i in best_blocks]
