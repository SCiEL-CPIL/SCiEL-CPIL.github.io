import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s kill

    # Precompute combinations as integer masks for speed
    # v=19, k=8, t=5
    
    # 1. Generate all t-subsets (masks)
    t_subsets = list(itertools.combinations(range(v), t))
    t_masks = [0] * len(t_subsets)
    for i, ts in enumerate(t_subsets):
        mask = 0
        for x in ts:
            mask |= (1 << x)
        t_masks[i] = mask
        
    num_t = len(t_masks)
    
    # 2. Generate all k-blocks (masks)
    k_blocks = list(itertools.combinations(range(v), k))
    k_masks = [0] * len(k_blocks)
    for i, blk in enumerate(k_blocks):
        mask = 0
        for x in blk:
            mask |= (1 << x)
        k_masks[i] = mask
        
    num_k = len(k_masks)
    
    # 3. Precompute block_to_ts_indices
    # For each block, which t-subsets does it cover?
    # A block covers a t-subset if (block_mask & t_mask) == t_mask
    # This check is O(1) per pair, but we have num_k * num_t pairs.
    # 75k * 11k is too slow.
    
    # Alternative: For each block, generate its t-subsets directly.
    # Each block has C(8,5) = 56 t-subsets.
    # Total operations: 75k * 56 = 4.2M. This is fast.
    
    # We need a map from t-subset tuple (or mask) to index
    t_tuple_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    block_to_ts_indices = []
    # Also build ts_to_block_indices for local search
    ts_to_block_indices = [[] for _ in range(num_t)]
    
    for j, blk in enumerate(k_blocks):
        ts_list = []
        # Generate combinations of the block's elements
        for ts in itertools.combinations(blk, t):
            idx = t_tuple_to_idx[ts]
            ts_list.append(idx)
            ts_to_block_indices[idx].append(j)
        block_to_ts_indices.append(ts_list)
        
    # Convert block_to_ts_indices to numpy arrays for faster slicing?
    # Lists are probably fine for the small size (56 elements).
    
    def run_greedy(seed):
        rng = random.Random(seed)
        # gain[b] = number of uncovered t-subsets covered by block b
        gain = np.full(num_k, 0, dtype=np.int32)
        is_uncovered = np.full(num_t, True, dtype=bool)
        is_selected = np.full(num_k, False, dtype=bool)
        sol = []
        
        # Initialize gains
        # For each block, count how many of its t-subsets are uncovered (all initially)
        # Since all are uncovered initially, gain[b] = C(k, t) for all b
        # C(8,5) = 56
        gain[:] = 56
        
        while True:
            if time.time() > deadline - 2.0:
                return None
            
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            
            # Find candidates with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            
            b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            
            sol.append(list(k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            bts = block_to_ts_indices[b_idx]
            
            # Find which of these t-subsets are still uncovered
            newly_covered = [ts_idx for ts_idx in bts if is_uncovered[ts_idx]]
            
            if not newly_covered:
                continue
                
            # Mark as covered
            for ts_idx in newly_covered:
                is_uncovered[ts_idx] = False
                
            # Update gains
            # For each newly covered t-subset, decrement gain of all blocks covering it
            # We can accumulate decrements in a dict or array
            # Since newly_covered is small (~56), and each t-subset is covered by ~300 blocks,
            # total updates ~ 56 * 300 = 16800.
            
            # Use a temporary array for decrements to avoid multiple passes?
            # Or just update directly. Direct update is O(1) per block.
            # Let's use a numpy array for decrements to vectorize if possible, 
            # but the indices are sparse.
            
            # Simple loop is likely fast enough for 16k ops
            # But we can do better by accumulating in a dict or using np.add.at on a sparse set?
            # Let's stick to a simple loop over newly_covered and their covering blocks.
            
            for ts_idx in newly_covered:
                for b in ts_to_block_indices[ts_idx]:
                    if not is_selected[b]:
                        gain[b] -= 1
                        
        return sol

    def remove_redundant(blocks):
        """
        Remove blocks that are not needed to maintain coverage.
        """
        if not blocks:
            return blocks
            
        # Map block tuple to index in the global k_blocks list
        # We assume blocks are from k_blocks.
        block_tuple_to_idx = {tuple(b): i for i, b in enumerate(k_blocks)}
        
        # Build coverage count
        cover_count = np.zeros(num_t, dtype=np.int32)
        block_indices = []
        for b in blocks:
            idx = block_tuple_to_idx.get(tuple(b), -1)
            if idx != -1:
                block_indices.append(idx)
                for ts_idx in block_to_ts_indices[idx]:
                    cover_count[ts_idx] += 1
            else:
                # Should not happen if we only use generated blocks
                pass
        
        n = len(blocks)
        if n == 0:
            return blocks
            
        # Iteratively remove redundant blocks
        # To be safe and fast, do a few passes
        changed = True
        max_passes = 10
        for _ in range(max_passes):
            if not changed:
                break
            changed = False
            for i in range(len(block_indices)):
                if time.time() > deadline - 5.0:
                    return [k_blocks[idx] for idx in block_indices]
                
                idx = block_indices[i]
                ts_list = block_to_ts_indices[idx]
                
                # Check if all t-subsets in this block are covered by >= 2 blocks
                # i.e., cover_count[ts] >= 2 for all ts in ts_list
                # If yes, we can remove this block.
                
                # Optimization: Check if any ts has count 1. If so, cannot remove.
                # We need to check all.
                redundant = True
                for ts_idx in ts_list:
                    if cover_count[ts_idx] < 2:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove block i
                    for ts_idx in ts_list:
                        cover_count[ts_idx] -= 1
                    block_indices.pop(i)
                    changed = True
                    break # Restart loop due to index shift
                    
        return [k_blocks[idx] for idx in block_indices]

    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        seed = num_restarts * 101
        blocks = run_greedy(seed)
        
        if blocks is None:
            continue
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # Local Search: Remove redundant
        # Only do this if we have time
        if time.time() < deadline - 5.0:
            improved_blocks = remove_redundant(blocks)
            if len(improved_blocks) < best_count:
                best_count = len(improved_blocks)
                best_blocks = improved_blocks
                
        # Optional: Another greedy restart with a different seed strategy?
        # The current loop does that.

    if best_blocks is None:
        # Fallback: Return a valid covering if we found one, or try one last greedy
        # If we didn't find any, return empty (invalid, but better than crash? No, must be valid)
        # The greedy should always find a covering.
        blocks = run_greedy(12345)
        if blocks:
            return blocks
        return []
        
    return best_blocks
