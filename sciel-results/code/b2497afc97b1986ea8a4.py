import itertools
import time
import random
import numpy as np
from collections import defaultdict

# Precomputation for v=19, k=8, t=5
# Note: The task says construct(v, k, t) will be called with these params.
# We should cache precomputation if possible, or do it inside construct if it's fast enough.
# 11628 subsets, 75582 blocks. Precomputation is < 1s.

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave margin for 120s kill
    
    # If parameters are not the standard ones, fallback to a simple greedy 
    # to ensure correctness, though the task specifies C(19,8,5).
    if v != 19 or k != 8 or t != 5:
        return _fallback_greedy(v, k, t)

    # 1. Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # 2. Precompute all k-blocks and the t-subsets they cover
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_covered_subsets[i] = list of indices of t-subsets covered by block i
    block_covered_subsets = []
    # blocks_containing_subset[j] = list of indices of k-blocks that contain t-subset j
    blocks_containing_subset = [[] for _ in range(n_t_subsets)]
    
    for i, block in enumerate(all_k_blocks):
        block_set = set(block)
        covered_idx = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_idx.append(idx)
            blocks_containing_subset[idx].append(i)
        block_covered_subsets.append(np.array(covered_idx, dtype=np.int32))
    
    # Convert to a list of arrays for fast access
    # We might need a matrix for the "restricted" approach? 
    # No, we just access block_covered_subsets[best_block_idx]
    
    def time_elapsed():
        return time.time() - start_time

    def greedy_cover_restricted(rng):
        """
        Greedy cover that picks a random uncovered t-subset, 
        then picks the best k-block containing it.
        """
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks_used = []
        
        # Map from block index to its actual points for return
        block_points_map = {i: list(block) for i, block in enumerate(all_k_blocks)}
        
        while np.any(uncovered):
            if time_elapsed() > time_limit:
                break
            
            # Pick a random uncovered t-subset
            # To be more strategic, we could pick the one with the most "constraints" 
            # or just random for diversity in restarts.
            # Let's try picking the one with the highest frequency in the remaining uncovered set?
            # That requires counting. 
            # Random is O(1) if we maintain a list of uncovered indices.
            
            # Efficiently pick a random uncovered index
            # np.nonzero(uncovered) is O(N). 
            # We can maintain a list `uncovered_indices` and swap-remove.
            # But for simplicity and speed of implementation, let's use np.nonzero 
            # or a list. 
            # Given N=11628, np.nonzero is fast.
            
            unc_idx = np.nonzero(uncovered)[0]
            if len(unc_idx) == 0:
                break
            
            # Randomly select one
            target_sub_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            
            # Get all blocks containing this subset
            candidate_block_indices = blocks_containing_subset[target_sub_idx]
            
            if not candidate_block_indices:
                # Should not happen for valid parameters, but safety
                break
                
            # Calculate gain for each candidate block
            # Gain = number of currently uncovered t-subsets covered by this block
            # We can vectorize this if we had a matrix, but we have lists.
            # Let's just loop over ~364 candidates.
            
            best_block_idx = -1
            best_gain = -1
            
            # Optimization: If we have many candidates, we can batch process?
            # 364 is small enough for a Python loop with numpy sum.
            for b_idx in candidate_block_indices:
                covered = block_covered_subsets[b_idx]
                # Count how many of these are still uncovered
                gain = np.sum(uncovered[covered])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            # Add block to solution
            blocks_used.append(all_k_blocks[best_block_idx])
            
            # Update uncovered
            uncovered[block_covered_subsets[best_block_idx]] = False
            
        return blocks_used

    def try_remove_redundant(blocks):
        """
        Remove blocks that are redundant.
        """
        if not blocks:
            return blocks
            
        # Map t-subset to count of blocks covering it
        # We need to know which blocks cover which t-subsets.
        # blocks is a list of tuples (points).
        
        # Re-index blocks to their global index for lookup
        # all_k_blocks is a list. We can create a dict.
        block_to_global_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
        
        coverage_count = defaultdict(int)
        block_t_subsets_cache = []
        
        for block in blocks:
            b_tuple = tuple(block)
            g_idx = block_to_global_idx[b_tuple]
            ts_indices = block_covered_subsets[g_idx]
            block_t_subsets_cache.append(ts_indices)
            for idx in ts_indices:
                coverage_count[idx] += 1
        
        new_blocks = []
        for i, block in enumerate(blocks):
            ts_indices = block_t_subsets_cache[i]
            # Check if all t-subsets in this block are covered by > 1 block
            redundant = True
            for idx in ts_indices:
                if coverage_count[idx] <= 1:
                    redundant = False
                    break
            
            if not redundant:
                new_blocks.append(block)
            else:
                # Remove this block, decrement counts
                for idx in ts_indices:
                    coverage_count[idx] -= 1
        
        return new_blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    seeds = [42, 123, 7, 99, 2024, 555, 111, 222, 333, 444]
    
    for seed in seeds:
        if time_elapsed() > time_limit - 5:
            break
            
        rng = random.Random(seed)
        blocks = greedy_cover_restricted(rng)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # Polish
        if time_elapsed() < time_limit - 10:
            polished = try_remove_redundant(best_blocks)
            if len(polished) < best_count:
                best_count = len(polished)
                best_blocks = polished

    # If we ran out of time or didn't find a full cover, ensure we return something valid
    # The greedy loop breaks if time runs out, so `uncovered` might not be empty.
    # We need to ensure the returned blocks actually cover everything.
    # If we broke early, we must finish the cover.
    
    if best_blocks is None:
        # Fallback
        rng = random.Random(0)
        best_blocks = greedy_cover_restricted(rng)
        
    # Verification and Completion
    covered = np.zeros(n_t_subsets, dtype=bool)
    for block in best_blocks:
        b_tuple = tuple(block)
        # Find global index
        # This lookup is slow if done in a loop for every block.
        # Let's use a dict for the final set of blocks.
        pass
        
    # Actually, let's just re-run the greedy logic to ensure completion if time permits, 
    # or just trust the greedy loop which only breaks on time limit.
    # If it breaks on time limit, the cover is invalid.
    # We must complete the cover.
    
    # Let's check if the current best_blocks is a valid cover.
    # If not, we need to add more blocks.
    
    covered = np.zeros(n_t_subsets, dtype=bool)
    block_to_global_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    
    for block in best_blocks:
        g_idx = block_to_global_idx[tuple(block)]
        covered[block_covered_subsets[g_idx]] = True
        
    if not np.all(covered):
        # We have uncovered subsets. We must cover them.
        # Use a simple greedy to finish.
        uncovered = ~covered
        while np.any(uncovered):
            if time_elapsed() > time_limit - 1:
                # Emergency: return what we have? No, must be valid.
                # We have 1s left. We can't guarantee validity if we can't finish.
                # But the greedy step is fast.
                pass
                
            unc_idx = np.nonzero(uncovered)[0]
            target_sub_idx = unc_idx[0] # Pick first
            candidate_block_indices = blocks_containing_subset[target_sub_idx]
            
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidate_block_indices:
                covered_arr = block_covered_subsets[b_idx]
                gain = np.sum(uncovered[covered_arr])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            best_blocks.append(all_k_blocks[best_block_idx])
            uncovered[block_covered_subsets[best_block_idx]] = False

    return [list(b) for b in best_blocks]

def _fallback_greedy(v, k, t):
    """Simple fallback for non-standard parameters."""
    start_time = time.time()
    time_limit = 115
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        ts = min(uncovered)
        block = set(ts)
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(sorted(block), t - 1):
                    ts_new = tuple(sorted(rest + (x,)))
                    if ts_new in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.add(best_point)
        
        blocks.append(sorted(block))
        for covered_ts in itertools.combinations(block, t):
            uncovered.discard(covered_ts)
            
    return blocks
