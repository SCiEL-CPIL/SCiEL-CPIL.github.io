import numpy as np
import time
import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precomputation
    # 1. List of all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. For each point, list of t-subsets containing it (for potential future use, maybe not needed for batch)
    # Actually, for batch construction, we need to score blocks.
    # A block covers C(k, t) t-subsets.
    # We want to pick blocks that cover many UNCOVERED t-subsets.
    
    # To speed up scoring, we can precompute the index of each t-subset.
    # But calculating the score of a random block requires checking if its t-subsets are uncovered.
    # If we maintain a boolean array `uncovered`, we can sum uncovered[subset_indices].
    
    # Precompute all possible blocks? C(19, 7) = 50388. This is small enough to precompute all their t-subset indices.
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map block to its index for identification
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute for each block, the list of t-subset indices it covers
    # This is a list of lists. 50388 blocks * 35 subsets = ~1.7M integers.
    # Memory is fine. Time to precompute: 50388 * 35 lookups. Fast.
    block_t_subset_indices = []
    for b in all_blocks:
        subs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_t_subset_indices.append(subs)
    
    # Convert to numpy arrays for vectorized operations if needed, or keep as lists for flexibility
    # Let's keep as lists of ints for now, or numpy arrays for faster summing with masks.
    block_t_subset_indices_np = [np.array(subs, dtype=np.int32) for subs in block_t_subset_indices]

    best_blocks = None
    best_count = float('inf')
    
    def batch_greedy(uncovered_mask):
        """
        Iteratively select a batch of blocks that maximize coverage of uncovered subsets.
        """
        current_blocks = []
        uncovered = uncovered_mask.copy()
        remaining_block_indices = list(range(num_blocks))
        
        # We will select blocks in batches.
        # Batch size: Let's try to select 10-20% of remaining blocks per iteration?
        # Or just select the top N scoring blocks.
        
        while uncovered.any() and time.time() < deadline:
            # Calculate score for all remaining blocks
            # Score = sum(uncovered[subset_indices])
            # This is a matrix multiplication or gather operation.
            # Since block sizes are small (35), a loop is okay if vectorized inside.
            
            scores = np.zeros(len(remaining_block_indices), dtype=np.int32)
            for i, block_idx in enumerate(remaining_block_indices):
                subs = block_t_subset_indices_np[block_idx]
                scores[i] = np.sum(uncovered[subs])
            
            # Find the best block
            if scores.max() == 0:
                break # No block covers any new subset (should not happen if uncovered is valid)
                
            best_local_idx = np.argmax(scores)
            best_global_idx = remaining_block_indices[best_local_idx]
            
            # Select this block
            current_blocks.append(all_blocks[best_global_idx])
            
            # Update uncovered mask
            subs = block_t_subset_indices_np[best_global_idx]
            uncovered[subs] = False
            
            # Remove this block from remaining pool
            remaining_block_indices.pop(best_local_idx)
            
            # Optimization: If we want a "batch" approach, we could select top K blocks at once.
            # But greedy sequential is safer for correctness of "maximizing immediate gain".
            # The "Batch" inspiration suggests selecting multiple at once.
            # Let's try a hybrid: Select top M blocks, then update mask.
            # Actually, the standard greedy is sequential. The "Parallel Batch" might mean
            # evaluating many candidates in parallel (which we do with numpy) but picking one.
            # Or it means picking a set of non-conflicting high-score blocks.
            
            # Let's stick to picking the single best block each step for stability,
            # but the vectorized scoring is the key speedup over the current python-loop greedy.
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        return current_blocks

    def local_search(blocks, time_limit):
        """
        Improve the set of blocks by removing redundant ones and swapping.
        """
        blocks = [b[:] for b in blocks]
        search_start = time.time()
        
        # Build coverage counts
        coverage_count = np.zeros(num_subsets, dtype=np.int32)
        block_subsets = []
        
        for block in blocks:
            subs = []
            # Find index of this block in precomputed list?
            # It's faster to just recompute indices for the current block set
            # since we have subset_to_idx.
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                coverage_count[idx] += 1
                subs.append(idx)
            block_subsets.append(subs)
        
        improved = True
        while improved and (time.time() - search_start) < time_limit:
            improved = False
            
            # 1. Remove redundant blocks
            for i in range(len(blocks) - 1, -1, -1):
                if (time.time() - search_start) > time_limit:
                    break
                redundant = True
                for idx in block_subsets[i]:
                    if coverage_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for idx in block_subsets[i]:
                        coverage_count[idx] -= 1
                    blocks.pop(i)
                    block_subsets.pop(i)
                    improved = True
            
            if (time.time() - search_start) > time_limit:
                break
            
            # 2. Try to swap blocks to improve coverage or allow further reduction
            # Simple swap: Replace block i with a random block j that covers all unique subsets of i.
            # To make this fast, we precompute for each block the set of subsets it uniquely covers.
            # But that changes as we remove blocks.
            
            # Alternative: Random restart / Perturbation
            # Let's try a "1-swap" where we replace a block with a random one if it doesn't lose coverage.
            for i in range(len(blocks)):
                if (time.time() - search_start) > time_limit:
                    break
                
                # Find subsets uniquely covered by block i
                unique_subsets = [idx for idx in block_subsets[i] if coverage_count[idx] == 1]
                if not unique_subsets:
                    continue
                
                unique_set = set(unique_subsets)
                
                # Try a few random replacements
                for _ in range(5):
                    if (time.time() - search_start) > time_limit:
                        break
                    new_block_idx = random.randint(0, num_blocks - 1)
                    new_block = all_blocks[new_block_idx]
                    new_subs = [subset_to_idx[sub] for sub in itertools.combinations(new_block, t)]
                    new_set = set(new_subs)
                    
                    # Check if new_set covers all unique_subsets
                    if unique_set.issubset(new_set):
                        # Perform swap
                        for idx in block_subsets[i]:
                            coverage_count[idx] -= 1
                        for idx in new_subs:
                            coverage_count[idx] += 1
                        blocks[i] = new_block
                        block_subsets[i] = new_subs
                        improved = True
                        break
        
        return blocks

    attempts = 0
    while time.time() < deadline - 10:  # Leave 10s for final check
        attempts += 1
        uncovered = np.ones(num_subsets, dtype=bool)
        
        # Use the fast vectorized greedy
        current_blocks = batch_greedy(uncovered)
        
        # Local search
        remaining = deadline - time.time()
        if remaining > 5:
            ls_time = min(remaining * 0.4, 30.0)
            current_blocks = local_search(current_blocks, ls_time)
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [b[:] for b in current_blocks]
            if best_count <= 131:
                break
        
        # If we are stuck at a high number, we might want to randomize the start
        # The current greedy is deterministic. Let's add randomness to the tie-breaking or initial selection.
        # For now, the vectorized speedup allows many more attempts or better local search.
        
        # To diversify, we can randomly pick one of the top 10 blocks in the first few steps.
        # Let's modify batch_greedy to accept a randomization parameter?
        # Or just run multiple times with different random seeds in local search.
        # The local search already uses random.
        
        # To get different greedy outcomes, we can shuffle the initial "uncovered" priorities?
        # No, greedy is based on counts.
        # Let's introduce a small random perturbation to the scores in batch_greedy.
        
    if best_blocks is None:
        # Fallback
        uncovered = np.ones(num_subsets, dtype=bool)
        best_blocks = batch_greedy(uncovered)
        
    return best_blocks
