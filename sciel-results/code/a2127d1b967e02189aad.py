"""Covering design C(v,k,t) with vectorized greedy + reduction + local search."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 120  # safety margin before 150s kill
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute all possible k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the indices of t-subsets it covers
    # block_to_tsubs[i] = list of t-subset indices covered by block i
    block_to_tsubs = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        block_to_tsubs.append(indices)
    
    # Convert to numpy array for vectorized operations
    # Use a sparse representation: for each block, store the list of t-subset indices
    # For efficient gain computation, we can use a dense matrix or precomputed sets
    
    # Precompute for each t-subset, which blocks cover it (reverse index)
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, indices in enumerate(block_to_tsubs):
        for idx in indices:
            tsub_to_blocks[idx].append(bi)
    
    def get_block_tsub_indices(block_tuple):
        """Return list of t-subset indices covered by a block (as tuple)."""
        indices = []
        for ts in itertools.combinations(block_tuple, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        return indices
    
    def greedy_build(seed):
        """Randomized greedy covering."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int32)
        blocks_used = []
        used_block_indices = set()
        
        while uncovered.any():
            # Find t-subsets with the fewest covering blocks (most constrained)
            # For each uncovered t-subset, count how many unused blocks cover it
            gains = np.zeros(n_tsubs, dtype=np.int32)
            
            # For efficiency, only consider uncovered t-subsets
            uncovered_indices = np.where(uncovered == 1)[0]
            if len(uncovered_indices) == 0:
                break
            
            # For each uncovered t-subset, find the best block to add
            # We want to maximize the number of NEW t-subsets covered
            
            # Randomize the order of considering uncovered t-subsets
            idx_order = uncovered_indices.tolist()
            rng.shuffle(idx_order)
            
            best_block_idx = -1
            best_gain = -1
            best_block_tsubs = []
            
            # Sample a subset of uncovered t-subsets to find a good block
            # If there are too many, sample randomly
            sample_size = min(len(idx_order), 100)
            sample_idx = rng.sample(idx_order, sample_size)
            
            for ts_idx in sample_idx:
                # Get blocks that cover this t-subset
                covering_blocks = tsub_to_blocks[ts_idx]
                if not covering_blocks:
                    continue
                
                # Shuffle to break ties randomly
                rng.shuffle(covering_blocks)
                
                # Try each covering block and compute gain
                for bi in covering_blocks[:10]:  # Limit to first 10 for speed
                    if bi in used_block_indices:
                        continue
                    
                    # Compute how many currently uncovered t-subsets this block covers
                    block_tsubs = block_to_tsubs[bi]
                    gain = 0
                    for idx in block_tsubs:
                        if uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = bi
                        best_block_tsubs = block_tsubs
            
            # If no block found, try all covering blocks for the first uncovered
            if best_block_idx == -1:
                ts_idx = idx_order[0]
                covering_blocks = tsub_to_blocks[ts_idx]
                rng.shuffle(covering_blocks)
                for bi in covering_blocks:
                    if bi in used_block_indices:
                        continue
                    block_tsubs = block_to_tsubs[bi]
                    gain = 0
                    for idx in block_tsubs:
                        if uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = bi
                        best_block_tsubs = block_t_subs if False else block_to_tsubs[bi]
            
            if best_block_idx == -1:
                # Should not happen if covering is possible
                break
            
            # Add the block
            used_block_indices.add(best_block_idx)
            blocks_used.append(all_blocks[best_block_idx])
            
            # Update uncovered
            for idx in best_block_tsubs:
                uncovered[idx] = 0
        
        return blocks_used
    
    def reduce_blocks(blocks):
        """Remove redundant blocks."""
        if not blocks:
            return blocks
        
        # Build coverage count
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs_list = []
        for block in blocks:
            indices = get_block_tsub_indices(tuple(block))
            block_tsubs_list.append(indices)
            for idx in indices:
                cov_count[idx] += 1
        
        blocks_list = list(blocks)
        tsubs_list = list(block_tsubs_list)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks_list)):
                # Try removing block i
                all_redundant = True
                for idx in tsubs_list[i]:
                    if cov_count[idx] <= 1:
                        all_redundant = False
                        break
                
                if all_redundant:
                    for idx in tsubs_list[i]:
                        cov_count[idx] -= 1
                    blocks_list.pop(i)
                    tsubs_list.pop(i)
                    improved = True
                    break
        
        return blocks_list
    
    def local_search(blocks, max_iters=10):
        """Try swapping blocks to improve coverage density."""
        if not blocks:
            return blocks
        
        blocks_list = list(blocks)
        
        for _ in range(max_iters):
            # Try removing each block and see if we can replace it with a better one
            for i in range(len(blocks_list)):
                # Temporarily remove block i
                saved_block = blocks_list[i]
                blocks_list.pop(i)
                
                # Check if covering is still valid
                if not _is_valid_covering(blocks_list, v, k, t, all_tsubs, tsub_index, n_tsubs):
                    blocks_list.insert(i, saved_block)
                    continue
                
                # Try to find a block that covers the most uncovered t-subsets
                # after removal
                # Compute current coverage
                cov_count = np.zeros(n_tsubs, dtype=np.int32)
                for block in blocks_list:
                    for idx in get_block_tsub_indices(tuple(block)):
                        cov_count[idx] += 1
                
                # Find t-subsets with lowest coverage
                min_cov = cov_count.min()
                if min_cov >= 1:
                    # Covering is still valid, keep the removal
                    continue
                else:
                    # Need to add a block back
                    blocks_list.insert(i, saved_block)
        
        return blocks_list
    
    def _is_valid_covering(blocks, v, k, t, all_tsubs, tsub_index, n_tsubs):
        """Check if the given blocks form a valid covering."""
        cov = np.zeros(n_tsubs, dtype=np.int32)
        for block in blocks:
            for idx in get_block_tsub_indices(tuple(block)):
                cov[idx] += 1
        return cov.min() >= 1
    
    # First, do a deterministic greedy to get a baseline
    blocks = greedy_build(0)
    best_blocks = reduce_blocks(blocks)
    best_count = len(best_blocks)
    
    # Then try randomized restarts
    seed = 1
    while time.time() < deadline - 20:
        blocks = greedy_build(seed)
        reduced = reduce_blocks(blocks)
        if len(reduced) < best_count:
            best_blocks = reduced
            best_count = len(reduced)
        seed += 1
        if seed % 5 == 0:
            # Occasionally try local search on the best so far
            if time.time() < deadline - 30:
                ls_blocks = local_search(best_blocks, max_iters=3)
                ls_blocks = reduce_blocks(ls_blocks)
                if len(ls_blocks) < best_count:
                    best_blocks = ls_blocks
                    best_count = len(ls_blocks)
    
    # Final reduction
    if best_blocks:
        best_blocks = reduce_blocks(best_blocks)
    
    return best_blocks if best_blocks else greedy_build(0)
