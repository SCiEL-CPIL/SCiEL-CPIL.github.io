import itertools
import numpy as np
import time
import random

# Precomputation for v=19, k=7, t=4
# This is specific to the task parameters, but the function takes v,k,t.
# We should handle general v,k,t but optimize for the specific case if possible?
# The prompt says "evaluator passes this task's parameters".
# We can cache precomputations if v,k,t are fixed, but construct is called once?
# "construct(v, k, t) -> list of blocks".
# We can do precomputation inside construct if it's fast enough, or use a global cache.

_V = 19
_K = 7
_T = 4

# Global cache for precomputation
_cache = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _cache:
        return _cache[key]
    
    # Map t-subsets to IDs
    t_subsets = list(itertools.combinations(range(v), t))
    t_id_map = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Map k-subsets to IDs and their t-subset IDs
    k_subsets = list(itertools.combinations(range(v), k))
    block_t_ids = np.zeros((len(k_subsets), t), dtype=np.int32) # Wait, t is 4, but we need 35 ids per block?
    # No, a block of size k contains C(k, t) t-subsets.
    # For v=19, k=7, t=4, C(7,4) = 35.
    # So block_t_ids should be shape (n_blocks, C(k,t))
    
    c_k_t = len(list(itertools.combinations(range(k), t)))
    block_t_ids = np.zeros((len(k_subsets), c_k_t), dtype=np.int32)
    
    for i, block in enumerate(k_subsets):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_ids[i, j] = t_id_map[sub]
            
    _cache[key] = (t_subsets, t_id_map, k_subsets, block_t_ids, n_t, c_k_t)
    return _cache[key]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    t_subsets, t_id_map, k_subsets, block_t_ids, n_t, c_k_t = _get_precomputed(v, k, t)
    n_blocks = len(k_subsets)
    
    best_blocks = None
    best_size = float('inf')
    
    # 1. Randomized Greedy Construction
    # We will run several iterations of greedy with random tie-breaking
    # and then apply a deletion pass.
    
    def run_greedy(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Calculate gains
            # gains[i] = number of uncovered t-subsets covered by block i
            # block_t_ids shape (n_blocks, c_k_t)
            # uncovered shape (n_t,)
            # We need sum(uncovered[ids] for ids in block)
            # Vectorized: 
            # This might be slow if done naively.
            # block_t_ids is (n_blocks, 35).
            # We can do: gains = uncovered[block_t_ids].sum(axis=1)
            # This creates an array of size (n_blocks, 35) of bools, then sums.
            # 50388 * 35 = 1.7M elements. This is fast in NumPy.
            
            gains = uncovered[block_t_ids].sum(axis=1)
            
            # Find max gain
            max_gain = gains.max()
            if max_gain == 0:
                # Should not happen if uncovered.any()
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one
            chosen_idx = candidates[rng.randint(len(candidates))]
            
            blocks.append(k_subsets[chosen_idx])
            
            # Update uncovered
            uncovered[block_t_ids[chosen_idx]] = False
            
        return blocks

    def deletion_pass(blocks):
        # blocks is a list of tuples
        # Convert to list of block indices for faster lookup?
        # Or just use the tuples.
        # We need to check if a block is redundant.
        # A block is redundant if all its t-subsets are covered by other blocks.
        
        # Let's maintain a count of coverage for each t-subset.
        counts = np.zeros(n_t, dtype=np.int32)
        
        # Initialize counts
        for b in blocks:
            # Find index of b in k_subsets?
            # Better: precompute a map from block tuple to index?
            # Or just use the t-subsets directly.
            # We can find the index of the block in k_subsets.
            # Since k_subsets is a list, we can create a dict.
            pass # We'll do this outside or pass indices.
        
        # Let's restructure to work with block indices.
        pass

    # Refined approach:
    # 1. Greedy returns list of block indices.
    # 2. Deletion pass works on indices.
    
    def run_greedy_indices(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered = np.ones(n_t, dtype=bool)
        block_indices = []
        
        while uncovered.any():
            gains = uncovered[block_t_ids].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(len(candidates))]
            block_indices.append(chosen_idx)
            uncovered[block_t_ids[chosen_idx]] = False
            
        return block_indices

    def deletion_pass_indices(block_indices):
        # block_indices is a list of ints
        # counts: coverage count for each t-subset
        counts = np.zeros(n_t, dtype=np.int32)
        
        # Initialize counts
        for bi in block_indices:
            counts[block_t_ids[bi]] += 1
            
        # Try to remove blocks
        # Shuffle order for randomness?
        # We'll iterate until no more removals.
        removed = True
        current_indices = list(block_indices)
        
        while removed:
            removed = False
            # Shuffle to avoid bias
            random.shuffle(current_indices)
            for bi in current_indices[:]: # Iterate over a copy
                # Check if all t-subsets of bi are covered by others
                # i.e., counts[block_t_ids[bi]] > 1 for all
                # If we remove bi, new counts = counts - 1 for those subsets.
                # If any new count is 0, it's not redundant.
                
                # Optimization: 
                # min_count = counts[block_t_ids[bi]].min()
                # If min_count > 1, then all are covered by others.
                
                if counts[block_t_ids[bi]].min() > 1:
                    # Safe to remove
                    counts[block_t_ids[bi]] -= 1
                    current_indices.remove(bi)
                    removed = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
                
        return current_indices

    # 3. SA on Deletion
    def sa_refinement(block_indices, max_iters=5000):
        # block_indices is a list of ints
        counts = np.zeros(n_t, dtype=np.int32)
        for bi in block_indices:
            counts[block_t_ids[bi]] += 1
            
        current_indices = list(block_indices)
        current_cost = len(current_indices)
        best_indices = list(current_indices)
        best_cost = current_cost
        
        T = 1.0
        cooling = 0.99
        M = 100 # Penalty multiplier
        
        rng = np.random.RandomState(42)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            if not current_indices:
                break
                
            # Pick random block
            idx = rng.randint(len(current_indices))
            bi = current_indices[idx]
            
            # Try to remove
            # Check if removal causes uncovered
            # new_counts = counts - 1 for block_t_ids[bi]
            # We don't want to modify counts in place if rejected.
            
            # Calculate uncovered count if removed
            # subset_ids = block_t_ids[bi]
            # new_counts_subset = counts[subset_ids] - 1
            # new_uncovered = sum(new_counts_subset == 0)
            
            subset_ids = block_t_ids[bi]
            new_counts_subset = counts[subset_ids] - 1
            new_uncovered = np.sum(new_counts_subset == 0)
            
            new_cost = (len(current_indices) - 1) + M * new_uncovered
            
            delta = new_cost - current_cost
            
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                # Accept
                # Update counts
                counts[subset_ids] -= 1
                current_indices.pop(idx)
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_indices = list(current_indices)
                    
            T *= cooling
            
        return best_indices

    # Main Loop
    # Run multiple greedy instances
    n_greedy_runs = 20
    
    for i in range(n_greedy_runs):
        if time.time() > deadline - 10:
            break
            
        bi_list = run_greedy_indices(i)
        bi_list = deletion_pass_indices(bi_list)
        
        # Refine with SA
        bi_list = sa_refinement(bi_list, max_iters=2000)
        
        if len(bi_list) < best_size:
            best_size = len(bi_list)
            best_blocks = [k_subsets[bi] for bi in bi_list]
            
    # Final fallback if no blocks found (shouldn't happen)
    if best_blocks is None:
        # Use a simple deterministic greedy
        bi_list = run_greedy_indices(0)
        bi_list = deletion_pass_indices(bi_list)
        best_blocks = [k_subsets[bi] for bi in bi_list]
        best_size = len(best_blocks)
        
    return [list(b) for b in best_blocks]
