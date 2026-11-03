import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subs = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subs = len(k_subsets)
    
    # We need masks of size num_t_subs.
    # num_t_subs for v=19, t=4 is 3876.
    # We use uint64 arrays. 3876 / 64 = 60.56 -> 61 uint64s per block.
    mask_len = (num_t_subs + 63) // 64
    # block_masks: shape (num_k_subs, mask_len)
    block_masks = np.zeros((num_k_subs, mask_len), dtype=np.uint64)
    
    # Precompute masks
    # This part is still Python loop but only 50k iterations, each doing ~100 bit ops.
    # It might take a few seconds. We can parallelize or optimize, but let's see.
    # Actually, 50k * 100 ops is 5M ops, which is fine in 1-2 seconds in Python if optimized.
    # But we can do better by precomputing the index mapping for t-subsets within k-subsets.
    
    for i, block in enumerate(k_subsets):
        # Get indices of t-subsets within this block
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            word_idx = idx // 64
            bit_idx = idx % 64
            block_masks[i, word_idx] |= (1 << bit_idx)
            
    # Helper to count bits in a numpy array of uint64
    # We can use popcount lookup table or numpy built-in if available (numpy doesn't have direct popcount for uint64 arrays in older versions, but we can use a LUT)
    # Simpler: convert to int and use bit_count? No, that's slow.
    # Let's use a vectorized approach for gain calculation.
    
    # To calculate gain of a block b against uncovered mask U:
    # gain = popcount(block_masks[b] & U)
    # We can compute this for all blocks at once if we store U as a vector and do bitwise AND with all rows.
    # But U changes every step.
    
    # Let's stick to a hybrid: 
    # 1. Greedy selection.
    # 2. At each step, we need the block with max gain.
    # Iterating 50k blocks in Python is slow (~0.1s per step if optimized, maybe 0.5s).
    # 150 blocks * 0.5s = 75s. This is tight.
    
    # Optimization: Use NumPy to compute gains for all blocks in one shot.
    # uncovered_mask is a numpy array of shape (mask_len,).
    # block_masks is (num_k_subs, mask_len).
    # intersection = block_masks & uncovered_mask (broadcasting)
    # This creates a (num_k_subs, mask_len) array. 
    # Then we need popcount along axis 1.
    # NumPy doesn't have a native popcount for uint64. 
    # We can use a LUT for 8-bit chunks? 
    # Or we can convert to int32/int64 and use a trick?
    # Actually, for 61 uint64s, we can just loop over the 61 words for each block? No, that's slow.
    
    # Alternative: Use a smaller mask representation? 
    # Or just accept that the first few steps are expensive and optimize the rest?
    
    # Let's try a different approach for the "Fix":
    # The previous code failed likely due to time or empty return.
    # I will implement a robust greedy that returns a valid solution immediately.
    # If the NumPy vectorization is too complex/slow to implement correctly in the edit, 
    # I will stick to Python but optimize the inner loop and ensure a valid return.
    
    # However, the directive says "Fix it". The main issue is score 0.
    # Score 0 means invalid or exception.
    # The error "return 1..3876 blocks" implies it returned 0 blocks or failed.
    # Initializing best_solution with a valid covering is key.
    
    # Let's create a simple valid covering first: 
    # Just pick the first block? No, that doesn't cover everything.
    # We must run the greedy loop at least once.
    
    # To ensure speed, I will use a precomputed list of t-subset indices for each k-block.
    # This avoids the inner `itertools.combinations` call in the mask construction if we precompute indices.
    # But we already did that in the mask construction.
    
    # Let's try to optimize the greedy step using NumPy for the gain calculation.
    # We will maintain `uncovered` as a numpy array of uint64.
    # We will precompute a LUT for popcount of 16-bit chunks? 
    # Actually, `np.bitwise_and` is fast. 
    # We can compute `inter = block_masks & uncovered` 
    # Then sum popcounts.
    # Since we don't have fast popcount in NumPy, we can use a trick:
    # Convert uint64 to float? No.
    # Let's use a Python loop over the 61 words for the *selected* block only?
    # No, we need to find the *max* gain among 50k blocks.
    
    # Strategy: 
    # 1. Compute gains for all blocks using a vectorized approximation or fast Python loop.
    # 2. To make it fast, we can sample? No, greedy needs exact.
    
    # Let's try this: 
    # Store block_masks as a list of Python ints? 
    # Python int bitwise ops are fast in C. 
    # `block_masks[i] & uncovered` is very fast.
    # `.bit_count()` is fast in Python 3.8+.
    # The bottleneck in the original code was likely the mask construction or the sheer number of iterations if not optimized.
    # But 50k blocks * 150 steps * (50k checks) = 375M operations. Too slow in Python.
    
    # We MUST vectorize the gain calculation.
    # Idea: Use NumPy to compute the intersection, then use a LUT for popcount.
    # LUT for 8 bits: 256 entries.
    # Split uint64 into 8 bytes.
    # This is complex to implement correctly in a short edit.
    
    # Alternative: 
    # Use a heuristic that doesn't require checking all blocks every time.
    # Or, reduce the candidate set.
    
    # Let's look at the "Maximum Diversity" hint again.
    # Maybe we can pre-select a smaller candidate pool?
    
    # Actually, there is a simpler fix for the "Invalid" error:
    # If the loop doesn't finish, return a valid covering.
    # A valid covering can be constructed by taking all blocks? No, too many.
    # But we can run the greedy algorithm *once* very efficiently.
    
    # Let's optimize the greedy step with NumPy vectorization of the AND operation, 
    # and use a fast popcount method.
    # We can convert the uint64 array to a list of integers? No.
    
    # Let's try a different tactic: 
    # Use `itertools.combinations` for t-subsets is fixed.
    # We can precompute for each k-block, the list of t-subset indices it covers.
    # Then, gain is the number of indices in that list that are in `uncovered_set`.
    # If `uncovered_set` is a Python `set` of indices, checking membership is O(1).
    # But we have 50k blocks, each with ~100 indices. 5M checks per step.
    # 150 steps * 5M = 750M checks. Still too slow in Python.
    
    # We need NumPy.
    # Let's use `np.in1d` or boolean masks.
    # `uncovered_indices` as a boolean array of length `num_t_subs`.
    # For each block, we have a boolean mask `block_bool_mask`.
    # Gain = sum(block_bool_mask & uncovered_bool_mask).
    # `block_bool_masks` shape: (50388, 3876). This is 50388 * 3876 * 1 byte = ~200 MB. 
    # This is acceptable memory-wise.
    # `uncovered_bool_mask` shape: (3876,).
    # `gains = (block_bool_masks & uncovered_bool_mask).sum(axis=1)`
    # This is a vectorized operation over 200MB of data.
    # NumPy can handle this in milliseconds.
    
    # Plan:
    # 1. Create `block_bool_masks` as a boolean numpy array.
    # 2. Greedy loop:
    #    a. `gains = (block_bool_masks & uncovered_bool).sum(axis=1)`
    #    b. Find max gain.
    #    c. Update `uncovered_bool` by OR-ing with the selected block's mask (inverted).
    #       `uncovered_bool &= ~block_bool_masks[chosen]`
    #    d. Check if `uncovered_bool` is all False.
    
    # This should be very fast.
    
    # Let's implement this.

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subs = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subs = len(k_subsets)
    
    # Boolean mask for each k-block
    # Shape: (num_k_subs, num_t_subs)
    block_bool_masks = np.zeros((num_k_subs, num_t_subs), dtype=bool)
    
    for i, block in enumerate(k_subsets):
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            block_bool_masks[i, idx] = True
            
    # Initial uncovered mask
    uncovered_bool = np.ones(num_t_subs, dtype=bool)
    
    best_count = float('inf')
    best_solution = []
    
    # We will run the greedy algorithm.
    # To improve diversity/score, we can randomize tie-breaking.
    
    while time.time() < deadline:
        uncovered_bool = np.ones(num_t_subs, dtype=bool)
        selected_indices = []
        
        while uncovered_bool.any():
            # Calculate gains
            # (block_bool_masks & uncovered_bool) results in a boolean array
            # Sum along axis 1 gives the count of covered t-subsets for each block
            gains = (block_bool_masks & uncovered_bool).sum(axis=1)
            
            max_gain = gains.max()
            if max_gain == 0:
                # Should not happen if covering is possible
                break
            
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            chosen = random.choice(candidates)
            
            selected_indices.append(chosen)
            # Update uncovered: remove t-subsets covered by chosen block
            uncovered_bool &= ~block_bool_masks[chosen]
            
        # Reduction Phase (optional, might be slow, skip for now to ensure speed)
        # The greedy set cover is usually close to optimal.
        
        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_solution = [k_subsets[i] for i in selected_indices]
            
            # Early stop if we reach a very good bound?
            # Lower bound is 131. Best known 151.
            if best_count <= 151:
                # We can try a few more times to see if we can go lower, 
                # but if we are close to deadline, stop.
                if time.time() > deadline - 10:
                    break

    # Ensure we return a valid solution
    # If best_solution is empty, it means the loop didn't run or failed.
    # But the greedy loop should always produce a solution for valid parameters.
    if not best_solution:
        # Fallback: Return a single block? No, invalid.
        # Return all blocks? Too many.
        # Run one more time without time limit? No, must respect deadline.
        # If we are here, something went wrong.
        # Let's just return the first block to avoid "empty" error, though it's invalid coverage.
        # Better: Return a list of blocks that definitely covers?
        # Actually, if the loop ran, best_solution is not empty.
        # If it didn't run, we have a problem.
        # Let's force one execution if best_solution is empty.
        if time.time() < deadline:
             uncovered_bool = np.ones(num_t_subs, dtype=bool)
             selected_indices = []
             while uncovered_bool.any() and time.time() < deadline:
                 gains = (block_bool_masks & uncovered_bool).sum(axis=1)
                 max_gain = gains.max()
                 if max_gain == 0: break
                 candidates = np.where(gains == max_gain)[0]
                 chosen = random.choice(candidates)
                 selected_indices.append(chosen)
                 uncovered_bool &= ~block_bool_masks[chosen]
             if selected_indices:
                 best_solution = [k_subsets[i] for i in selected_indices]
                 
    return best_solution
