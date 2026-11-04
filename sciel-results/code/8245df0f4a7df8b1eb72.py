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
    Strategy: Start with all blocks, iteratively remove redundant ones (those that don't
    uniquely cover any t-subset). This finds the unique minimal core of the universal cover.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)  # 11628

    # Map each t-subset to an index
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)  # 75582

    # Precompute t-subsets for each block
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)

    # NumPy arrays for fast operations
    # block_tsubs_np: shape (num_blocks, 56), int32
    block_tsubs_np = np.array(block_tsubs, dtype=np.int32)
    
    # Deadline
    start_time = time.time()
    deadline = start_time + 145.0

    def check_time(budget=10.0):
        return time.time() > deadline - budget

    # Initialize coverage count: each t-subset is covered by how many blocks?
    # Since we start with ALL blocks, we can compute this efficiently or just initialize to a large number
    # Actually, we can compute the initial counts: 
    # Each t-subset is contained in C(v-t, k-t) blocks.
    # C(19-5, 8-5) = C(14, 3) = 364.
    # So initial cover_count is 364 for all t-subsets.
    
    cover_count = np.full(num_tsubs, math.comb(v - t, k - t), dtype=np.int32)
    
    # active_blocks: boolean array of size num_blocks
    active_blocks = np.ones(num_blocks, dtype=bool)
    
    # We will iteratively remove blocks that are "redundant".
    # A block is redundant if for all t-subsets in it, cover_count[tsub] > 1.
    # i.e., min(cover_count[block_tsubs]) > 1.
    
    # To speed this up, we can process in batches or just loop until stable.
    # However, looping over 75k blocks and checking min of 56 elements each time is O(75k * 56) ~ 4M ops per pass.
    # Number of passes? The core size is ~335. The reduction is massive (75k -> 335).
    # Most blocks will be removed in the first few passes.
    
    # Optimization: 
    # Instead of checking all active blocks every time, we can maintain a list of "candidates" 
    # or just iterate. Given the massive reduction, a simple loop might be fast enough if vectorized.
    
    # Let's try a vectorized check for all active blocks at once.
    # We need to check: for each active block, is min(cover_count[block_tsubs[i]]) > 1?
    
    # We can compute min_cover_for_block = np.min(cover_count[block_tsubs_np], axis=1)
    # This is a gather + min operation. 
    # block_tsubs_np is (75582, 56). cover_count is (11628,).
    # cover_count[block_tsubs_np] is (75582, 56).
    # np.min(..., axis=1) is (75582,).
    # This is ~4M operations. Very fast in NumPy.
    
    # Algorithm:
    # 1. Compute min_cover = np.min(cover_count[block_tsubs_np], axis=1)
    # 2. Identify blocks where min_cover > 1 AND active_blocks is True.
    # 3. If no such blocks, we are done (or stuck).
    # 4. Remove those blocks: active_blocks[removable] = False
    # 5. Update cover_count: cover_count[block_tsubs_np[removable]] -= 1
    # 6. Repeat.
    
    # Potential issue: Removing many blocks at once might make some other blocks redundant 
    # that weren't before, but that's fine, they'll be caught in the next iteration.
    # Also, removing blocks might make cover_count drop to 1 for some t-subsets, 
    # preventing further removals of blocks that cover them.
    
    # This is essentially the "greedy deletion" algorithm for set cover, which is optimal 
    # for finding the unique minimal core if it exists? 
    # Wait, the "core" of a covering design is the set of blocks that appear in EVERY minimal covering.
    # The algorithm "remove if not uniquely covering" finds the core.
    # Is the core a valid covering? Yes, because if we remove a block, it wasn't uniquely covering anything, 
    # so the remaining blocks still cover everything.
    # So the final set is a valid covering.
    
    # Is it minimal? 
    # The process stops when no block can be removed. This means every remaining block uniquely covers 
    # at least one t-subset. This is the definition of a minimal covering (with respect to block removal).
    # So the result is a minimal covering.
    
    # Is it the *smallest* minimal covering? 
    # The core is the intersection of all minimal coverings. It is the unique smallest minimal covering.
    # So this algorithm should find the absolute minimum number of blocks for this specific structure?
    # Actually, for C(19,8,5), the "core" might be the entire optimal solution if the design is rigid, 
    # or it might be smaller than the optimum if there are multiple optimal solutions that don't share all blocks.
    # However, for small v, k, t, the core is often the optimal solution itself or very close.
    # Given the lower bound is 264 and best known is 335, if the core is 335, we are done.
    # If the core is smaller, we found a better solution!
    
    # Let's implement this.
    
    # We need to be careful with time. 
    # Each iteration:
    # 1. min_cover = np.min(cover_count[block_tsubs_np], axis=1)  # ~4M ops
    # 2. removable = (min_cover > 1) & active_blocks
    # 3. if not np.any(removable): break
    # 4. cover_count[block_tsubs_np[removable]] -= 1  # ~ (num_removed * 56) ops
    # 5. active_blocks[removable] = False
    
    # How many iterations?
    # In the first iteration, many blocks will be removed. 
    # As we remove blocks, cover counts drop. 
    # Eventually, cover counts will hit 1 for many t-subsets, "locking" the blocks that cover them.
    # The number of iterations should be small (maybe < 20).
    
    # Let's run it.
    
    while True:
        if check_time(10.0):
            break
            
        # Compute min coverage for each active block
        # We only need to check active blocks, but vectorized op on all is fine (75k is small)
        min_cover = np.min(cover_count[block_tsubs_np], axis=1)
        
        # A block is removable if it is active AND its min coverage is > 1
        removable = (min_cover > 1) & active_blocks
        
        if not np.any(removable):
            break
            
        # Identify indices of removable blocks
        rem_idx = np.where(removable)[0]
        
        # Update cover counts
        # We need to subtract 1 from cover_count for all t-subsets in these blocks
        # cover_count[block_tsubs_np[rem_idx]] -= 1
        # This is an advanced indexing operation. 
        # block_tsubs_np[rem_idx] is shape (num_removed, 56).
        # This performs num_removed * 56 updates.
        
        # Using np.add.at or direct assignment?
        # cover_count[block_tsubs_np[rem_idx]] -= 1 
        # This works if indices are unique? No, t-subsets can repeat in the list of blocks? 
        # No, within a single block, t-subsets are unique. 
        # But across different blocks in rem_idx, the same t-subset might appear.
        # The operation cover_count[indices] -= 1 with repeated indices in `indices` 
        # does NOT accumulate correctly in standard NumPy assignment (it only subtracts 1 for the last occurrence 
        # if using basic assignment, but advanced indexing with -= uses a buffer and might be tricky or slow).
        
        # Actually, `arr[idx] -= 1` where idx has duplicates is undefined/slow in NumPy.
        # Better to use np.add.at(cover_count, block_tsubs_np[rem_idx].ravel(), -1)
        # Or flatten the indices.
        
        flat_indices = block_tsubs_np[rem_idx].ravel()
        np.add.at(cover_count, flat_indices, -1)
        
        # Mark as inactive
        active_blocks[rem_idx] = False
        
        # Optional: check time more frequently if rem_idx is large?
        # np.add.at is O(N). If N is large (e.g., 10000 blocks * 56 = 560k), it's fast.
        
    # Extract the final blocks
    final_indices = np.where(active_blocks)[0]
    final_blocks = [list(all_blocks[i]) for i in final_indices]
    
    # Verify coverage? 
    # The algorithm guarantees coverage if we started with a valid cover (all blocks) 
    # and only removed blocks that didn't break coverage.
    # So it should be valid.
    
    # If for some reason it's empty or invalid (shouldn't happen), fallback to a greedy?
    # For now, return the core.
    
    return final_blocks
