import itertools
import numpy as np
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using an optimized greedy algorithm.
    """
    start_time = time.time()
    deadline = start_time + 110  # Keep a 10-second safety margin from 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets and map them to indices 0..N-1
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    
    # Map tuple -> index for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Initialize uncovered mask: True means uncovered
    uncovered = np.ones(n_subsets, dtype=bool)
    
    # Precompute for each point x, and for each potential "rest" combination?
    # Instead, we will compute gains on the fly.
    # To speed up gain calculation:
    # For a candidate point x and current block B (size j), 
    # new subsets = { (x,) + y for y in combinations(B, t-1) }
    # We need to count how many of these have uncovered[idx] == True.
    
    # We can store the indices for all combinations of points?
    # Let's precompute the indices of t-subsets for every (t-1)-subset combined with every point x.
    # Actually, simpler: Precompute for every point x, the list of (subset_tuple, index) 
    # where x is in the subset.
    # Then, for a block B, the subsets involving x and B are those in x's list 
    # that are also subsets of B U {x}.
    
    # Alternative fast path for specific v,k,t or general:
    # Let's just use the formula: gain(x) = sum(1 for y in combinations(B, t-1) if uncovered[subset_to_idx[(x,)+y]] is True)
    # This requires mapping (x,)+y to index.
    
    blocks = []
    
    # Optimization: If the problem is small, pure Python sets might be competitive if NumPy overhead is high.
    # But n_subsets is 11628 for v=19, t=5. NumPy is very fast.
    
    # Precompute indices for all possible (t-1) subsets combined with each x?
    # Let's create a lookup: idx_of[tuple] is already there.
    
    # To make `uncovered[subset_to_idx[...]]` fast, we can just use the dict lookup.
    
    # Greedy Loop
    while np.any(uncovered):
        if time.time() > deadline:
            # Return best effort (which is incomplete, but the prompt says "return best result found")
            # However, the task requires validity. If we return invalid, score is 0.
            # So we must ensure we finish. 
            # If we don't finish, we should probably return the current blocks? No, they are invalid.
            # The prompt says: "invalid result ... scores 0".
            # So we MUST finish. The greedy algorithm usually finishes well within time for these params.
            # C(19,8,5) is 335 blocks. Each block selection involves <k> iterations.
            # 335 * 8 * (combinations(8,4) lookups) is very small.
            pass
        
        # Start block with the first uncovered subset to ensure progress
        # Find first True in uncovered
        first_uncov_idx = np.argmax(uncovered)
        first_uncov_subset = all_t_subsets[first_uncov_idx]
        
        block = list(first_uncov_subset)
        
        # We need to mark these subsets as "tentatively covered" to avoid re-counting?
        # No, the greedy "gain" is based on *currently* uncovered.
        # As we add points, the set of uncovered subsets shrinks.
        # We should update `uncovered` only when we commit the block.
        # BUT, the gain of adding point x depends on the current state of `uncovered`.
        # If we don't update `uncovered` during block construction, the gain calculation is still valid 
        # because we are counting subsets that are *currently* uncovered. 
        # Wait, if I add x1, then x2. The subsets covered by x1 are still "uncovered" in my mental model 
        # until I commit. So when I calculate gain for x2, I might count subsets that x1 already "covered" 
        # in the real block construction.
        # This is a common issue. 
        # Standard Greedy: 
        # 1. Pick initial points.
        # 2. For each next point, calculate gain based on the *current* set of uncovered subsets in the global state.
        # 3. This leads to double counting if not careful.
        
        # Correct Standard Greedy for Covering:
        # Maintain `uncovered` global state.
        # When building a block, we pick points one by one.
        # The gain of adding x to current block B is the number of subsets S such that:
        #   S is uncovered (globally)
        #   S contains x
        #   S contains t-1 points from B
        #   S is NOT already fully contained in B (impossible since |B|<k and t<=k, but if t-1 points are in B, 
        #   the t-th must be x or another. If S is fully in B, it was covered when B reached size t).
        
        # Actually, if we don't update the global `uncovered` until the block is full, 
        # the "gain" calculation for the 5th point might include subsets that were already "covered" by the first 4 points 
        # of this *same* block? 
        # No, because a subset needs t points. If B has size < t-1, no subset is fully covered by B alone.
        # If B has size >= t, then subsets fully inside B are already covered by B. 
        # But we are adding x. The subsets we cover are those containing x and t-1 points from B.
        # If such a subset S was already covered by a *previous* block, `uncovered[idx(S)]` is False, so it's not counted.
        # If S is covered by the *current* block prefix B? 
        # S contains x, so S is not a subset of B. 
        # So, as long as we check the global `uncovered` array, we are correct.
        # The subsets covered by B (without x) are not counted because they don't contain x.
        
        # So the algorithm is:
        # 1. Start with empty block? Or start with some seeds?
        #    Starting with the lexicographically first uncovered subset ensures we cover that specific subset.
        #    Let block = list(first_uncov_subset).
        #    Note: These points are in the block.
        
        # 2. While len(block) < k:
        #    For each candidate x not in block:
        #       gain = 0
        #       for rest in combinations(block, t-1):
        #           s = tuple(sorted(rest + (x,)))
        #           if s in subset_to_idx:
        #               if uncovered[subset_to_idx[s]]:
        #                   gain += 1
        #    Pick x with max gain.
        #    Add x to block.
        
        # 3. Once block is full (size k):
        #    For all t-subsets of block:
        #       uncovered[subset_to_idx[s]] = False
        #    Append block to blocks.
        
        # This is correct and standard.
        
        # Optimization: 
        # Instead of iterating all x in range(v) for each step, we can just do it. v=19 is small.
        
        # Implementation details:
        # Precompute combinations of block? Block changes each iteration.
        
        # Let's refine the gain calculation speed.
        # We can precompute for each point x, and each combination of t-1 points, the index of the t-subset.
        # Since v=19, t=5, t-1=4.
        # Precompute `idx_cache[x][tuple_4]` -> index or -1.
        # tuple_4 is a tuple of 4 points.
        
        # Number of 4-subsets of 19 is C(19,4) = 3876.
        # For each x (19), and each 4-subset (3876), we check if x is in it (no, x is distinct).
        # The subset is x + 4 points.
        # So we need a map: (x, tuple4) -> index.
        
        # Let's build `gain_indices[x]` as a dict or array.
        # Actually, `subset_to_idx` is already built. 
        # `tuple(sorted(rest + (x,)))` takes a bit of time.
        # We can optimize by ensuring `rest` is sorted and `x` is placed correctly.
        
        # Given the small size, the Python loop might be fine. 
        # 335 blocks * 8 steps * 19 candidates * C(7,4)=35 combinations * lookup.
        # 335 * 8 * 19 * 35 ~ 1.7M operations. Python can do this in <1s.
        
        # Let's stick to the pure Python implementation with the dict lookup, it's robust.
        
        # Wait, I already initialized `uncovered` as NumPy.
        # I will use the NumPy array for the boolean check, but the loop in Python.
        
        # Re-initialize block selection
        if len(blocks) == 0 or True: # Just to keep structure
             pass
            
        # The loop logic above is inside the `while` block.
        # Let's re-write the inner loop cleanly.
        
        # --- Inner Loop for One Block ---
        block = list(first_uncov_subset)
        # It's possible that first_uncov_subset has duplicates? No, it's a set of points.
        
        while len(block) < k:
            best_gain = -1
            best_x = -1
            # Precompute combinations of current block of size t-1
            # Note: if len(block) < t-1, combinations is empty, gain is 0 for all?
            # If len(block) < t-1, we can't form a t-subset with just these points and one more.
            # We need t-1 points from block. So if len(block) < t-1, gain is 0.
            # However, we started with t points (from first_uncov_subset). So len(block) >= t.
            # So len(block) >= 5, t-1 = 4. Combinations exist.
            
            block_combos = list(itertools.combinations(block, t-1))
            
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in block_combos:
                    # Create the t-subset
                    # rest is a tuple of t-1 points
                    # We need to add x and sort
                    # Optimization: check if x is already in rest? No, x not in block, rest subset of block.
                    s = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx.get(s)
                    if idx is not None:
                        if uncovered[idx]:
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            # If no gain, break? Or add arbitrary? 
            # If best_gain == 0, adding x doesn't cover anything new. 
            # We should stop adding points if no gain? 
            # But we need a full block of size k? 
            # The problem asks for k-element blocks. 
            # If we stop early, the block size is < k. 
            # The contract says "k-element blocks". 
            # So we must fill it to k, even if gain is 0? 
            # Yes, "Each block is a list of k distinct integers".
            # So we pick the best x (even if gain 0) to fill the block.
            
            block.append(best_x)
            
        # Block is full
        # Update uncovered
        for s in itertools.combinations(block, t):
            idx = subset_to_idx.get(tuple(s))
            if idx is not None:
                uncovered[idx] = False
                
        blocks.append(tuple(block)) # Store as tuple for distinctness check later? 
        # "blocks must be distinct"
        
        # Check distinctness?
        # The greedy method might generate duplicate blocks if not careful?
        # Unlikely if we always start with the first uncovered subset.
        # But to be safe, we can track seen blocks.
        
    return [list(b) for b in blocks]

# Wait, the `while` loop structure was mixed in my thought process.
# I need to put the block construction inside the `while np.any(uncovered):` loop.
# And handle the deadline.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    uncovered = np.ones(n_subsets, dtype=bool)
    
    blocks = []
    seen_blocks = set()
    
    # Helper to check time
    def time_check():
        return time.time() > deadline
        
    while np.any(uncovered):
        if time_check():
            # We cannot return an invalid covering.
            # If we time out, we return what we have? No, it's invalid.
            # We must hope it finishes.
            # For v=19, it will.
            break
        
        # Find first uncovered
        first_uncov_idx = int(np.argmax(uncovered))
        first_uncov_subset = all_t_subsets[first_uncov_idx]
        
        block = list(first_uncov_subset)
        
        # Ensure block is unique?
        # Since we pick the first uncovered, and we mark covered, 
        # we might generate the same block again if it didn't cover anything new? 
        # But it covered first_uncov_subset, so it's different from any block that didn't cover it.
        # However, if we generated this exact block before, it would have covered first_uncov_subset already, 
        # so first_uncov_subset wouldn't be uncovered.
        # So blocks are naturally distinct.
        
        # Build the rest of the block
        while len(block) < k:
            if time_check():
                break # Emergency break, but might leave invalid state.
            
            best_gain = -1
            best_x = -1
            
            # Precompute combinations of current block
            # If len(block) < t-1, gain is 0 for all.
            if len(block) >= t - 1:
                block_combos = list(itertools.combinations(block, t-1))
            else:
                block_combos = []
            
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in block_combos:
                    # Construct t-subset
                    # Simple sort
                    s = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx.get(s)
                    if idx is not None:
                        if uncovered[idx]:
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            if best_x == -1:
                # No candidates? (Should not happen if k <= v and block < k)
                break
                
            block.append(best_x)
            
        # Finalize block
        if len(block) == k:
            # Check if block is new
            btup = tuple(sorted(block))
            if btup not in seen_blocks:
                seen_blocks.add(btup)
                blocks.append(list(btup))
                
                # Update uncovered
                for s in itertools.combinations(btup, t):
                    idx = subset_to_idx.get(s)
                    if idx is not None:
                        uncovered[idx] = False
                        
        else:
            # Incomplete block due to timeout? 
            # We should not add it.
            pass
            
    return blocks

# One more optimization: 
# The `subset_to_idx.get(s)` inside the inner loop is a bit slow.
# We can precompute the indices for all possible (rest, x) pairs?
# rest is a (t-1)-tuple. x is an int.
# Since v=19, t=5, t-1=4.
# There are C(19,4) = 3876 possible rests.
# For each rest, and each x not in rest, there is a unique t-subset.
# We can create a dictionary `idx_lookup` where key is (tuple(sorted(rest)), x) ?
# Or better: key is the tuple of the t-subset itself.
# We already have `subset_to_idx`.
# The bottleneck is `tuple(sorted(rest + (x,)))`.
# We can avoid sorting if we insert x into the sorted rest.
# rest is sorted.
# We can do:
# pos = bisect.bisect_left(rest, x)
# s = rest[:pos] + (x,) + rest[pos:]
# This is faster than sort.

import bisect

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    uncovered = np.ones(n_subsets, dtype=bool)
    
    blocks = []
    seen_blocks = set()
    
    def time_check():
        return time.time() > deadline
        
    while np.any(uncovered):
        if time_check():
            break
        
        first_uncov_idx = int(np.argmax(uncovered))
        first_uncov_subset = all_t_subsets[first_uncov_idx]
        
        block = list(first_uncov_subset)
        
        while len(block) < k:
            if time_check():
                break
            
            best_gain = -1
            best_x = -1
            
            if len(block) >= t - 1:
                block_combos = list(itertools.combinations(block, t-1))
            else:
                block_combos = []
            
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in block_combos:
                    # Fast construction of t-subset
                    pos = bisect.bisect_left(rest, x)
                    s = rest[:pos] + (x,) + rest[pos:]
                    idx = subset_to_idx.get(s)
                    if idx is not None:
                        if uncovered[idx]:
                            gain += 1
                            
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            if best_x == -1:
                break
                
            block.append(best_x)
            
        if len(block) == k:
            btup = tuple(sorted(block))
            if btup not in seen_blocks:
                seen_blocks.add(btup)
                blocks.append(list(btup))
                
                for s in itertools.combinations(btup, t):
                    idx = subset_to_idx.get(s)
                    if idx is not None:
                        uncovered[idx] = False
                        
    return blocks

# Final check on imports and structure.
# The code looks solid.
# The use of bisect avoids sorting overhead.
# The use of NumPy for the mask allows fast `np.any` and `np.argmax`.
# The logic is standard greedy covering.
