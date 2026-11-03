import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    # Set a conservative deadline to ensure we return before the 150s kill
    # The evaluator kills at 150s. We want to finish by 110s to be safe.
    deadline = start_time + 110
    
    all_points = list(range(v))
    
    # Precompute all t-subsets of the universe
    # For v=19, t=4, there are 3876 subsets.
    all_t_subsets = set(itertools.combinations(all_points, t))
    
    # Precompute for every possible k-block, the list of t-subsets it covers.
    # There are C(19, 7) = 50388 blocks.
    # Each has C(7, 4) = 35 t-subsets.
    # Total precomputation: ~1.7 million tuple creations. This is fast.
    block_to_subsets = {}
    for block in itertools.combinations(all_points, k):
        # Store as a tuple of tuples for hashability and fast iteration
        block_to_subsets[block] = tuple(itertools.combinations(block, t))
        
    # We will maintain:
    # 1. `blocks`: list of current blocks (tuples)
    # 2. `cover_count`: dict mapping t-subset tuple -> count of blocks covering it
    # 3. `uncovered`: set of t-subsets with cover_count 0
    
    blocks = []
    cover_count = {s: 0 for s in all_t_subsets}
    uncovered = set(all_t_subsets)
    
    def add_block(block):
        blocks.append(block)
        for s in block_to_subsets[block]:
            cover_count[s] += 1
            if cover_count[s] == 1:
                uncovered.remove(s)
                
    def remove_block(block):
        # Remove from list (O(N))
        blocks.remove(block)
        for s in block_to_subsets[block]:
            cover_count[s] -= 1
            if cover_count[s] == 0:
                uncovered.add(s)

    # --- Phase 1: Greedy Construction ---
    # We want to build a valid cover quickly.
    # Randomized greedy: pick a random uncovered subset, then find the best block containing it?
    # Or: Pick a random point, find the best block containing it?
    # Standard greedy: Pick the block that covers the most *currently* uncovered subsets.
    # This is O(N_blocks * 35) per step. N_blocks = 50388. 50388 * 35 = 1.7M checks per block added.
    # If we add ~150 blocks, that's 250M checks. Too slow in Python.
    
    # Better Greedy: 
    # 1. Pick a random uncovered t-subset `u`.
    # 2. Find all blocks that contain `u`.
    # 3. Among those, pick the one that covers the most *other* uncovered subsets.
    
    # To make this fast, we can index blocks by their t-subsets?
    # Or, since v is small, we can just iterate through all blocks and filter?
    # Let's try a simpler greedy first:
    # While uncovered:
    #   Pick random uncovered subset u.
    #   Find a random block containing u. (We can precompute `subsets_to_blocks` mapping)
    #   Evaluate gain of that block.
    #   Maybe try a few candidates?
    
    # Let's precompute `subset_to_blocks`
    subset_to_blocks = {s: [] for s in all_t_subsets}
    for block, subs in block_to_subsets.items():
        for s in subs:
            subset_to_blocks[s].append(block)
            
    # Now Greedy:
    while uncovered and time.time() < deadline:
        # Pick a random uncovered subset
        target = random.choice(list(uncovered))
        
        # Get candidate blocks that cover this target
        candidates = subset_to_blocks[target]
        if not candidates:
            # Should not happen if target is a valid subset of v points and k >= t
            continue
            
        # Evaluate candidates
        best_block = None
        best_gain = -1
        
        # To speed up, don't evaluate ALL candidates if there are many.
        # For t=4, k=7, how many blocks contain a specific 4-subset?
        # We need to choose 3 more points from the remaining 19-4=15 points.
        # C(15, 3) = 455 candidates. This is small enough to evaluate all.
        
        for b in candidates:
            gain = 0
            for s in block_to_subsets[b]:
                if s in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
                
        if best_block is not None:
            add_block(best_block)
            
    # If we still have uncovered items (shouldn't happen if greedy worked), 
    # but if we ran out of time, we must return what we have? 
    # NO, the result must be VALID. 
    # The greedy loop `while uncovered` ensures validity upon exit.
    # If we exit due to time, we might have uncovered items.
    # We must force validity.
    
    # --- Phase 1.5: Force Validity ---
    # If we didn't finish greedy, we need to cover remaining.
    # This is unlikely with 110s budget, but let's be safe.
    if uncovered:
        # Use a faster, less optimal method to cover remaining
        # Just pick random blocks until covered.
        while uncovered:
            target = random.choice(list(uncovered))
            # Pick a random block containing target
            b = random.choice(subset_to_blocks[target])
            if b not in blocks: # Avoid duplicates
                add_block(b)
            # If b is already in blocks, it didn't cover target? 
            # No, if b contains target, it covers it. 
            # If it's already in blocks, target would have been covered.
            # So this implies b was not in blocks.
            # Wait, if b is in blocks, cover_count[target] > 0, so target not in uncovered.
            # So b is never in blocks if target is uncovered.
            
    # Now we have a valid covering.
    # --- Phase 2: Local Search / Optimization ---
    # Goal: Minimize number of blocks.
    # Strategy: 
    # 1. Remove redundant blocks.
    # 2. Try to replace a block with another block that allows removal of others? 
    #    Simpler: Try "1-0" moves (remove block if redundant).
    #    Try "1-1" moves (replace block A with block B if B covers all unique subsets of A).
    #    Try "2-1" or "1-2"? Too complex for time.
    
    # Optimization: 
    # Iterate through blocks, check if redundant.
    # If redundant, remove.
    # Repeat until no more redundant blocks found.
    
    # To speed up redundancy check:
    # A block is redundant if all its t-subsets have cover_count > 1.
    
    redundant_found = True
    while redundant_found and time.time() < deadline:
        redundant_found = False
        # Shuffle blocks to avoid bias? 
        # Or just iterate.
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            is_redundant = True
            for s in block_to_subsets[b]:
                if cover_count[s] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                remove_block(b)
                redundant_found = True
                break # Restart loop after removal
                
    # If time remains, try swaps (1-1 moves) to enable further reductions
    if time.time() < deadline - 5: # Leave 5s for final safety
        # We can try to improve by swapping blocks.
        # A swap is valid if the new block covers all subsets that were UNIQUE to the old block.
        # This might not reduce count immediately, but might enable future reductions.
        # Given time constraints, simple redundancy removal is often the biggest win.
        # Let's stick to redundancy removal as it's robust.
        pass

    return [list(b) for b in blocks]
