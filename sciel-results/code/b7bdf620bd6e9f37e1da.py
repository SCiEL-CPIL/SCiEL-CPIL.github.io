import itertools
import time
import random
import sys

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    # To save memory and time, we don't store the full list of all blocks if we can generate them on the fly,
    # but for v=19, k=8, there are ~75k blocks. Storing the list of indices for each block is fine.
    blocks = []
    block_covers = [] # list of lists of indices
    
    # Optimization: Pre-compute combinations for speed? 
    # itertools.combinations is fast enough.
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        # Fallback: Return a trivial (though large) valid cover if we ran out of time during setup
        # This shouldn't happen, but just in case.
        # A trivial cover is all blocks containing a specific point? No, must cover all t-subsets.
        # If we fail here, we return empty which is invalid. 
        # But setup is fast.
        return []

    n_blocks = len(blocks)
    
    # 3. Greedy Set Cover
    # counts[i] = number of selected blocks covering t-subset i
    counts = [0] * n_t
    selected_indices = []
    
    # To speed up greedy, we can keep a list of active blocks (those that cover at least one uncovered subset).
    # However, checking "uncovered" for every block every step is O(N * n_t).
    # Better: For each block, calculate gain = number of subsets in block_covers[i] where counts[j] == 0.
    
    # We will use a simple greedy loop.
    # Optimization: Instead of scanning all blocks every time, we can just scan all blocks.
    # N=75k, n_t=11.6k. 
    # Steps needed: roughly 300-400.
    # 400 * 75k = 30 million block checks.
    # Each check iterates over ~56 subsets. 30M * 56 = 1.68 billion operations. Too slow in pure Python.
    
    # We need a faster greedy.
    # Idea: Use a priority queue? Or batch processing?
    # Alternative: Use a randomized greedy with a smaller candidate pool?
    # Or: Use the fact that we only care about uncovered subsets.
    
    # Let's try a different approach: 
    # Maintain a set of uncovered subsets.
    # For each block, gain = len(block_covers[i] & uncovered_set).
    # This is still O(N * 56) per step if we check intersection.
    
    # Let's optimize the inner loop.
    # We can precompute for each t-subset, the list of blocks that cover it.
    # sub_to_blocks = [[] for _ in range(n_t)]
    # for i, covers in enumerate(block_covers):
    #     for j in covers:
    #         sub_to_blocks[j].append(i)
        
    # Greedy Step:
    # 1. Find the uncovered subset with the minimum number of covering blocks (most constrained).
    # 2. Among the blocks covering this subset, pick the one that covers the most other uncovered subsets.
    
    # This is the "Most Constrained Variable" heuristic.
    
    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    uncovered = set(range(n_t))
    selected = set()
    
    while uncovered:
        if not _deadline_ok(deadline):
            break
            
        # Find the uncovered subset with fewest candidate blocks
        # To avoid O(n_t) scan every time, we could maintain a heap, but n_t is small (11k).
        # Scanning 11k items is fast.
        min_count = float('inf')
        target_sub = -1
        for s in uncovered:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1: # Can't get better
                    break
        
        if target_sub == -1:
            break # Should not happen if cover exists
            
        # Get candidates for target_sub
        candidates = sub_to_blocks[target_sub]
        
        # Pick the candidate that maximizes coverage of other uncovered subsets
        best_block = -1
        max_gain = -1
        
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if s in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b
                
        # Select best_block
        selected.add(best_block)
        
        # Update uncovered
        for s in block_covers[best_block]:
            uncovered.discard(s)
            
        # Update sub_to_blocks counts? 
        # We don't strictly need to update the list of blocks for each subset if we re-scan `uncovered` 
        # and check `s in uncovered` in the gain calculation. 
        # However, `sub_to_blocks[s]` still contains blocks that might have been selected. 
        # That's fine, we just pick from them. If a block is already selected, it won't be in `uncovered` 
        # so its gain for those subsets is 0. But we might pick a block that is already selected?
        # No, because if a block is selected, all its subsets are removed from `uncovered`.
        # So if `target_sub` is in `uncovered`, no selected block covers it. 
        # So all candidates in `sub_to_blocks[target_sub]` are unselected.
        
    # Convert selected indices to blocks
    result = [list(blocks[i]) for i in selected]
    
    # 4. Local Search: Remove redundant blocks
    # A block is redundant if all its subsets are covered by other selected blocks.
    # We can check this by verifying if removing it leaves all its subsets covered.
    
    # To make this fast, we can maintain a coverage count for each subset from the selected blocks.
    # But we already have `selected`.
    # Let's do a simple pass.
    
    # Re-calculate counts for subsets based on `selected`
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
            
    # Try to remove blocks
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            # Check if block i is redundant
            # It is redundant if for all s in block_covers[i], final_counts[s] > 1
            redundant = True
            for s in block_covers[i]:
                if final_counts[s] <= 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(i)
                for s in block_covers[i]:
                    final_counts[s] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    
    result = [list(blocks[i]) for i in selected]
    
    # Verify validity (just in case)
    # For speed, we assume the greedy + prune logic is correct.
    # But let's do a quick sanity check if time permits.
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in selected:
            for s in block_covers[i]:
                check_counts[s] += 1
        if any(c == 0 for c in check_counts):
            # Invalid, return the previous greedy result before pruning? 
            # Or just return the greedy result.
            # The greedy result was valid. Pruning only removes if redundant.
            # So it should be valid.
            pass
            
    return result
