import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using NumPy-accelerated greedy and local search.
    """
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, store the indices of the t-subsets it covers
    # Shape: (num_blocks, C(k, t))
    num_t_in_block = int(len(list(itertools.combinations(range(k), t))))
    block_t_indices = np.zeros((num_blocks, num_t_in_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        # Map the t-subsets of this block to global indices
        for j, ts in enumerate(itertools.combinations(block, t)):
             block_t_indices[i, j] = t_sub_to_idx[ts]
             
    # Initialize uncovered array
    uncovered = np.ones(num_t_subs, dtype=np.int8)
    
    # Greedy Construction
    blocks_used_indices = []
    used_block_mask = np.zeros(num_blocks, dtype=bool)
    
    while uncovered.sum() > 0:
        # Calculate gains for all unused blocks
        # gains[i] = number of uncovered t-subsets covered by block i
        gains = uncovered[block_t_indices].sum(axis=1)
        
        # Mask out used blocks
        gains[used_block_mask] = -1
        
        # Find best block
        best_idx = np.argmax(gains)
        
        if gains[best_idx] <= 0:
            break
            
        blocks_used_indices.append(best_idx)
        used_block_mask[best_idx] = True
        
        # Update uncovered
        uncovered[block_t_indices[best_idx]] = 0
        
    # Local Search: Greedy Deletion
    # We have blocks_used_indices. We want to remove redundant blocks.
    # Coverage count for each t-subset
    t_sub_count = np.zeros(num_t_subs, dtype=np.int32)
    
    # Initialize counts
    for idx in blocks_used_indices:
        t_sub_count[block_t_indices[idx]] += 1
        
    # Deletion Phase
    # We iterate backwards through the list of block indices.
    # If a block is redundant (all its t-subsets have count > 1), we remove it.
    # We do this in a single pass? No, removing one might make another redundant.
    # But if we iterate backwards and remove, we might miss blocks that become redundant 
    # due to the removal of a *later* block? 
    # Actually, if we remove a block, counts decrease. A block that was NOT redundant 
    # (had a count 1) might become redundant? No, counts decrease, so if it had a 1, 
    # it stays 1 or becomes 0 (invalid). It cannot become "more redundant". 
    # Wait. Redundant means MIN count > 1. 
    # If I remove block B, counts of B's t-subsets decrease.
    # Another block A might have a t-subset that was covered by B and A. 
    # If count was 2, it becomes 1. A becomes critical. 
    # So removing a block can make OTHERS critical, but it cannot make a non-redundant block redundant.
    # It can only make redundant blocks... well, if a block was redundant (min count > 1), 
    # and we remove another block that covers some of its t-subsets, the counts drop.
    # If they drop to 1, it is NO LONGER redundant.
    # So, if we remove a block, we might "un-redundantize" other blocks.
    # Therefore, a single pass is NOT sufficient if we want to maximize deletions.
    # We must repeat until no more blocks can be removed.
    
    changed = True
    while changed:
        changed = False
        # Iterate backwards to allow safe popping
        for i in range(len(blocks_used_indices) - 1, -1, -1):
            idx = blocks_used_indices[i]
            # Check if redundant
            # np.min(t_sub_count[block_t_indices[idx]]) > 1
            if np.min(t_sub_count[block_t_indices[idx]]) > 1:
                # Remove
                t_sub_count[block_t_indices[idx]] -= 1
                blocks_used_indices.pop(i)
                changed = True
                # Do NOT break. Continue checking previous blocks.
                # The removal of this block might have made a previous block non-redundant?
                # No, previous block's counts might have decreased.
                # If previous block was redundant (min > 1), it might now have min == 1.
                # So it is no longer redundant. We should NOT remove it.
                # So continuing is safe. We just won't remove it if it's not redundant anymore.
                # But what if the removal of THIS block makes a PREVIOUS block redundant?
                # No, counts only decrease. If a block was NOT redundant (min <= 1), 
                # decreasing counts keeps min <= 1. It stays non-redundant.
                # So removing a block can only turn redundant blocks into non-redundant ones.
                # It cannot turn non-redundant into redundant.
                # So if we scan from end to start, and remove a block, we don't need to 
                # re-check the blocks we've already passed (indices < i). 
                # Wait. If I remove block at index i, blocks at indices < i are "previous".
                # Their counts decreased. If they were redundant, they might not be anymore.
                # So if I already checked index i-1 and decided it was redundant and removed it,
                # that's fine.
                # But if I checked index i-1 and decided it was NOT redundant (min <= 1),
                # and then I remove block i, the count for i-1 decreases further.
                # It remains non-redundant.
                # So a single pass from end to start is actually sufficient to find ALL removable blocks?
                # Let's trace:
                # Blocks: A, B, C.
                # Suppose C is redundant. Remove C.
                # A and B counts decrease.
                # Suppose A was redundant (min=2). Now min=1. A is not redundant.
                # Suppose B was redundant (min=2). Now min=1. B is not redundant.
                # So removing C "saved" A and B.
                # If we had removed A first (if we went forward), we would have updated counts, 
                # then checked B.
                # The order matters for WHICH blocks survive, but we want to maximize deletions.
                # Is it possible that removing C first prevents us from removing A and B, 
                # whereas removing A first allows removing B and C?
                # This is a set cover problem variant. Greedy deletion is not guaranteed optimal.
                # However, repeating the pass until no change is the standard approach.
                # With the "no break" logic, one pass removes all blocks that are redundant 
                # *at the time of checking*.
                # If removing C makes A non-redundant, and we check A *after* C (i.e. A is before C in list, 
                # and we go end-to-start), we check C first. Remove C. Then check A. 
                # A is now non-redundant. We skip A.
                # If we went start-to-end: Check A. A is redundant. Remove A. 
                # Check B. B is redundant. Remove B. 
                # Check C. C is redundant. Remove C.
                # So start-to-end might remove MORE blocks in this specific scenario?
                # Let's re-evaluate.
                # If A, B, C are all redundant initially.
                # End-to-start:
                # Check C. Redundant. Remove C. Counts drop.
                # Check B. Was redundant. Counts dropped. Maybe still redundant? Maybe not.
                # If B becomes non-redundant, we keep B.
                # Check A. Same.
                # Start-to-end:
                # Check A. Redundant. Remove A. Counts drop.
                # Check B. Was redundant. Counts dropped. Maybe non-redundant?
                # If B becomes non-redundant, we keep B.
                # Check C. Same.
                # It seems symmetric.
                # However, if removing A makes B redundant? No, counts drop.
                # Removing a block only decreases counts.
                # Redundancy requires min count > 1.
                # Decreasing counts makes it harder to be redundant.
                # So removing a block can NEVER make another block redundant.
                # It can only make a redundant block non-redundant.
                # Therefore, the set of redundant blocks can only SHRINK as we remove blocks.
                # So if a block is redundant at the start of the pass, it is a candidate.
                # If we remove another block first, it might no longer be redundant.
                # So we should prioritize removing blocks that are "most redundant"?
                # Or just any.
                # Since the set of redundant blocks shrinks, a single pass that removes ALL currently redundant blocks 
                # (if we could do that atomically) would be good.
                # But we can't remove all at once because removing one affects others.
                # However, since removing one only makes others LESS likely to be redundant, 
                # if we identify a set of blocks that are ALL redundant, and we remove them ALL,
                # are they still redundant?
                # If A and B are redundant, and they share t-subsets.
                # Count(t) = 2 (covered by A and B).
                # If we remove A, Count(t) = 1. B is no longer redundant for t.
                # So we can't remove both if they are the only coverers.
                # But if Count(t) = 3 (A, B, C), removing A makes Count(t)=2. B is still redundant for t.
                # So, we can remove multiple blocks in one pass IF we are careful.
                # The safest and simplest is:
                # Loop:
                #   Find all blocks that are currently redundant.
                #   If none, break.
                #   Remove them all? 
                #   If we remove them all, we might invalidate some.
                #   Better: Remove ONE, update, repeat.
                #   To speed up, we can remove all blocks that are redundant AND don't share 
                #   critical t-subsets? Too complex.
                #   
                #   Let's stick to the "remove one, continue" but do it in a while loop.
                #   The current code does `break` and restarts. This is O(N^2).
                #   My proposed code does NOT break. It continues.
                #   Is that correct?
                #   If I remove block at index i, and continue to i-1.
                #   Block i-1 was checked? No, I haven't checked i-1 yet in this pass.
                #   So I check i-1 with the NEW counts.
                #   This is correct.
                #   Do I need to re-check blocks > i?
                #   I already checked them. They were either removed or kept.
                #   If kept, they were not redundant at the time.
                #   Removing block i (which is "later" in the list) decreases counts.
                #   So a block j > i that was kept (not redundant) remains not redundant.
                #   So no need to re-check.
                #   So a single pass from end to start, removing any redundant block found, 
                #   and continuing, is a valid heuristic that removes a maximal subset of redundant blocks 
                #   in one pass?
                #   Actually, it removes a set of blocks.
                #   Then we repeat the pass to catch blocks that might have been redundant 
                #   but became non-redundant? No, we want to remove redundant ones.
                #   Wait, if a block was redundant, and we removed another block, it might become non-redundant.
                #   So we might "miss" removing it if we checked it BEFORE the removal that saved it?
                #   No, if we check it AFTER the removal, we see it's non-redundant and skip it. Correct.
                #   If we check it BEFORE the removal, we see it's redundant and remove it. 
                #   Is that bad?
                #   If we remove it, we might have "wasted" a removal opportunity on a block that was "essential" 
                #   for keeping others redundant? No, removing a redundant block is always safe.
                #   The only risk is that by removing it, we prevent removing OTHERS?
                #   No, removing a block only decreases counts. It doesn't increase them.
                #   So it doesn't make others MORE redundant.
                #   So removing a redundant block is always "good" or neutral.
                #   So we should remove as many as possible.
                #   The "end to start, no break" strategy removes a set of blocks.
                #   Are there blocks that are redundant but not removed in this pass?
                #   Yes, if they were "saved" by the removal of a block they were checked after?
                #   No, if they are checked after, they see the updated counts.
                #   If they are checked before, they are removed.
                #   So every block that is redundant at the moment it is checked gets removed.
                #   This seems like a good greedy strategy.
                #   We repeat this pass until no more blocks are removed.

                pass # End of thought process

    # Convert indices to blocks
    blocks = [list(all_blocks[idx]) for idx in blocks_used_indices]
    
    return blocks
