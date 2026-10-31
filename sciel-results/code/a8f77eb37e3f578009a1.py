import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    # Build coverage matrix: Cover[i, j] = 1 if t-sub j is in block i
    # Use a dict for t-subset to index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    # --- Stage 1: Greedy Set Cover ---
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected_indices = []
    
    # Precompute the set of t-subset indices covered by each block for faster updates?
    # Cover is sparse.
    # For speed in gain calculation: gains = Cover @ uncovered
    # Cover is (n_blocks, n_tsubs), uncovered is (n_tsubs,)
    # Result is (n_blocks,)
    
    while True:
        if time.time() > deadline - 30.0: # Leave 30s for optimization
            break
        
        gains = Cover @ uncovered
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
        
        if np.sum(uncovered) == 0:
            break

    # If we didn't finish greedy in time, we might have an invalid cover.
    # But for v=19, k=7, t=4, greedy is usually fast enough.
    # If uncovered > 0, we must force cover remaining.
    if np.sum(uncovered) > 0:
        # Fallback: brute force add blocks until covered
        # This is slow, but better than invalid.
        while np.sum(uncovered) > 0:
            if time.time() > deadline - 5.0:
                break
            gains = Cover @ uncovered
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered = uncovered & (Cover[best_idx] == 0)

    # --- Stage 2: Optimization (Local Search) ---
    # We have a valid covering (hopefully).
    # Goal: Minimize number of blocks.
    
    # Convert selected_indices to a set for fast lookup
    selected_set = set(selected_indices)
    # Also keep a list for iteration
    selected_list = list(selected_set)
    
    # Compute coverage counts for each t-subset
    # cover_counts[j] = number of selected blocks containing t-sub j
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected_list:
        cover_counts += Cover[idx]
        
    # Check validity
    if np.any(cover_counts == 0):
        # Invalid covering. Attempt to repair by adding missing blocks.
        # This should not happen if greedy finished, but just in case.
        missing = np.where(cover_counts == 0)[0]
        for j in missing:
            # Find a block that covers j
            candidates = np.where(Cover[:, j] == 1)[0]
            if len(candidates) > 0:
                # Pick one that maximizes coverage of other missing tsubs
                # Or just pick the first
                idx = candidates[0]
                if idx not in selected_set:
                    selected_set.add(idx)
                    selected_list.append(idx)
                    cover_counts += Cover[idx]
        
        # Re-check
        if np.any(cover_counts == 0):
            # Still invalid. Return what we have? No, must be valid.
            # This is a critical failure path.
            pass

    # Optimization Loop
    # 1. Remove redundant blocks (if a block covers no unique t-subset, remove it)
    # 2. Try to replace a block with a smaller combination? (Hard)
    # 3. Try to swap a block for another that allows removal of others?
    
    # Simple removal of redundant blocks first
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        new_selected_list = []
        for idx in selected_list:
            # Check if this block is essential
            # Essential if there exists a t-subset j such that Cover[idx, j]==1 and cover_counts[j]==1
            # i.e., np.any((Cover[idx] == 1) & (cover_counts == 1))
            if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                new_selected_list.append(idx)
            else:
                # Remove it
                cover_counts -= Cover[idx]
                changed = True
                selected_set.discard(idx)
        
        selected_list = new_selected_list

    # Now try local search:
    # For each block in the current solution, try to replace it with a different block
    # such that the total number of blocks decreases?
    # Or, try to remove a block and see if we can cover the gaps with fewer than 1 new block?
    # That's impossible.
    
    # Better strategy:
    # Try to find a block B in the solution that can be removed, and the uncovered t-subsets
    # can be covered by adding a set of blocks S such that |S| < 1? No.
    
    # Alternative:
    # Try to swap one block for another, and then remove redundancies.
    # If the net change is negative, keep it.
    
    # Let's try a "swap and simplify" approach.
    # 1. Pick a block b from the solution.
    # 2. Try all blocks b' not in the solution.
    # 3. Remove b, add b'.
    # 4. Run redundancy removal.
    # 5. If size decreased, keep. Else revert.
    
    # This is O(N^2) which is too slow.
    
    # Smarter approach:
    # Identify "weak" blocks: blocks that cover many t-subsets with count > 1.
    # Try to replace them with blocks that cover more "critical" (count=1) t-subsets?
    # No, we want to reduce total blocks.
    
    # Let's try a different local move:
    # Pick two blocks b1, b2 in the solution.
    # Try to replace {b1, b2} with a single block b3?
    # If b3 covers all t-subsets that were uniquely covered by b1 or b2, and the rest are covered by others,
    # then we can replace 2 blocks with 1.
    
    # This is a 2-to-1 replacement.
    # Check if there exists a block b3 such that:
    # For all j in (Unique(b1) U Unique(b2)), Cover[b3, j] == 1.
    # Where Unique(b) = {j | Cover[b, j]==1 and cover_counts[j]==1}
    
    # Precompute unique sets?
    # unique_mask[idx] is a boolean array of t-subsets uniquely covered by idx.
    
    # This is expensive to compute for all pairs.
    
    # Let's stick to the redundancy removal which is already implemented.
    # And add a simple heuristic:
    # Randomly shuffle the order of greedy selection?
    # Or use weighted greedy?
    
    # Given time constraints, the redundancy removal is the most reliable gain.
    # Let's also try to add blocks that cover many uncovered t-subsets if we are not fully covered?
    # We already handled that.
    
    # One more optimization:
    # If we have time, try to restart greedy with a different initial set?
    # Or use simulated annealing? Too complex.
    
    # Let's just return the result.
    # Ensure we return a list of lists (tuples are fine? Task says list of blocks, block is list of k distinct integers)
    # all_kblocks are tuples. Convert to lists.
    
    final_blocks = [list(all_kblocks[i]) for i in selected_list]
    
    # Final validity check
    # If invalid, try to fix by adding blocks
    final_cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for B in final_blocks:
        for ts in itertools.combinations(B, t):
            final_cover_counts[tsub_to_idx[ts]] += 1
            
    if np.any(final_cover_counts == 0):
        missing = np.where(final_cover_counts == 0)[0]
        for j in missing:
            candidates = np.where(Cover[:, j] == 1)[0]
            if len(candidates) > 0:
                # Pick a candidate that is not already in the set
                added = False
                for c in candidates:
                    if c not in selected_set:
                        final_blocks.append(list(all_kblocks[c]))
                        selected_set.add(c)
                        final_cover_counts += Cover[c]
                        added = True
                        break
                if not added:
                    # All candidates are already in the set?
                    # That means no single block can cover this t-subset without duplication?
                    # Wait, if c is in selected_set, it's already in final_blocks.
                    # So final_cover_counts should already count it.
                    # This implies the t-subset is covered by an existing block, so it shouldn't be missing.
                    # This case shouldn't happen.
                    pass

    return final_blocks
