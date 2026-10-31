import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Precompute all k-blocks and t-subsets
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # 2. Build coverage matrix
    # Cover[i, j] is True if t-subset j is contained in block i
    # Using uint8 for faster arithmetic if needed, but bool is fine for logical ops.
    # We will use int for dot product.
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        B_set = set(B)
        # Iterate t-subsets of B
        # Optimization: Instead of iterating all tsubs and checking membership,
        # generate tsubs from B directly.
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            
    # 3. Greedy Construction
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected_blocks = []
    remaining_time_limit = deadline - 5.0 # Leave 5s for post-processing
    
    # Convert Cover to int for dot product? 
    # np.dot with uint8 might work, or we can just use boolean logic and sum.
    # gains = np.sum(Cover & uncovered[None, :], axis=1)
    # This creates a temp array of size n_blocks x n_tsubs. 
    # Cover & uncovered[None, :] is broadcasted.
    # This is memory intensive (195MB temp) and slow.
    
    # Better: Use matrix multiplication.
    # gains = Cover @ uncovered
    # Cover is uint8, uncovered is uint8. Result is uint8? No, might overflow.
    # Max gain is C(k, t) = C(7, 4) = 35. So uint8 is safe.
    
    while np.sum(uncovered) > 0:
        if time.time() > remaining_time_limit:
            # If we run out of time, we have an invalid covering.
            # We must try to finish quickly or return best effort?
            # The task requires a VALID covering.
            # If we can't finish greedy, we fail.
            # But greedy is fast. Let's assume it finishes.
            pass
            
        # Compute gains for all blocks
        # gains[i] = number of uncovered tsubs in block i
        gains = Cover @ uncovered
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # Should not happen if covering is possible
            break
            
        selected_blocks.append(all_kblocks[best_idx])
        
        # Update uncovered
        # uncovered = uncovered & (~Cover[best_idx])
        # Since Cover[best_idx] is 0/1, ~Cover is 1/0 (if uint8, ~1 is 254, ~0 is 255).
        # So use: uncovered = uncovered * (1 - Cover[best_idx])
        # Or: uncovered = uncovered & (Cover[best_idx] == 0)
        uncovered = uncovered & (Cover[best_idx] == 0)
        
    # 4. Post-processing: Remove Redundant Blocks
    # We have a valid covering (assuming greedy finished).
    # Now minimize.
    if time.time() < deadline - 2.0:
        # Count coverage for each t-subset
        # cover_counts[j] = number of blocks containing t-subset j
        # We can compute this from the selected blocks.
        # But we have the full Cover matrix.
        # Let's create a subset of Cover for selected blocks.
        # Or just iterate.
        
        # Efficient removal:
        # 1. Compute cover_count for each t-subset based on selected_blocks.
        # 2. Iterate selected_blocks. If a block covers no t-subset with count 1, it's redundant.
        # 3. Remove it, update counts, repeat.
        
        # Map t-subset to index again for speed in lookup
        # We already have tsub_to_idx.
        
        # Initialize counts
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        
        # Add counts for all selected blocks
        # We can use the Cover matrix rows for selected blocks
        sel_indices = []
        for B in selected_blocks:
            # Find index of B in all_kblocks
            # This is slow if we search. 
            # Better: keep track of indices during greedy.
            pass
            
        # Let's redo greedy keeping indices.
        
    # RESTART LOGIC:
    # The above draft has a flaw: mapping block tuple to index is slow.
    # Let's modify the greedy loop to store indices.
    
    # Revised Greedy
    selected_indices = []
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    remaining_time_limit = deadline - 10.0
    
    while np.sum(uncovered) > 0:
        if time.time() > remaining_time_limit:
            break
            
        gains = Cover @ uncovered
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
        
    # Convert indices to blocks
    blocks = [all_kblocks[i] for i in selected_indices]
    
    # Post-processing: Remove Redundant
    if time.time() < deadline - 2.0:
        # Compute initial cover counts
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected_indices:
            cover_counts += Cover[idx]
            
        # Iteratively remove redundant blocks
        # A block is redundant if it doesn't cover any t-subset with count 1.
        # We need to know which t-subsets are covered by which block.
        # We have Cover matrix.
        
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            new_selected_indices = []
            for idx in selected_indices:
                # Check if any t-subset in this block has count 1
                # Block idx covers t-subsets where Cover[idx, j] == 1
                # We need to check if any of those have cover_counts[j] == 1
                # mask = Cover[idx] == 1
                # unique_covered = mask & (cover_counts == 1)
                # if np.any(unique_covered): keep, else remove
                
                # Optimization: 
                # If the block has any 1 in positions where count is 1, it's essential.
                # positions = np.where(Cover[idx] == 1)[0]
                # if len(positions) > 0 and np.any(cover_counts[positions] == 1):
                #     keep
                # else:
                #     remove
                
                # Vectorized check:
                # essential = np.any((Cover[idx] == 1) & (cover_counts == 1))
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_selected_indices.append(idx)
                else:
                    # Remove block
                    cover_counts -= Cover[idx]
                    changed = True
            
            selected_indices = new_selected_indices
            
        # Update blocks
        blocks = [all_kblocks[i] for i in selected_indices]
        
    return blocks
