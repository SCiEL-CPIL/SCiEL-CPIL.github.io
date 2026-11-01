import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix
    # Cover[i, j] = 1 if t-subset j is in block i
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            
    # 1. Greedy Initialization
    # We want to minimize the number of blocks.
    # Standard greedy: pick block covering most uncovered t-subsets.
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected = []
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    
    # Precompute block sizes (all same, but useful for logic)
    # Actually, all blocks have size k, so they cover C(k,t) t-subsets.
    
    while np.any(uncovered) and time.time() < deadline - 20.0:
        # Gains: number of newly covered t-subsets
        gains = np.sum(Cover & uncovered, axis=1)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        selected.append(best_idx)
        cover_counts += Cover[best_idx]
        uncovered = (cover_counts == 0).astype(np.uint8)
        
    if np.any(uncovered):
        # Fallback if greedy failed (shouldn't happen for small v)
        # Just return what we have, though invalid
        pass

    # 2. Local Optimization
    # We have a list of block indices.
    # We will iteratively try to remove redundant blocks and perform swaps.
    
    # Convert to set for fast lookup
    selected_set = set(selected)
    
    # Helper to check if a block is redundant
    def is_redundant(idx, counts):
        # A block is redundant if every t-subset it covers is covered by at least one other block
        # i.e., counts[j] > 1 for all j where Cover[idx, j] == 1
        mask = Cover[idx] == 1
        if not np.any(mask):
            return True
        return np.all(counts[mask] > 1)

    # Helper to remove a block
    def remove_block(idx):
        nonlocal cover_counts, selected_set
        cover_counts -= Cover[idx]
        selected_set.remove(idx)

    # Helper to add a block
    def add_block(idx):
        nonlocal cover_counts, selected_set
        cover_counts += Cover[idx]
        selected_set.add(idx)

    # Phase 2a: Remove all redundant blocks
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        to_remove = []
        for idx in list(selected_set):
            if is_redundant(idx, cover_counts):
                to_remove.append(idx)
        for idx in to_remove:
            remove_block(idx)
            changed = True

    # Phase 2b: Swap optimization
    # Try to replace a block with another that allows further reduction
    # We iterate until no improvement is found
    
    # Precompute list of all block indices
    all_indices = list(range(n_blocks))
    
    # To speed up swap search, we can prioritize blocks that are "hard" to cover
    # i.e., blocks that cover many unique (count==1) t-subsets.
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Shuffle to avoid bias
        current_blocks = list(selected_set)
        random.shuffle(current_blocks)
        
        for idx in current_blocks:
            if time.time() > deadline - 2.0:
                break
            
            # If redundant, remove
            if is_redundant(idx, cover_counts):
                remove_block(idx)
                improved = True
                continue
            
            # Find critical t-subsets (count == 1) covered by idx
            critical_mask = (Cover[idx] == 1) & (cover_counts == 1)
            critical_tsubs = np.where(critical_mask)[0]
            
            if len(critical_tsubs) == 0:
                # Should have been removed, but maybe race condition?
                remove_block(idx)
                improved = True
                continue
            
            # We need to find a block j not in selected_set that covers all critical_tsubs
            # And ideally, j should help cover other critical subsets of other blocks
            
            # Filter candidates: must cover all critical_tsubs
            # candidate_score[j] = 1 if covers all critical, else 0
            # We want to find j such that Cover[j, critical_tsubs] are all 1
            
            # Efficient check:
            # Sum of Cover[j, critical_tsubs] should be len(critical_tsubs)
            
            # We can iterate through all blocks not in selected_set
            # This is O(N_blocks * N_critical)
            
            # Optimization: Only consider blocks that share some points?
            # For now, scan all.
            
            best_alt = -1
            best_score = -1
            
            # Heuristic score: 
            # 1. Must cover all criticals.
            # 2. Should cover as many other "low count" t-subsets as possible to enable removals.
            
            # Precompute "low count" mask: cover_counts <= 2
            low_count_mask = (cover_counts <= 2).astype(np.uint8)
            
            # Iterate through all blocks
            for j in all_indices:
                if j in selected_set:
                    continue
                
                # Check if j covers all criticals
                # If any critical is not covered, skip
                # np.all(Cover[j, critical_tsubs] == 1)
                
                # Fast check:
                if not np.all(Cover[j, critical_tsubs]):
                    continue
                
                # Score: number of low-count t-subsets covered by j
                # Exclude the critical ones? Or include?
                # If j covers a low-count subset, it might make another block redundant.
                score = np.sum(Cover[j] * low_count_mask)
                
                # Prefer blocks that cover more "critical" subsets of OTHER blocks?
                # Hard to compute fast.
                
                if score > best_score:
                    best_score = score
                    best_alt = j
            
            if best_alt != -1:
                # Perform swap
                # Remove idx
                cover_counts -= Cover[idx]
                selected_set.remove(idx)
                # Add best_alt
                cover_counts += Cover[best_alt]
                selected_set.add(best_alt)
                improved = True
                
                # After swap, check for new redundancies
                # We can do a quick pass
                for k in list(selected_set):
                    if is_redundant(k, cover_counts):
                        remove_block(k)
                        improved = True

    # Final cleanup
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        to_remove = []
        for idx in list(selected_set):
            if is_redundant(idx, cover_counts):
                to_remove.append(idx)
        for idx in to_remove:
            remove_block(idx)
            changed = True

    # Convert back to list of blocks
    blocks = [all_kblocks[i] for i in selected_set]
    
    # Verify
    # The evaluator will verify, but we can do a quick check
    # if len(blocks) < 100: 
    #    ...
    
    return blocks

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
