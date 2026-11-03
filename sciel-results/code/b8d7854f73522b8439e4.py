"""Vectorized greedy set cover with redundancy removal for C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(v, k, t):
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build incidence matrix: M[b, i] = 1 if block b covers t-subset i
    # Using uint8 to save memory (195 MB for v=19, k=7, t=4)
    M = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for b, blk in enumerate(all_blocks):
        for ts in itertools.combinations(blk, t):
            idx = tsub_to_idx[ts]
            M[b, idx] = 1
            
    # uncovered: 1 if t-subset is not yet covered
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    
    selected_blocks = []
    used = np.zeros(n_blocks, dtype=bool)
    
    # Greedy Phase
    while uncovered.any():
        # Score each block: number of uncovered t-subsets it covers
        # scores[b] = sum(M[b, :] * uncovered)
        scores = M.dot(uncovered)
        
        # Find best unused block
        # Mask used blocks with -1 or 0 to ignore them
        # Since scores are non-negative, we can just set used scores to -1
        valid_scores = scores.copy()
        valid_scores[used] = -1
        
        best_idx = int(np.argmax(valid_scores))
        best_score = valid_scores[best_idx]
        
        if best_score <= 0:
            # Should not happen if a solution exists and we haven't covered all
            break
            
        used[best_idx] = True
        selected_blocks.append(best_idx)
        
        # Update uncovered
        # uncovered[i] = 0 if M[best_idx, i] == 1
        # uncovered = uncovered & ~M[best_idx]
        uncovered[M[best_idx] == 1] = 0
        
    # Convert indices back to blocks
    blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    # Redundancy Removal Phase
    # Try to remove blocks that are not essential
    # We need to know if a t-subset is covered by *other* selected blocks.
    # Let's maintain a count of how many selected blocks cover each t-subset.
    
    if not blocks:
        return blocks
        
    # Initialize coverage counts
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for b_idx in selected_blocks:
        cover_count[M[b_idx] == 1] += 1
        
    # List of indices of selected blocks
    sel_indices = list(selected_blocks)
    
    # Iterate and try to remove
    # We iterate over a copy of the list because we modify it
    i = 0
    while i < len(sel_indices):
        b_idx = sel_indices[i]
        # Check if any t-subset is covered ONLY by this block
        # i.e., cover_count[ts] == 1 for some ts where M[b_idx, ts] == 1
        block_mask = M[b_idx] == 1
        # If all t-subsets covered by this block are covered by at least 1 OTHER block,
        # then this block is redundant.
        # This is equivalent to: for all ts where M[b_idx, ts]==1, cover_count[ts] > 1
        # If this holds, we can remove it.
        
        # Efficient check:
        # Let S be the set of t-subsets covered by block b_idx.
        # If min(cover_count[S]) > 1, then redundant.
        
        # Get cover counts for this block's t-subsets
        counts_for_block = cover_count[block_mask]
        
        if len(counts_for_block) > 0 and np.min(counts_for_block) > 1:
            # Remove this block
            # Decrement cover_count for its t-subsets
            cover_count[block_mask] -= 1
            sel_indices.pop(i)
            # Do not increment i, check the next element at same index
        else:
            i += 1
            
    # Final list of blocks
    final_blocks = [list(all_blocks[i]) for i in sel_indices]
    
    return final_blocks
# EVOLVE-BLOCK-END
