import itertools
import numpy as np
import scipy.sparse as sp
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Generate all k-subsets (candidate blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build sparse matrix A of size n_k x n_t
    # A[i, j] = 1 if t_subs[j] is subset of k_subs[i]
    row_indices = []
    col_indices = []
    data = []
    
    # For each block, find which t-subsets it contains
    for i, block in enumerate(k_subs):
        block_set = set(block)
        # Find all t-subsets of this block
        # It's faster to iterate t_subs and check, or generate t-subsets of block
        # Generating t-subsets of block is C(8,5)=56, checking against map is O(1)
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)
                
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)
    
    # Uncovered vector
    u = np.ones(n_t, dtype=np.int8)
    
    selected_blocks = []
    
    # Greedy Set Cover
    while u.sum() > 0:
        if time.time() > deadline:
            # Return whatever we have, though it might be invalid
            # But for the purpose of the task, we must return a valid covering.
            # If we run out of time, we might not have a valid one.
            # However, with this fast method, we won't run out.
            break
            
        # Calculate gain for each block
        gains = A @ u
        # Find best block
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # Should not happen if a covering exists
            break
            
        selected_blocks.append(best_idx)
        
        # Update uncovered vector
        # Get columns where A[best_idx, :] is 1
        # In CSR, we can access the row
        row_start = A.indptr[best_idx]
        row_end = A.indptr[best_idx + 1]
        cols = A.indices[row_start:row_end]
        u[cols] = 0
        
    # Convert indices back to blocks
    blocks = [list(k_subs[i]) for i in selected_blocks]
    
    # Redundancy Removal
    # We want to remove blocks that don't cover any subset uniquely
    # cover_count[j] = number of selected blocks covering t_subs[j]
    cover_count = np.zeros(n_t, dtype=np.int32)
    
    # Pre-compute lists of t-subset indices for each selected block for faster access
    # selected_t_subs[i] is a list of t-subset indices in block i
    selected_t_subs = []
    for idx in selected_blocks:
        row_start = A.indptr[idx]
        row_end = A.indptr[idx + 1]
        cols = A.indices[row_start:row_end]
        selected_t_subs.append(cols)
        
    # Iteratively remove redundant blocks
    # We iterate through blocks. If a block has no unique subsets, remove it.
    # We need to be careful with the order.
    # A simple approach:
    # 1. Initialize cover_count based on all selected blocks.
    for cols in selected_t_subs:
        cover_count[cols] += 1
        
    # Now check each block
    # We can't just iterate once because removing one block might make another redundant.
    # So we loop until no change.
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        # Iterate in reverse to avoid index shifting issues if we pop
        # Or just mark for removal
        to_remove = []
        for i in range(len(selected_blocks)):
            cols = selected_t_subs[i]
            # Check if any col in cols has cover_count == 1
            # If yes, this block is essential (covers at least one subset uniquely)
            # If no, it is redundant
            # unique_count = np.sum(cover_count[cols] == 1)
            # If unique_count > 0, keep. Else, remove.
            
            # Optimization: if any element in cols has count 1, keep.
            # We can check if min(cover_count[cols]) == 1? No, if min is 1, it's essential.
            # If min > 1, it's redundant.
            # But we need to be sure.
            # Let's just check if any is 1.
            # Since cols is small (56), this is fast.
            is_essential = False
            for c in cols:
                if cover_count[c] == 1:
                    is_essential = True
                    break
            
            if not is_essential:
                to_remove.append(i)
                
        if to_remove:
            changed = True
            # Remove blocks from the end to maintain indices
            for i in sorted(to_remove, reverse=True):
                # Update cover_count
                cols = selected_t_subs[i]
                cover_count[cols] -= 1
                # Remove from lists
                selected_blocks.pop(i)
                selected_t_subs.pop(i)
                
    # Final list of blocks
    final_blocks = [list(k_subs[i]) for i in selected_blocks]
    
    return final_blocks
