import itertools
import numpy as np
import scipy.sparse as sp
import time
import random

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
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)
                
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)
    
    # Pre-compute the column indices for each block (for fast access)
    block_cols = []
    for i in range(n_k):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        cols = A.indices[row_start:row_end]
        block_cols.append(cols)
        
    # Also create a CSC matrix for faster column access if needed, 
    # but CSR is fine for row access (block -> subsets).
    # For finding blocks covering a set of subsets, we need column access or transpose.
    A_csc = A.tocsc()
    
    def greedy_cover(seed_blocks=None, max_iter=None):
        """Greedy set cover with optional seed blocks."""
        u = np.ones(n_t, dtype=np.int8)
        selected_blocks = []
        
        # If we have seed blocks, start with them
        if seed_blocks:
            for idx in seed_blocks:
                cols = block_cols[idx]
                u[cols] = 0
                selected_blocks.append(idx)
                
        if max_iter is None:
            max_iter = 10000
            
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if u.sum() == 0:
                break
                
            # Calculate gain for each block
            gains = A @ u
            # Find best block
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain == 0:
                break
                
            selected_blocks.append(best_idx)
            
            # Update uncovered vector
            cols = block_cols[best_idx]
            u[cols] = 0
            
        return selected_blocks
    
    def remove_redundant(selected_blocks):
        """Remove redundant blocks from the covering."""
        if not selected_blocks:
            return selected_blocks
            
        # Pre-compute lists of t-subset indices for each selected block
        selected_t_subs = []
        for idx in selected_blocks:
            cols = block_cols[idx]
            selected_t_subs.append(cols)
            
        # Initialize cover_count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for cols in selected_t_subs:
            cover_count[cols] += 1
            
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            for i in range(len(selected_blocks)):
                cols = selected_t_subs[i]
                # Check if any col in cols has cover_count == 1
                is_essential = False
                # Optimization: if we just updated counts, we can check quickly
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
                    
        return selected_blocks

    def local_search(selected_blocks, max_rounds=10):
        """
        Advanced local search:
        1. Remove redundant blocks.
        2. Try to swap a block with another to allow removal of a third block.
        """
        if not selected_blocks:
            return selected_blocks
            
        # Maintain cover_count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for idx in selected_blocks:
            cover_count[block_cols[idx]] += 1
            
        # Keep track of which blocks are in the cover for O(1) check
        in_cover = np.zeros(n_k, dtype=bool)
        for idx in selected_blocks:
            in_cover[idx] = True
            
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            if time.time() > deadline:
                break
            improved = False
            rounds += 1
            
            # 1. Remove redundant
            to_remove = []
            for i, idx in enumerate(selected_blocks):
                cols = block_cols[idx]
                if not np.any(cover_count[cols] == 1):
                    to_remove.append(i)
                    
            if to_remove:
                for i in sorted(to_remove, reverse=True):
                    idx = selected_blocks[i]
                    cover_count[block_cols[idx]] -= 1
                    in_cover[idx] = False
                    selected_blocks.pop(i)
                improved = True
                continue # Restart the loop as counts changed significantly
                
            # 2. Try swaps
            # For each block in the cover, try to replace it
            # We iterate over a copy because we might modify the list
            for i in range(len(selected_blocks)):
                if time.time() > deadline:
                    break
                    
                idx_old = selected_blocks[i]
                cols_old = block_cols[idx_old]
                
                # Temporarily remove idx_old
                cover_count[cols_old] -= 1
                
                # Find uncovered subsets
                uncovered_mask = (cover_count == 0)
                if not np.any(uncovered_mask):
                    # It was redundant, which should have been caught above, 
                    # but handle just in case
                    in_cover[idx_old] = False
                    selected_blocks.pop(i)
                    improved = True
                    # Revert cover_count? No, we popped it.
                    # But we are in a loop, so we break and restart outer loop logic?
                    # Better to break inner and let outer handle it.
                    break 
                
                # Find candidate blocks that cover at least one uncovered subset
                # Use A_csc to find rows (blocks) that have non-zeros in uncovered columns
                # A_csc[:, uncovered_mask] is a sparse matrix.
                # We want the row indices of non-zero elements.
                
                # Optimization: Instead of full matrix slice, just check candidates?
                # n_k is ~75k. n_t is ~11628.
                # uncovered_mask has ~few dozen True values usually.
                
                # Get candidate blocks
                # A_csc is CSC. 
                # We can get the non-zero indices in the columns where uncovered_mask is True.
                # This is effectively a sparse matrix-vector product or similar.
                
                # Simpler: Iterate over uncovered subsets and find blocks covering them.
                # But that's slow if many uncovered.
                
                # Let's use the sparse matrix structure.
                # A_csc.indices contains column indices for each row? No, CSC stores row indices per column.
                # A_csc.indices[k] are row indices for column k.
                
                candidate_set = set()
                for col_idx in np.where(uncovered_mask)[0]:
                    # Get rows (blocks) that cover this column
                    start = A_csc.indptr[col_idx]
                    end = A_csc.indptr[col_idx + 1]
                    rows = A_csc.indices[start:end]
                    candidate_set.update(rows)
                    
                # Filter candidates: must not be in cover (except idx_old which we removed)
                # Actually, if a candidate is in cover, swapping with it is useless or complex.
                # We only consider blocks NOT in cover.
                valid_candidates = [c for c in candidate_set if not in_cover[c]]
                
                if not valid_candidates:
                    # Revert
                    cover_count[cols_old] += 1
                    continue
                    
                # Score candidates: how many uncovered subsets do they cover?
                best_candidate = -1
                best_score = 0
                for c in valid_candidates:
                    cols_c = block_cols[c]
                    score = np.sum(uncovered_mask[cols_c])
                    if score > best_score:
                        best_score = score
                        best_candidate = c
                        
                if best_candidate == -1:
                    cover_count[cols_old] += 1
                    continue
                    
                # Try adding best_candidate
                idx_new = best_candidate
                cols_new = block_cols[idx_new]
                
                # Tentatively add idx_new
                cover_count[cols_new] += 1
                in_cover[idx_new] = True
                selected_blocks.append(idx_new)
                
                # Check if any block in the cover (excluding idx_old and idx_new) is now redundant
                redundant_found = -1
                for j, idx_check in enumerate(selected_blocks):
                    if idx_check == idx_old or idx_check == idx_new:
                        continue
                    cols_check = block_cols[idx_check]
                    if not np.any(cover_count[cols_check] == 1):
                        redundant_found = j
                        break
                        
                if redundant_found != -1:
                    # Success! Remove redundant block
                    idx_redundant = selected_blocks[redundant_found]
                    cover_count[block_cols[idx_redundant]] -= 1
                    in_cover[idx_redundant] = False
                    selected_blocks.pop(redundant_found)
                    
                    # Commit: Remove idx_old
                    # Note: idx_old is still in selected_blocks? 
                    # We didn't pop it yet. 
                    # We added idx_new. 
                    # We removed idx_redundant.
                    # Now we need to remove idx_old from the list.
                    # Find index of idx_old
                    old_pos = selected_blocks.index(idx_old)
                    selected_blocks.pop(old_pos)
                    in_cover[idx_old] = False
                    
                    improved = True
                    # Break and restart outer loop to recalculate all redundancies
                    break 
                else:
                    # Failed, revert
                    cover_count[cols_new] -= 1
                    in_cover[idx_new] = False
                    selected_blocks.pop() # Remove idx_new
                    cover_count[cols_old] += 1 # Restore idx_old count
                    # in_cover[idx_old] was not changed, so it's still True.
                    
        return selected_blocks

    # Run greedy with different random seeds
    best_result = None
    best_size = float('inf')
    
    # First, run standard greedy
    result = greedy_cover()
    result = remove_redundant(result)
    result = local_search(result)
    if len(result) < best_size:
        best_result = result
        best_size = len(result)
    
    # Try randomized restarts
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Create a random seed
        # Pick a few random blocks to start with
        seed_size = min(10, n_k)
        seed_blocks = random.sample(range(n_k), seed_size)
        
        result = greedy_cover(seed_blocks=seed_blocks)
        result = remove_redundant(result)
        result = local_search(result)
        
        if len(result) < best_size:
            best_result = result
            best_size = len(result)
            
    # Convert indices back to blocks
    if best_result is None:
        best_result = []
        
    final_blocks = [list(k_subs[i]) for i in best_result]
    
    return final_blocks
