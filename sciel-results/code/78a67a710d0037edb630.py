import itertools
import time
import random
import numpy as np
import scipy.sparse as sp

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    # 1. Precomputation
    # Map t-subsets to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Map k-subsets (blocks) to list of t-subset indices they cover
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Build sparse matrix A of size num_k_subsets x num_t_subsets
    # A[i, j] = 1 if t_subsets[j] is subset of k_subsets[i]
    row_indices = []
    col_indices = []
    data = []
    
    # For each block, find which t-subsets it contains
    for i, block in enumerate(all_k_subsets):
        for t_sub in itertools.combinations(block, t):
            j = t_subset_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_k_subsets, num_t_subsets), dtype=np.int8)
    
    # Pre-compute the column indices for each block (for fast access)
    block_cols = []
    for i in range(num_k_subsets):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        cols = A.indices[row_start:row_end]
        block_cols.append(cols)
        
    # Helper: Fast Greedy to cover a set of t-subset indices
    def fast_greedy(uncovered_indices, seed_blocks=None):
        """
        Greedy set cover.
        uncovered_indices: set or list of t-subset indices to cover.
        seed_blocks: list of block indices to include initially.
        """
        # Initialize uncovered vector
        u = np.zeros(num_t_subsets, dtype=np.int8)
        if isinstance(uncovered_indices, set):
            u[list(uncovered_indices)] = 1
        else:
            u[uncovered_indices] = 1
            
        selected_blocks = []
        
        # Add seed blocks if provided
        if seed_blocks:
            for b_idx in seed_blocks:
                cols = block_cols[b_idx]
                u[cols] = 0
                selected_blocks.append(b_idx)
                
        while u.sum() > 0:
            if time.time() > deadline:
                break
                
            # Calculate gain for each block: number of uncovered t-subsets it covers
            # A is (num_k x num_t), u is (num_t). A @ u gives (num_k) vector of gains.
            gains = A @ u
            
            # Find best block
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain == 0:
                break # Should not happen if u.sum() > 0 and coverage is possible
                
            selected_blocks.append(best_idx)
            
            # Update uncovered vector
            cols = block_cols[best_idx]
            u[cols] = 0
            
        return selected_blocks

    # Initial Greedy
    initial_uncovered = list(range(num_t_subsets))
    current_blocks = fast_greedy(initial_uncovered)
    
    # Helper to remove redundant blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
            
        # Count coverage
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            counts[block_cols[b_idx]] += 1
            
        redundant = []
        for b_idx in blocks:
            cols = block_cols[b_idx]
            # If all cols in this block have count > 1, it's redundant
            if np.all(counts[cols] > 1):
                redundant.append(b_idx)
                
        if redundant:
            # Remove one random redundant block
            b_to_remove = random.choice(redundant)
            new_blocks = [b for b in blocks if b != b_to_remove]
            return new_blocks
        return blocks

    # Local Search
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    while time.time() < deadline:
        # 1. Try removing redundant blocks
        new_blocks = remove_redundant(current_blocks)
        if len(new_blocks) < len(current_blocks):
            current_blocks = new_blocks
            if len(current_blocks) < best_size:
                best_blocks = list(current_blocks)
                best_size = len(best_blocks)
            continue # Loop again to remove more if possible
        
        # 2. Kick: Remove a random block and re-cover
        if not current_blocks:
            break
            
        b_to_remove = random.choice(current_blocks)
        
        # Find t-subsets that become uncovered
        # Count coverage without b_to_remove
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            if b_idx == b_to_remove:
                continue
            counts[block_cols[b_idx]] += 1
            
        cols_removed = block_cols[b_to_remove]
        uncovered_after_removal = []
        for t_idx in cols_removed:
            if counts[t_idx] == 0:
                uncovered_after_removal.append(t_idx)
                
        if not uncovered_after_removal:
            # This block was redundant (should have been caught above, but just in case)
            current_blocks = [b for b in current_blocks if b != b_to_remove]
            if len(current_blocks) < best_size:
                best_blocks = list(current_blocks)
                best_size = len(best_blocks)
            continue
            
        # Greedily cover the newly uncovered t-subsets
        # We start with the remaining blocks as seeds? No, fast_greedy adds to an empty set.
        # We need to cover `uncovered_after_removal` using blocks.
        # The resulting blocks from fast_greedy will cover these.
        # We then combine them with the remaining blocks.
        # However, fast_greedy might pick blocks that are already in the remaining set, which is fine (duplicates removed later) 
        # or might pick blocks that cover more than just the uncovered ones (which is good).
        
        added_blocks = fast_greedy(uncovered_after_removal)
        
        # Combine
        new_current_blocks = [b for b in current_blocks if b != b_to_remove] + added_blocks
        
        # Deduplicate
        new_current_blocks = list(set(new_current_blocks))
        
        current_blocks = new_current_blocks
        
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(best_blocks)
            
        # Optional: If we get stuck for a while, reset to best?
        # For now, just continue. The random kick ensures exploration.

    # Convert block indices back to actual blocks (lists of points)
    unique_best_blocks = list(set(best_blocks))
    result_blocks = [list(all_k_subsets[b_idx]) for b_idx in unique_best_blocks]
    
    # Final check: ensure all t-subsets are covered
    covered_final = set()
    for b_idx in unique_best_blocks:
        covered_final.update(block_cols[b_idx])
        
    if len(covered_final) < num_t_subsets:
        # Fallback to initial greedy which is guaranteed to cover
        result_blocks = [list(all_k_subsets[b_idx]) for b_idx in current_blocks]
        # Re-verify current_blocks
        covered_check = set()
        for b_idx in current_blocks:
            covered_check.update(block_cols[b_idx])
        if len(covered_check) < num_t_subsets:
             # This should not happen. Return a trivial cover? 
             # Or just the initial one.
             pass

    return result_blocks
