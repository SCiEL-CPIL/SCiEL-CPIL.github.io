import itertools
import time
import random
import numpy as np
import scipy.sparse as sp

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(all_points, k))
    n_k = len(all_k_blocks)
    
    # Pre-compute block to t-subset indices mapping
    # M is a sparse matrix of shape (n_k, n_t)
    # M[i, j] = 1 if t-sub j is contained in block i
    
    # Efficient construction of sparse matrix
    row_indices = []
    col_indices = []
    data = []
    
    for i, block in enumerate(all_k_blocks):
        # Get all t-subsets of this block
        for sub in itertools.combinations(block, t):
            j = t_sub_to_idx[sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    M = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int32)
    
    # Greedy Construction
    covered_count = np.zeros(n_t, dtype=np.int32)
    uncovered_mask = np.ones(n_t, dtype=np.int32) # 1 if uncovered
    
    blocks_indices = [] # List of indices into all_k_blocks
    
    def time_ok():
        return time.time() < deadline

    # Initial Greedy Phase
    while time_ok():
        # Calculate gains for all blocks
        # gain[i] = number of currently uncovered t-subsets covered by block i
        gains = M.dot(uncovered_mask)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Pick a block with max gain
        # To avoid bias, pick randomly from those with max gain? 
        # Or just the first one. Let's pick the one with index argmax.
        best_idx = np.argmax(gains)
        
        blocks_indices.append(best_idx)
        
        # Update coverage
        # Get the non-zero columns of M[best_idx]
        # M[best_idx] is a sparse row
        row = M[best_idx]
        cols = row.indices
        covered_count[cols] += 1
        # Update uncovered_mask
        uncovered_mask[cols] = (covered_count[cols] == 0).astype(np.int32)
        
        # Optimization: If we have covered everything, break
        if np.sum(uncovered_mask) == 0:
            break

    # Local Search Phase
    # We will try to reduce the number of blocks.
    # Moves: Deletion, Swap
    
    current_blocks = blocks_indices[:]
    
    def is_redundant(block_idx):
        """Check if removing block_idx leaves all t-subsets covered."""
        row = M[block_idx]
        cols = row.indices
        # If any of these cols have count 1, removing this block makes them uncovered
        # So it is NOT redundant if min(covered_count[cols]) == 1
        # It IS redundant if min(covered_count[cols]) > 1
        if len(cols) == 0:
            return True
        min_count = np.min(covered_count[cols])
        return min_count > 1

    def get_unique_cols(block_idx):
        """Get columns where this block is the only cover."""
        row = M[block_idx]
        cols = row.indices
        # Filter cols where covered_count is 1
        mask = (covered_count[cols] == 1)
        return cols[mask]

    def try_swap(block_idx, candidate_idx):
        """
        Try to swap block_idx with candidate_idx.
        Returns True if swap is valid (maintains coverage).
        """
        if block_idx == candidate_idx:
            return False
        
        # 1. Get unique subsets of block_idx
        unique_cols = get_unique_cols(block_idx)
        
        # 2. Check if candidate_idx covers all unique_cols
        if len(unique_cols) > 0:
            # Check if candidate covers all unique_cols
            # candidate_row = M[candidate_idx]
            # We need M[candidate_idx, unique_cols].sum() == len(unique_cols)
            # Efficient check:
            cand_row = M[candidate_idx]
            # Intersect indices?
            # Simpler: check if all unique_cols are in cand_row.indices
            # Since unique_cols is small (usually), we can just check membership or sum
            # M[candidate_idx, unique_cols] is a sparse vector
            # Let's just use the data structure
            # Note: M is CSR. Accessing specific columns is O(1) if we have COO or CSC?
            # M is CSR. M[i, j] access is slow.
            # Better: Convert M to CSC for column access?
            # Or just check:
            # The number of unique cols is small.
            # Let's check if candidate covers them.
            # We can pre-compute a set of indices for candidate? No, too slow.
            # Let's use the property that M is sparse.
            # We'll just check the count.
            # M[candidate_idx, unique_cols] creates a sparse matrix.
            # .sum() is fast.
            if M[candidate_idx, unique_cols].sum() != len(unique_cols):
                return False
        
        # If we get here, the swap is valid.
        # Perform swap
        # Remove old
        old_row = M[block_idx]
        old_cols = old_row.indices
        covered_count[old_cols] -= 1
        uncovered_mask[old_cols] = (covered_count[old_cols] == 0).astype(np.int32)
        
        # Add new
        new_row = M[candidate_idx]
        new_cols = new_row.indices
        covered_count[new_cols] += 1
        uncovered_mask[new_cols] = (covered_count[new_cols] == 0).astype(np.int32)
        
        return True

    # Local Search Loop
    # We alternate between deletion and swap attempts.
    # We use a random order to avoid cycles.
    
    # Sort blocks by "criticality" (number of unique subsets) to try deleting critical ones last?
    # Or just iterate.
    
    improved = True
    while improved and time_ok():
        improved = False
        
        # Pass 1: Deletion
        # Iterate in random order
        for i in range(len(current_blocks) - 1, -1, -1):
            if not time_ok():
                break
            b_idx = current_blocks[i]
            if is_redundant(b_idx):
                # Remove it
                row = M[b_idx]
                cols = row.indices
                covered_count[cols] -= 1
                uncovered_mask[cols] = (covered_count[cols] == 0).astype(np.int32)
                current_blocks.pop(i)
                improved = True
        
        # Pass 2: Swap
        # Try to swap each block with a random candidate
        # We limit the number of candidates per block to keep it fast
        for i in range(len(current_blocks)):
            if not time_ok():
                break
            b_idx = current_blocks[i]
            
            # Get unique subsets
            unique_cols = get_unique_cols(b_idx)
            
            # If no unique subsets, it's redundant (handled in Pass 1, but maybe became redundant after previous swaps)
            if len(unique_cols) == 0:
                continue
            
            # Try a few random candidates
            # We want candidates that cover unique_cols.
            # We can sample random blocks from all_k_blocks
            # Probability of finding a good one is low if unique_cols is large.
            # If unique_cols is small (e.g., 1 or 2), it's easier.
            
            # Optimization: If unique_cols is large, skip?
            # Let's try 10 random candidates.
            for _ in range(10):
                cand_idx = random.randint(0, n_k - 1)
                if cand_idx == b_idx:
                    continue
                if try_swap(b_idx, cand_idx):
                    current_blocks[i] = cand_idx
                    improved = True
                    break # Move to next block
                    
    # Final Deletion Pass
    if time_ok():
        for i in range(len(current_blocks) - 1, -1, -1):
            if not time_ok():
                break
            b_idx = current_blocks[i]
            if is_redundant(b_idx):
                row = M[b_idx]
                cols = row.indices
                covered_count[cols] -= 1
                current_blocks.pop(i)

    # Convert indices back to blocks
    final_blocks = [list(all_k_blocks[i]) for i in current_blocks]
    
    # Verify validity (safety check)
    # This is O(N_blocks * C(k,t)), which is fast.
    # But we trust the logic.
    
    return final_blocks
