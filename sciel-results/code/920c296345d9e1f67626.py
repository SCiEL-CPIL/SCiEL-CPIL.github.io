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
    
    def local_improve(selected_blocks, max_rounds=3):
        """Try to improve the covering by swapping blocks."""
        if not selected_blocks:
            return selected_blocks
            
        best = list(selected_blocks)
        best_size = len(best)
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            if len(best) == 0:
                break
                
            improved = False
            # Try to replace each block with a better one
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                    
                # Temporarily remove block i
                cols_to_remove = block_cols[best[i]]
                
                # Find which t-subsets would become uncovered
                # Count how many other blocks cover each of these t-subsets
                temp_cover = np.zeros(n_t, dtype=np.int8)
                for j, idx in enumerate(best):
                    if j == i:
                        continue
                    temp_cover[block_cols[idx]] += 1
                
                # A t-subset becomes uncovered if temp_cover[c] == 0 for c in cols_to_remove
                # We want to find a block that covers at least one of these
                # and doesn't add too many new blocks
                
                # Actually, let's try a different approach:
                # For each candidate block, compute how many currently covered t-subsets
                # it would help cover if we remove block i
                # This is complex, so let's do a simpler local search:
                # Try adding a block and removing a different block if it helps
                
                # Simple approach: try to find a block that covers more uncovered subsets
                # than the current block, considering the rest of the covering
                current_gain = 0
                for c in cols_to_remove:
                    if temp_cover[c] == 0:
                        current_gain += 1
                        
                if current_gain == 0:
                    # This block is essential, skip
                    continue
                
                # Try all other blocks to see if any can replace it with better coverage
                # This is expensive, so let's limit it
                # Actually, let's just try to add a block that covers some of the
                # t-subsets that would become uncovered, then remove a redundant block
                
                # For now, let's do a simpler local search:
                # Try swapping block i with a random other block if it doesn't break coverage
                
                # Pick a random candidate
                candidate = random.randint(0, n_k - 1)
                if candidate == best[i]:
                    continue
                    
                # Check if we can swap
                # After removing i and adding candidate, is everything still covered?
                # We need to check that all t-subsets covered by i are also covered by
                # the remaining blocks or by candidate
                new_cover = temp_cover.copy()
                new_cover[block_cols[candidate]] += 1
                
                # Check if all t-subsets in cols_to_remove are covered
                can_swap = True
                for c in cols_to_remove:
                    if new_cover[c] == 0:
                        can_swap = False
                        break
                        
                if can_swap:
                    # Check if the new block is redundant (i.e., all its t-subsets are
                    # already covered by the remaining blocks)
                    # If so, we might be able to remove it later
                    best[i] = candidate
                    improved = True
                    break
                    
            if not improved:
                break
                
        return best
    
    # Run greedy with different random seeds
    best_result = None
    best_size = float('inf')
    
    # First, run standard greedy
    result = greedy_cover()
    result = remove_redundant(result)
    if len(result) < best_size:
        best_result = result
        best_size = len(result)
    
    # Try randomized restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Create a random seed
        # Pick a few random blocks to start with
        seed_size = min(5, n_k)
        seed_blocks = random.sample(range(n_k), seed_size)
        
        result = greedy_cover(seed_blocks=seed_blocks)
        result = remove_redundant(result)
        
        if len(result) < best_size:
            best_result = result
            best_size = len(result)
    
    # Try local improvement on the best result
    if best_result and time.time() < deadline - 10:
        improved = local_improve(best_result, max_rounds=2)
        improved = remove_redundant(improved)
        if len(improved) < best_size:
            best_result = improved
            best_size = len(improved)
    
    # Convert indices back to blocks
    if best_result is None:
        best_result = []
        
    final_blocks = [list(k_subs[i]) for i in best_result]
    
    return final_blocks
