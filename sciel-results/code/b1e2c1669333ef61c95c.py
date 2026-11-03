import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Generate all k-subsets (candidate blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # Precompute: for each block, list of t-subset indices it covers
    block_tsets = []
    for block in k_subs:
        tset_indices = []
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                tset_indices.append(j)
        block_tsets.append(tset_indices)
    
    # Build sparse matrix A of size n_k x n_t
    row_indices = []
    col_indices = []
    for i, tset_list in enumerate(block_tsets):
        for j in tset_list:
            row_indices.append(i)
            col_indices.append(j)
    
    A = sp.csr_matrix((np.ones(len(row_indices), dtype=np.int8), (row_indices, col_indices)), 
                       shape=(n_k, n_t), dtype=np.int8)
    
    # Transpose: A_T is n_t x n_k, where A_T[j, i] = 1 if block i covers t-subset j
    A_T = A.T.tocsr()
    
    # SPECTRAL CLUSTERING APPROACH:
    # 1. Build co-occurrence matrix of t-subsets: M[j1, j2] = number of blocks covering both j1 and j2
    # 2. Use spectral clustering to find communities of t-subsets that tend to be covered together
    # 3. For each community, greedily select blocks that cover many subsets in that community
    
    # Co-occurrence: M = A_T @ A (n_t x n_t)
    # This is expensive for large n_t, so we use a sampled approximation
    # For v=19, k=8, t=5: n_t = C(19,5) = 11628, which is manageable
    # M[j1, j2] = sum_i A_T[j1,i] * A_T[j2,i] = number of blocks containing both t-subsets
    
    # Use a more efficient approach: for each block, add to M the outer product of its covered t-subsets
    # But that's also expensive. Let's use a different strategy.
    
    # ALTERNATIVE SPECTRAL APPROACH:
    # Use the block incidence structure to find dense subgraphs
    # 1. Compute degree of each t-subset (how many blocks cover it)
    # 2. Use power iteration on A_T @ A to find principal eigenvector
    # 3. Cluster t-subsets by similarity to the principal eigenvector
    
    # Step 1: Degree of each t-subset
    t_degree = np.array(A_T.sum(axis=1)).flatten().astype(np.float64)
    
    # Step 2: Approximate spectral embedding using a few iterations
    # We want to find clusters where t-subsets have similar coverage patterns
    # Use random projection + k-means style clustering
    
    # For efficiency, use a simpler approach:
    # Group t-subsets by which points they contain (overlap structure)
    # Two t-subsets are "compatible" if they can be in the same block
    
    # Actually, let's use a different structural approach:
    # 1. Start with a good initial solution using a modified greedy
    # 2. Then do aggressive local search with swaps
    
    # MODIFIED GREEDY with community awareness:
    # 1. Compute for each t-subset its "coverage difficulty" (fewer covering blocks = harder)
    # 2. Greedily select blocks, but weight uncovered subsets by their difficulty
    
    u = np.ones(n_t, dtype=np.int8)
    
    # Weight uncovered subsets: harder ones get higher weight
    # difficulty[j] = 1 / degree[j] (inverse of how many blocks can cover it)
    difficulty = 1.0 / np.maximum(t_degree, 1)
    difficulty = difficulty / difficulty.sum()  # Normalize
    
    selected_blocks = []
    
    # Phase 1: Weighted greedy with community bias
    # Use a few random restarts and keep the best
    best_solution = None
    best_size = float('inf')
    
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:  # Leave time for local search
            break
            
        u = np.ones(n_t, dtype=np.int8)
        sel = []
        
        # Add some randomness to break ties
        if restart > 0:
            # Perturb the difficulty weights
            random.seed(restart * 12345 + 678)
            perturb = np.random.uniform(0.8, 1.2, size=n_t)
            weighted_difficulty = difficulty * perturb
        else:
            weighted_difficulty = difficulty
        
        # Greedy selection
        while u.sum() > 0:
            if time.time() > deadline - 60:
                break
            
            # Gain: sum of weighted difficulties of uncovered subsets covered by each block
            gains = A @ (u * weighted_difficulty)
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain == 0:
                break
            
            sel.append(best_idx)
            
            # Update uncovered
            row_start = A.indptr[best_idx]
            row_end = A.indptr[best_idx + 1]
            cols = A.indices[row_start:row_end]
            u[cols] = 0
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_solution = sel[:]
        
        if time.time() > deadline - 30:
            break
    
    if best_solution is None:
        # Fallback to standard greedy
        u = np.ones(n_t, dtype=np.int8)
        best_solution = []
        while u.sum() > 0:
            if time.time() > deadline - 10:
                break
            gains = A @ u
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            best_solution.append(best_idx)
            row_start = A.indptr[best_idx]
            row_end = A.indptr[best_idx + 1]
            cols = A.indices[row_start:row_end]
            u[cols] = 0
    
    selected_blocks = best_solution[:]
    
    # LOCAL SEARCH: Aggressive 1-for-1 swaps and deletions
    # Precompute cover counts
    cover_count = np.zeros(n_t, dtype=np.int32)
    for idx in selected_blocks:
        row_start = A.indptr[idx]
        row_end = A.indptr[idx + 1]
        cols = A.indices[row_start:row_end]
        cover_count[cols] += 1
    
    # Precompute for each selected block, the list of t-subsets it covers
    sel_tsets = []
    for idx in selected_blocks:
        row_start = A.indptr[idx]
        row_end = A.indptr[idx + 1]
        sel_tsets.append(A.indices[row_start:row_end].copy())
    
    # Deletion pass: remove blocks that don't uniquely cover any subset
    improved = True
    while improved:
        if time.time() > deadline - 15:
            break
        improved = False
        
        # Find redundant blocks
        to_remove = []
        for i in range(len(selected_blocks)):
            cols = sel_tsets[i]
            # Check if any subset is covered only by this block
            if len(cols) > 0 and np.min(cover_count[cols]) == 1:
                continue  # Essential
            else:
                to_remove.append(i)
        
        if to_remove:
            improved = True
            for i in sorted(to_remove, reverse=True):
                cols = sel_tsets[i]
                cover_count[cols] -= 1
                selected_blocks.pop(i)
                sel_tsets.pop(i)
    
    # Swap pass: try to replace pairs of blocks with one block
    # For each pair (i, j), check if there's a block that covers all subsets 
    # that are uniquely covered by i or j
    if time.time() < deadline - 20:
        # Try 2-for-1 reductions
        n_sel = len(selected_blocks)
        for i in range(n_sel):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, n_sel):
                if time.time() > deadline - 15:
                    break
                
                # Subsets uniquely covered by i or j
                cols_i = sel_tsets[i]
                cols_j = sel_tsets[j]
                
                # Find subsets that would become uncovered if we remove both i and j
                # These are subsets where cover_count == 1 and covered by i or j
                unique_to_ij = []
                for c in cols_i:
                    if cover_count[c] == 1:
                        unique_to_ij.append(c)
                for c in cols_j:
                    if cover_count[c] == 1:
                        unique_to_ij.append(c)
                
                if not unique_to_ij:
                    continue
                
                # Check if there's a single block that covers all unique_to_ij
                # and doesn't create issues
                unique_set = set(unique_to_ij)
                
                # Find candidate blocks that cover all unique_to_ij
                # This is expensive, so we limit the search
                found_replacement = False
                for cand in range(n_k):
                    if cand in selected_blocks:
                        continue
                    # Check if cand covers all unique_to_ij
                    row_start = A.indptr[cand]
                    row_end = A.indptr[cand + 1]
                    cand_cols = set(A.indices[row_start:row_end])
                    
                    if unique_set.issubset(cand_cols):
                        # Check if removing i and j and adding cand maintains coverage
                        # We need to verify that all subsets still covered
                        # Subsets covered by i but not j: need to be covered by cand or others
                        # This is complex; for now, just check the unique ones
                        # Actually, we need to be more careful
                        
                        # Simpler: just try the swap and verify
                        # Remove i and j, add cand
                        # Update cover_count
                        cover_count[cols_i] -= 1
                        cover_count[cols_j] -= 1
                        
                        # Check if any subset became uncovered
                        if np.min(cover_count) >= 1:
                            # Add cand
                            cand_cols_arr = A.indices[A.indptr[cand]:A.indptr[cand + 1]]
                            cover_count[cand_cols_arr] += 1
                            
                            # Perform the swap
                            # Remove j first (higher index)
                            selected_blocks.pop(j)
                            sel_tsets.pop(j)
                            selected_blocks.pop(i)
                            sel_tsets.pop(i)
                            selected_blocks.append(cand)
                            sel_tsets.append(cand_cols_arr.copy())
                            
                            improved = True
                            found_replacement = True
                            break
                        else:
                            # Revert
                            cover_count[cols_i] += 1
                            cover_count[cols_j] += 1
                
                if found_replacement:
                    # Adjust indices since we modified the list
                    break
            
            if found_replacement:
                continue
    
    # Final redundancy removal after swaps
    if time.time() < deadline - 5:
        # Recompute cover_count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for idx in selected_blocks:
            row_start = A.indptr[idx]
            row_end = A.indptr[idx + 1]
            cols = A.indices[row_start:row_end]
            cover_count[cols] += 1
        
        sel_tsets = []
        for idx in selected_blocks:
            row_start = A.indptr[idx]
            row_end = A.indptr[idx + 1]
            sel_tsets.append(A.indices[row_start:row_end].copy())
        
        improved = True
        while improved:
            if time.time() > deadline - 2:
                break
            improved = False
            to_remove = []
            for i in range(len(selected_blocks)):
                cols = sel_tsets[i]
                if len(cols) > 0 and np.min(cover_count[cols]) == 1:
                    continue
                else:
                    to_remove.append(i)
            
            if to_remove:
                improved = True
                for i in sorted(to_remove, reverse=True):
                    cols = sel_tsets[i]
                    cover_count[cols] -= 1
                    selected_blocks.pop(i)
                    sel_tsets.pop(i)
    
    # Convert to final blocks
    final_blocks = [list(k_subs[i]) for i in selected_blocks]
    
    return final_blocks
