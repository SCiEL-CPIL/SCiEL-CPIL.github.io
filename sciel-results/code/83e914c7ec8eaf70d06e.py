import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    points = list(range(v))
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Build incidence matrix A (num_blocks x num_t_subsets)
    row_indices = []
    col_indices = []
    data = []
    
    block_t_subsets = []
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        block_t_subsets.append(subs)
        for t_sub in subs:
            j = t_subset_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
    
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    
    # Greedy set cover
    active = np.ones(num_t_subsets, dtype=np.float64)
    selected = []
    
    while True:
        if time.time() > deadline - 15.0:
            break
        
        gains = A.dot(active)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        
        b_idx = np.argmax(gains)
        selected.append(b_idx)
        
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        active[covered] = 0
        
        if np.all(active == 0):
            break
    
    # Local search: deletion pass
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
    
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            if np.all(cover_counts[covered] > 1):
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected = [b for i, b in enumerate(selected) if i not in remove_set]
            changed = True
    
    # Iterated local search: swap-based improvement
    # Precompute overlap between all block pairs
    # For each block, we know which t-subsets it covers
    # Two blocks "conflict" if removing one and adding the other improves things
    
    # Use a different strategy: try replacing one block with another that covers 
    # more of the remaining gaps while maintaining coverage
    
    current_count = len(selected)
    selected_set = set(selected)
    
    # Try swap moves: remove one block, try to cover its unique t-subsets with one new block
    # Compute for each block, which t-subsets are ONLY covered by it
    unique_to_block = {}
    for b_idx in selected:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        # Find t-subsets where cover_count == 1 and this block covers them
        mask = (cover_counts[covered] == 1)
        if np.any(mask):
            unique_to_block[b_idx] = covered[mask]
    
    improved = True
    while improved and time.time() < deadline - 20.0:
        improved = False
        
        # Try to find a block to remove that has minimal unique coverage,
        # then try to cover those unique subsets with a new block
        # Sort by number of unique t-subsets (prefer removing blocks with fewer uniques)
        candidates = []
        for b_idx in selected:
            if b_idx in unique_to_block:
                n_unique = len(unique_to_block[b_idx])
            else:
                n_unique = 0
            candidates.append((n_unique, b_idx))
        
        candidates.sort()
        
        for n_unique, b_idx in candidates:
            if time.time() > deadline - 20.0:
                break
            
            if b_idx not in selected_set:
                continue
            
            if n_unique == 0:
                # Safe to remove
                selected_set.discard(b_idx)
                selected = list(selected_set)
                # Update cover counts
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
                # Recompute unique_to_block
                unique_to_block = {}
                for b2 in selected:
                    cov2 = A.indices[A.indptr[b2]:A.indptr[b2+1]]
                    mask2 = (cover_counts[cov2] == 1)
                    if np.any(mask2):
                        unique_to_block[b2] = cov2[mask2]
                improved = True
                break
            
            elif n_unique <= 3:
                # Try to find a new block that covers all these unique t-subsets
                # and check if the swap is beneficial
                target_tsubs = unique_to_block[b_idx]
                
                # Find blocks that cover all target t-subsets
                # A new block must contain all these t-subsets
                found_replacement = False
                for new_b in range(num_blocks):
                    if new_b in selected_set:
                        continue
                    
                    cov_new = A.indices[A.indptr[new_b]:A.indptr[new_b+1]]
                    # Check if new block covers all target t-subsets
                    if all(t in set(cov_new) for t in target_tsubs):
                        # Do the swap
                        selected_set.discard(b_idx)
                        selected_set.add(new_b)
                        selected = list(selected_set)
                        
                        # Update cover counts
                        covered_old = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_counts[covered_old] -= 1
                        covered_new = A.indices[A.indptr[new_b]:A.indptr[new_b+1]]
                        cover_counts[covered_new] += 1
                        
                        # Recompute unique_to_block
                        unique_to_block = {}
                        for b2 in selected:
                            cov2 = A.indices[A.indptr[b2]:A.indptr[b2+1]]
                            mask2 = (cover_counts[cov2] == 1)
                            if np.any(mask2):
                                unique_to_block[b2] = cov2[mask2]
                        
                        improved = True
                        found_replacement = True
                        break
                
                if found_replacement:
                    break
    
    # Final deletion pass
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
    
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            if np.all(cover_counts[covered] > 1):
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected = [b for i, b in enumerate(selected) if i not in remove_set]
            changed = True
    
    blocks = [list(all_blocks[b_idx]) for b_idx in selected]
    return blocks
