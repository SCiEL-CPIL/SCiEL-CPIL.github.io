import itertools
import time
import numpy as np
import random
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin for 150s limit
    
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Build incidence matrix: block -> t-subsets it covers
    row_indices = []
    col_indices = []
    data = []
    
    for i, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            j = t_subset_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
    
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    
    # Precompute for each t-subset, which blocks cover it (for later use)
    # A.T gives t_subset -> blocks
    A_T = A.T.tocsr()
    
    # Phase 1: Greedy Set Cover (fast)
    active = np.ones(num_t_subsets, dtype=np.float64)
    selected_block_indices = []
    
    while True:
        if time.time() > deadline - 30.0:
            break
        
        gains = A.dot(active)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        
        b_idx = int(np.argmax(gains))
        selected_block_indices.append(b_idx)
        
        covered_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        active[covered_indices] = 0
        
        if np.all(active == 0):
            break
    
    # Phase 2: Local Search with Block Exchanges
    # Maintain cover counts for each t-subset
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
    
    # Helper: get list of blocks that cover a given t-subset
    def blocks_covering_subset(sub_idx):
        start = A_T.indptr[sub_idx]
        end = A_T.indptr[sub_idx + 1]
        return A_T.indices[start:end]
    
    # Helper: check if removing a block leaves all its subsets still covered
    def is_redundant(b_idx, counts):
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        for c in covered:
            if counts[c] == 1:
                return False
        return True
    
    # Helper: try to find a replacement block for a given block
    def find_replacement(b_idx_to_remove, current_blocks_set, counts, time_limit=0.5):
        """Find a block not in current solution that covers all uniquely-covered subsets of b_idx_to_remove."""
        covered_by_removed = A.indices[A.indptr[b_idx_to_remove]:A.indptr[b_idx_to_remove+1]]
        
        # Find subsets that are uniquely covered by this block
        unique_subsets = [c for c in covered_by_removed if counts[c] == 1]
        
        if not unique_subsets:
            return None  # Block is redundant, can just remove
        
        # Find candidate blocks that cover all unique subsets
        # Start with blocks covering the first unique subset
        candidates = set(blocks_covering_subset(unique_subsets[0]))
        candidates.discard(b_idx_to_remove)
        candidates -= current_blocks_set
        
        for sub in unique_subsets[1:]:
            if not candidates:
                break
            candidates &= set(blocks_covering_subset(sub))
            candidates -= current_blocks_set
        
        if candidates:
            return int(next(iter(candidates)))
        return None
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        current_blocks_set = set(selected_block_indices)
        
        # Try 1-for-0 (remove redundant blocks)
        to_remove = []
        for b_idx in selected_block_indices:
            if is_redundant(b_idx, cover_counts):
                to_remove.append(b_idx)
        
        for b_idx in to_remove:
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            cover_counts[covered] -= 1
            selected_block_indices.remove(b_idx)
            current_blocks_set.discard(b_idx)
            improved = True
        
        if not selected_block_indices:
            break
        
        # Try 1-for-1 exchanges
        for i, b_idx in enumerate(list(selected_block_indices)):
            if time.time() > deadline - 10.0:
                break
            
            # Temporarily remove this block
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            cover_counts[covered] -= 1
            current_blocks_set.discard(b_idx)
            
            # Find subsets now uncovered
            uncovered = [c for c in covered if cover_counts[c] == 0]
            
            if not uncovered:
                # Block was redundant, keep it removed
                improved = True
                continue
            
            # Find a block that covers all uncovered subsets
            candidates = set(blocks_covering_subset(uncovered[0]))
            candidates.discard(b_idx)
            candidates -= current_blocks_set
            
            for sub in uncovered[1:]:
                if not candidates:
                    break
                candidates &= set(blocks_covering_subset(sub))
                candidates -= current_blocks_set
            
            if candidates:
                new_b = int(next(iter(candidates)))
                # Verify: does this block cover all previously uncovered?
                new_covered = A.indices[A.indptr[new_b]:A.indptr[new_b+1]]
                all_covered = all(cover_counts[c] > 0 or c in new_covered for c in covered)
                if all_covered:
                    # Accept the exchange
                    new_covered_arr = set(new_covered)
                    for c in new_covered:
                        cover_counts[c] += 1
                    selected_block_indices[i] = new_b
                    current_blocks_set.add(new_b)
                    improved = True
                    continue
            
            # Restore
            cover_counts[covered] += 1
            current_blocks_set.add(b_idx)
        
        # Try 1-for-2: remove one block, add two blocks that together cover everything
        # This is more expensive, so we do it less frequently
        if time.time() < deadline - 20.0 and len(selected_block_indices) > 5:
            for i, b_idx in enumerate(list(selected_block_indices)):
                if time.time() > deadline - 15.0:
                    break
                
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
                current_blocks_set.discard(b_idx)
                
                uncovered = [c for c in covered if cover_counts[c] == 0]
                
                if not uncovered:
                    improved = True
                    continue
                
                # Try to find two blocks that together cover all uncovered
                # Use the first uncovered subset to generate candidates
                cand1_list = list(blocks_covering_subset(uncovered[0]))
                cand1_list = [c for c in cand1_list if c != b_idx and c not in current_blocks_set]
                
                found_pair = False
                for c1 in cand1_list[:50]:  # Limit candidates
                    if time.time() > deadline - 15.0:
                        break
                    
                    # What's still uncovered after adding c1?
                    c1_covered = set(A.indices[A.indptr[c1]:A.indptr[c1+1]])
                    still_uncovered = [c for c in uncovered if c not in c1_covered and cover_counts[c] == 0]
                    
                    if not still_uncovered:
                        # c1 alone covers all, so 1-for-1 would have worked
                        continue
                    
                    # Find c2 that covers all still_uncovered
                    cand2_list = list(blocks_covering_subset(still_uncovered[0]))
                    cand2_list = [c for c in cand2_list if c != b_idx and c != c1 and c not in current_blocks_set]
                    
                    for c2 in cand2_list[:50]:
                        c2_covered = set(A.indices[A.indptr[c2]:A.indptr[c2+1]])
                        all_covered = all(c in c1_covered or c in c2_covered or cover_counts[c] > 0 for c in covered)
                        if all_covered:
                            # Accept: remove b_idx, add c1 and c2
                            for c in A.indices[A.indptr[c1]:A.indptr[c1+1]]:
                                cover_counts[c] += 1
                            for c in A.indices[A.indptr[c2]:A.indptr[c2+1]]:
                                cover_counts[c] += 1
                            selected_block_indices[i] = c1
                            selected_block_indices.append(c2)
                            current_blocks_set.add(c1)
                            current_blocks_set.add(c2)
                            improved = True
                            found_pair = True
                            break
                    
                    if found_pair:
                        break
                
                if not found_pair:
                    # Restore
                    cover_counts[covered] += 1
                    current_blocks_set.add(b_idx)
        
        # Try 2-for-1: remove two blocks, add one block
        if time.time() < deadline - 15.0 and len(selected_block_indices) > 10:
            for i in range(len(selected_block_indices)):
                if time.time() > deadline - 15.0:
                    break
                for j in range(i + 1, len(selected_block_indices)):
                    if time.time() > deadline - 15.0:
                        break
                    
                    b1 = selected_block_indices[i]
                    b2 = selected_block_indices[j]
                    
                    cov1 = A.indices[A.indptr[b1]:A.indptr[b1+1]]
                    cov2 = A.indices[A.indptr[b2]:A.indptr[b2+1]]
                    
                    cover_counts[cov1] -= 1
                    cover_counts[cov2] -= 1
                    current_blocks_set.discard(b1)
                    current_blocks_set.discard(b2)
                    
                    # Find all subsets that are now uncovered
                    all_cov = set(cov1) | set(cov2)
                    uncovered = [c for c in all_cov if cover_counts[c] == 0]
                    
                    if not uncovered:
                        # Both were redundant
                        selected_block_indices.pop(j)
                        selected_block_indices.pop(i)
                        improved = True
                        continue
                    
                    # Find a single block that covers all uncovered
                    candidates = set(blocks_covering_subset(uncovered[0]))
                    candidates.discard(b1)
                    candidates.discard(b2)
                    candidates -= current_blocks_set
                    
                    for sub in uncovered[1:]:
                        if not candidates:
                            break
                        candidates &= set(blocks_covering_subset(sub))
                        candidates -= current_blocks_set
                    
                    if candidates:
                        new_b = int(next(iter(candidates)))
                        new_covered = set(A.indices[A.indptr[new_b]:A.indptr[new_b+1]])
                        all_covered = all(c in new_covered or cover_counts[c] > 0 for c in all_cov)
                        if all_covered:
                            for c in A.indices[A.indptr[new_b]:A.indptr[new_b+1]]:
                                cover_counts[c] += 1
                            selected_block_indices[i] = new_b
                            selected_block_indices.pop(j)
                            current_blocks_set.add(new_b)
                            improved = True
                            continue
                    
                    # Restore
                    cover_counts[cov1] += 1
                    cover_counts[cov2] += 1
                    current_blocks_set.add(b1)
                    current_blocks_set.add(b2)
    
    # Final cleanup: remove any redundant blocks
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
    
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        to_remove = []
        for b_idx in selected_block_indices:
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(b_idx)
        
        if to_remove:
            for b_idx in to_remove:
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
                selected_block_indices.remove(b_idx)
            changed = True
    
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
    return blocks
