import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute covered t-subsets for each block
    block_t_subsets = []
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        block_t_subsets.append(subs)
    
    # Combinatorial Inclusion-Exclusion Pruning:
    # Remove blocks whose coverage is dominated by another block (superset coverage)
    # Two blocks B1, B2 where B1's t-subsets are all also in B2 means B1 is dominated
    # For 7-subsets of 19 points, two blocks share t-subsets if they share at least t points
    
    # Build a mapping from each t-subset to the list of blocks that contain it
    t_sub_to_blocks = {}
    for i, subs in enumerate(block_t_subsets):
        for t_sub in subs:
            if t_sub not in t_sub_to_blocks:
                t_sub_to_blocks[t_sub] = []
            t_sub_to_blocks[t_sub].append(i)
    
    # For each block, compute how many unique t-subsets it has (not covered by any other block)
    # A block is "essential" if it has at least one unique t-subset
    # But for pruning, we want to remove blocks that are completely redundant
    # Block B1 is redundant if every t-subset of B1 is also in some other block
    
    # More efficient: use pairwise overlap to identify dominated blocks
    # For two k-subsets B1 and B2, if |B1 ∩ B2| >= k-1, they share many t-subsets
    # Specifically, if B1 ⊂ B2 (impossible for same size) or they share k-1 elements,
    # they share C(k-1, t) t-subsets
    
    # Key insight: Two 7-subsets that share 6 elements share C(6,4)=15 of their 35 t-subsets
    # Two 7-subsets that share 7 elements are identical
    
    # Build intersection graph: for each block, find blocks that share k-1 elements
    # These are the only candidates that could dominate it
    
    # For efficiency, use a dictionary mapping (point, block) -> block
    # Then for each pair of blocks sharing k-1 points, check dominance
    
    # Actually, simpler: just check if any block's coverage set is a subset of another's
    # Use bitset representation for fast subset checks
    
    # Convert each block's t-subset coverage to a frozenset for quick subset checks
    block_coverage_sets = []
    for subs in block_t_subsets:
        block_coverage_sets.append(frozenset(subs))
    
    # Identify dominated blocks: B1 is dominated by B2 if coverage(B1) ⊆ coverage(B2)
    # For same-size blocks, this only happens if they're identical (which can't happen)
    # OR if B1's t-subsets are all covered by B2, which requires high overlap
    
    # For 7-subsets with t=4: B1's t-subsets ⊆ B2's t-subsets iff B1 ⊆ B2
    # But B1 and B2 have same size, so B1 ⊆ B2 iff B1 = B2
    # So no block is dominated by another single block!
    
    # Instead, use "2-cover dominance": B1 is redundant if every t-subset of B1
    # is covered by the union of two other blocks
    # This is more complex but can still prune
    
    # Simpler pruning: remove blocks that are "rare" - share few t-subsets with others
    # Actually, let's just proceed with greedy but use better tie-breaking
    
    # Build sparse matrix
    row_indices = []
    col_indices = []
    data = []
    
    for i, subs in enumerate(block_t_subsets):
        for t_sub in subs:
            j = t_subset_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    
    active = np.ones(num_t_subsets, dtype=np.float64)
    selected_block_indices = []
    
    # Greedy with improved tie-breaking using secondary gain
    while True:
        if time.time() > deadline - 20.0:
            break
            
        gains = A.dot(active)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Find all blocks with maximum gain
        max_indices = np.where(gains == max_gain)[0]
        
        if len(max_indices) == 1:
            b_idx = max_indices[0]
        else:
            # Among max-gain blocks, use secondary criteria:
            # 1. Prefer blocks that cover more "rare" t-subsets (low total coverage)
            # 2. Then prefer blocks with higher average overlap with already selected blocks
            #    (to maximize future redundancy for deletion)
            
            # Compute rarity score: sum of 1/total_coverage for each t-subset covered
            # Precompute total coverage count for each t-subset
            total_coverage = np.array(A.sum(axis=0)).flatten()
            total_coverage[total_coverage == 0] = 1  # avoid division by zero
            
            rarity_scores = np.zeros(len(max_indices))
            for idx, b_idx_candidate in enumerate(max_indices):
                covered = A.indices[A.indptr[b_idx_candidate]:A.indptr[b_idx_candidate+1]]
                rarity_scores[idx] = np.sum(1.0 / total_coverage[covered])
            
            # Pick the block with highest rarity score
            best_local_idx = np.argmax(rarity_scores)
            b_idx = max_indices[best_local_idx]
        
        selected_block_indices.append(b_idx)
        
        covered_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        active[covered_indices] = 0
        
        if np.all(active == 0):
            break
            
    # Local Search: Deletion
    # Maintain cover counts
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
        
    # Try to remove blocks - multiple passes
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            # Check if all t-subsets of this block are covered by others
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            # Remove them
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
        else:
            break
    
    # Swap-based local search with actual replacement attempts
    if time.time() < deadline - 8.0:
        # Build a set of selected blocks for fast lookup
        selected_set = set(selected_block_indices)
        
        # Try to find replacements that reduce total count
        for iteration in range(3):
            if time.time() > deadline - 8.0:
                break
            
            improved = False
            n_sel = len(selected_block_indices)
            
            for i in range(n_sel):
                if time.time() > deadline - 8.0:
                    break
                
                b_old = selected_block_indices[i]
                covered_old = A.indices[A.indptr[b_old]:A.indptr[b_old+1]]
                
                # Find t-subsets that are only covered by this block
                unique_to_old = []
                for c in covered_old:
                    if cover_counts[c] == 1:
                        unique_to_old.append(c)
                
                if len(unique_to_old) == 0:
                    # This block is redundant, remove it
                    for c in covered_old:
                        cover_counts[c] -= 1
                    selected_block_indices.pop(i)
                    selected_set.discard(b_old)
                    improved = True
                    break
                
                # Try to find a replacement block that covers all unique_to_old
                # and is not already selected
                best_new = -1
                best_extra_gain = -1
                
                # Only check blocks that could potentially cover the unique subsets
                # For efficiency, sample a subset of candidate blocks
                candidates = []
                for c in unique_to_old:
                    blocks_covering_c = t_sub_to_blocks.get(t_sub, [])
                    for b in blocks_covering_c:
                        if b not in selected_set and b != b_old:
                            candidates.append(b)
                
                # Deduplicate and limit candidates
                candidates = list(set(candidates))
                if len(candidates) > 50:
                    # Sample 50 random candidates
                    import random
                    candidates = random.sample(candidates, 50)
                
                for b_new in candidates:
                    covered_new = A.indices[A.indptr[b_new]:A.indptr[b_new+1]]
                    # Check if new block covers all unique_to_old
                    covers_all = True
                    for c in unique_to_old:
                        if c not in covered_new:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Score: how many additional subsets does it cover that are currently covered?
                    # We want to maximize overlap with already-covered subsets to enable future deletions
                    extra_covered = 0
                    for c in covered_new:
                        if c not in covered_old and cover_counts[c] > 0:
                            extra_covered += 1
                    
                    if extra_covered > best_extra_gain:
                        best_extra_gain = extra_covered
                        best_new = b_new
                
                if best_new != -1:
                    # Perform the swap
                    # Remove old block
                    for c in covered_old:
                        cover_counts[c] -= 1
                    selected_block_indices.pop(i)
                    selected_set.discard(b_old)
                    
                    # Add new block
                    covered_new = A.indices[A.indptr[best_new]:A.indptr[best_new+1]]
                    for c in covered_new:
                        cover_counts[c] += 1
                    selected_block_indices.append(best_new)
                    selected_set.add(best_new)
                    
                    improved = True
                    break
            
            if not improved:
                break
        
        # Rebuild cover counts after swaps
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected_block_indices:
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            cover_counts[covered] += 1
    
    # Final aggressive deletion pass with pair removal
    for _ in range(3):
        if time.time() > deadline - 3.0:
            break
        
        # Single block deletion
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
        
        # Pair deletion: try to remove two blocks at once
        n_sel = len(selected_block_indices)
        pair_removed = False
        for i in range(n_sel):
            if time.time() > deadline - 3.0:
                break
            for j in range(i+1, n_sel):
                if time.time() > deadline - 3.0:
                    break
                
                b1 = selected_block_indices[i]
                b2 = selected_block_indices[j]
                covered1 = A.indices[A.indptr[b1]:A.indptr[b1+1]]
                covered2 = A.indices[A.indptr[b2]:A.indptr[b2+1]]
                
                # Check if removing both is safe
                can_remove = True
                for c in covered1:
                    if cover_counts[c] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for c in covered2:
                        if cover_counts[c] <= 1:
                            can_remove = False
                            break
                
                if can_remove:
                    for c in covered1:
                        cover_counts[c] -= 1
                    for c in covered2:
                        cover_counts[c] -= 1
                    selected_block_indices.pop(j)
                    selected_block_indices.pop(i)
                    pair_removed = True
                    break
            if pair_removed:
                break
    
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
    
    # Verify validity before returning
    if len(blocks) > 0:
        covered_all = np.zeros(num_t_subsets, dtype=np.bool_)
        for b_idx in selected_block_indices:
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            covered_all[covered] = True
        
        if not np.all(covered_all):
            # Invalid - should not happen, but add missing coverage with greedy
            missing = np.where(~covered_all)[0]
            for m_idx in missing:
                # Find a block that covers this t-subset
                for b_idx in range(num_blocks):
                    covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                    if m_idx in covered and b_idx not in selected_set:
                        selected_block_indices.append(b_idx)
                        selected_set.add(b_idx)
                        break
    
    blocks = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
    return blocks
