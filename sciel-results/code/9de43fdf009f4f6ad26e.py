import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 10s for cleanup/return
    deadline = start_time + 110
    
    # 1. Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # 2. Generate candidate k-subsets and their masks
    candidates_mask = []
    candidates_block = []
    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        candidates_mask.append(mask)
        candidates_block.append(block)
        
    n_candidates = len(candidates_mask)
    
    # 3. Precompute coverers: for each t-subset, which candidates cover it?
    # This is useful for the repair phase.
    # coverers[t_idx] = list of candidate indices
    coverers = [[] for _ in range(n_t)]
    for i, mask in enumerate(candidates_mask):
        # Iterate set bits in mask
        m = mask
        while m:
            # Get lowest set bit index
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverers[idx].append(i)
            m ^= lsb

    uncovered = (1 << n_t) - 1
    used = [False] * n_candidates
    selected_indices = []
    
    # Phase 1: Global Greedy
    # We want to find the candidate that maximizes new coverage.
    while uncovered and time.time() < deadline - 10:
        best_gain = -1
        best_idx = -1
        
        # Scan all candidates
        for i in range(n_candidates):
            if used[i]:
                continue
            mask = candidates_mask[i]
            # Quick check
            if (mask & uncovered) == 0:
                continue
            gain = (mask & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == n_t:
                    break
        
        if best_idx == -1:
            # No candidate covers any uncovered subset?
            # This implies the problem is infeasible or we ran out of candidates.
            # But we have all k-subsets, so it must be feasible.
            # Break to let repair phase handle it (though repair should also fail if truly infeasible)
            break
            
        used[best_idx] = True
        selected_indices.append(best_idx)
        uncovered &= ~candidates_mask[best_idx]
        
    # Phase 2: Repair
    # If there are still uncovered subsets, cover them one by one with any available block.
    if uncovered:
        # Iterate through all t-subset indices
        for idx in range(n_t):
            if time.time() > deadline - 2:
                break
            # Check if this subset is uncovered
            if (uncovered >> idx) & 1:
                # Find an available candidate that covers this subset
                found = False
                for cand_idx in coverers[idx]:
                    if not used[cand_idx]:
                        used[cand_idx] = True
                        selected_indices.append(cand_idx)
                        uncovered &= ~candidates_mask[cand_idx]
                        found = True
                        break
                if not found:
                    # Should not happen
                    pass

    # Phase 3: Local Search - Remove Redundant Blocks
    # We try to remove any block that doesn't contribute unique coverage.
    # To check if block i is redundant, we check if its mask is a subset of the union of all other selected blocks.
    
    # Optimization: 
    # Instead of recomputing the union of all others from scratch for each i (O(N^2)),
    # we can do a single pass.
    # Let U be the union of all selected masks.
    # For each block i, if (mask_i & ~U) == 0, it's redundant? No.
    # U includes mask_i.
    # We need U_others.
    
    # Given N is small (~350), O(N^2) is fine.
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        # We iterate backwards to delete safely, or use a flag
        for i in range(len(selected_indices)):
            if time.time() > deadline - 1:
                break
            
            idx_i = selected_indices[i]
            mask_i = candidates_mask[idx_i]
            
            # Calculate union of all other selected blocks
            union_others = 0
            for j in range(len(selected_indices)):
                if i == j:
                    continue
                idx_j = selected_indices[j]
                union_others |= candidates_mask[idx_j]
                
            # If mask_i is subset of union_others, it is redundant
            if (mask_i & ~union_others) == 0:
                # Remove block i
                del selected_indices[i]
                improved = True
                break # Restart loop because indices shifted
    
    # Construct final list of blocks
    blocks = [list(candidates_block[i]) for i in selected_indices]
    
    return blocks
