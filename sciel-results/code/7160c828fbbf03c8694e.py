# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin under 120s
    rng = random.Random(98765)

    # ---------- enumerate all t-subsets and all k-subsets as bitmasks ----------
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    n_tsub = len(tsub_masks)

    ksub_masks = []
    block_points_indices = []
    for comb in itertools.combinations(range(v), k):
        m = 0
        for p in comb:
            m |= (1 << p)
        ksub_masks.append(m)
        block_points_indices.append(comb)
    n_ksub = len(ksub_masks)
    
    # Convert block_points_indices to numpy array for vectorized cost calculation
    # Shape: (n_ksub, k)
    block_pts_arr = np.array(block_points_indices, dtype=np.int32)

    # map t-subset mask -> index
    idx_of_tsub = {m: i for i, m in enumerate(tsub_masks)}

    # ---------- precompute: for each k-subset, the indices of t-subsets it covers ----------
    c_k_t = len(list(itertools.combinations(range(k), t)))
    cover_matrix = np.zeros((n_ksub, c_k_t), dtype=np.int32)
    
    for j, m in enumerate(ksub_masks):
        pts = block_points_indices[j]
        idxs = []
        for sub in itertools.combinations(pts, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            idxs.append(idx_of_tsub[sm])
        cover_matrix[j, :] = idxs

    # ---------- state ----------
    uncovered = np.ones(n_tsub, dtype=bool)  # True = not yet covered
    chosen = []          # list of ksub indices chosen
    chosen_set = set()   # for O(1) membership
    point_count = np.zeros(v, dtype=np.int32) # frequency of each point in chosen blocks

    def deadline_hit():
        return time.time() > deadline

    # ---------- main greedy loop ----------
    # Alpha controls the penalty for using points that are already frequently used.
    # Higher alpha -> more diversity.
    alpha = 0.5 
    
    while True:
        if deadline_hit():
            break
        
        # Compute raw gain for all ksubs
        # np.take(uncovered, cover_matrix) -> (n_ksub, c_k_t) boolean array
        # sum along axis 1 -> (n_ksub,)
        raw_gains = np.take(uncovered, cover_matrix).sum(axis=1)
        
        # Compute point cost for all ksubs
        # point_count is (v,), block_pts_arr is (n_ksub, k)
        # point_count[block_pts_arr] -> (n_ksub, k)
        # sum(axis=1) -> (n_ksub,)
        point_costs = point_count[block_pts_arr].sum(axis=1)
        
        # Effective gain
        effective_gains = raw_gains - alpha * point_costs
        
        # Exclude already chosen blocks
        for j in chosen_set:
            effective_gains[j] = -1
            raw_gains[j] = -1 # Just in case, though we use effective
            
        max_eff_gain = int(np.max(effective_gains))
        if max_eff_gain <= 0:
            break
        
        # Find all candidates with max effective gain
        candidates = np.where(effective_gains == max_eff_gain)[0]
        
        if len(candidates) == 1:
            best_j = int(candidates[0])
        else:
            # Tie-break: prefer ksub with fewer points already in chosen blocks (diversity)
            # We already have point_costs which is the sum of counts.
            # A better tie-break might be the max count of any single point in the block?
            # Or just the sum (which is point_costs).
            # Since effective_gains are tied, point_costs are likely similar.
            # Let's use the raw_gain as secondary? No, raw_gain is part of effective.
            # Let's use a random tie-breaker to avoid deterministic loops, or min max-point-count.
            
            # Calculate max point count in the block for each candidate
            # cand_pts = block_pts_arr[candidates] -> (n_cand, k)
            # cand_counts = point_count[cand_pts] -> (n_cand, k)
            # max_counts = cand_counts.max(axis=1)
            
            cand_pts = block_pts_arr[candidates]
            cand_counts = point_count[cand_pts]
            max_counts = cand_counts.max(axis=1)
            
            # Prefer lower max count (most "even" usage)
            best_local = int(np.argmin(max_counts))
            best_j = int(candidates[best_local])

        # Commit best_j
        chosen_set.add(best_j)
        chosen.append(best_j)
        
        # Update uncovered: zero out tsubsets covered by best_j
        uncovered[cover_matrix[best_j, :]] = False
        
        # Update point counts
        point_count[block_pts_arr[best_j, :]] += 1
        
        # Optional: Decay alpha?
        # alpha *= 0.99

    # ---------- REPAIR ----------
    # For each uncovered t-subset, add a block covering it
    uncovered_idx = np.where(uncovered)[0]
    repair_blocks = []
    for i in uncovered_idx:
        sm = tsub_masks[i]
        have = [p for p in range(v) if (sm >> p) & 1]
        need = k - len(have)
        extra = []
        for p in range(v):
            if need <= 0:
                break
            if not ((sm >> p) & 1):
                extra.append(p)
                need -= 1
        block = have + extra
        block.sort()
        mask = 0
        for p in block:
            mask |= (1 << p)
        repair_blocks.append(mask)
        
    # Convert to result
    result_masks = set()
    for j in chosen:
        result_masks.add(ksub_masks[j])
        
    for mask in repair_blocks:
        result_masks.add(mask)

    # ---------- POLISH: Remove redundant blocks ----------
    # Re-verify coverage and remove blocks that are not needed.
    # This is O(N_blocks * N_tsubs) if done naively, but N_blocks is small (~150).
    # We can check each block: if all its t-subsets are covered by OTHER blocks, remove it.
    
    # Build a set of current blocks
    current_blocks = list(result_masks)
    n_current = len(current_blocks)
    
    # Precompute t-subsets for each current block
    block_tsubs = []
    for mask in current_blocks:
        pts = [p for p in range(v) if (mask >> p) & 1]
        subs = set()
        for sub in itertools.combinations(pts, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            subs.add(sm)
        block_tsubs.append(subs)
        
    # Count how many blocks cover each t-subset
    # We only care about t-subsets covered by at least one block in current_blocks.
    # Since the set is valid, all tsub_masks are covered.
    # We can just iterate and check.
    
    to_remove = []
    for i in range(n_current):
        if deadline_hit():
            break
        # Check if block i is redundant
        redundant = True
        for sm in block_tsubs[i]:
            # Check if any other block j != i covers sm
            covered_by_other = False
            for j in range(n_current):
                if i == j:
                    continue
                if sm in block_tsubs[j]:
                    covered_by_other = True
                    break
            if not covered_by_other:
                redundant = False
                break
        if redundant:
            to_remove.append(i)
            
    # Remove redundant blocks
    if to_remove:
        remove_set = set(to_remove)
        new_blocks = [b for i, b in enumerate(current_blocks) if i not in remove_set]
        result_masks = set(new_blocks)

    # Convert to lists
    result = []
    for mask in result_masks:
        pts = [p for p in range(v) if (mask >> p) & 1]
        pts.sort()
        result.append(pts)

    return result
# EVOLVE-BLOCK-END
