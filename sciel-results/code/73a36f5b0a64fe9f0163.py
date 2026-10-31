import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Build coverage matrix: n_ksubs x n_tsubs (uint8), where M[j, i] = 1 if k-sub j covers t-sub i
    cov_matrix = np.zeros((n_ksubs, n_tsubs), dtype=np.uint8)
    for j, ks in enumerate(all_ksubs):
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            cov_matrix[j, idx] = 1

    # Precompute row sums (total coverage of each block)
    total_cov = cov_matrix.sum(axis=1)

    def greedy_cover(seed, max_time):
        """Run greedy set cover with randomization for tie-breaking."""
        rng = np.random.RandomState(seed)
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        remaining = n_tsubs
        used = np.zeros(n_ksubs, dtype=bool)
        selected = []
        t_start = time.time()

        while remaining > 0 and time.time() < max_time:
            # Vectorized gain computation
            gains = (cov_matrix & uncovered[None, :]).sum(axis=1)
            gains[used] = 0

            # Find max gain
            max_gain = gains.max()
            if max_gain == 0:
                break

            # Random tie-breaking
            candidates = np.where(gains >= max_gain)[0]
            chosen_idx = candidates[rng.randint(len(candidates))]

            selected.append(chosen_idx)
            used[chosen_idx] = True
            uncovered = uncovered & (1 - cov_matrix[chosen_idx])
            remaining = uncovered.sum()

        return selected

    def remove_redundant(selected, max_time):
        """Remove redundant blocks from the selected set."""
        if not selected:
            return selected

        active = list(selected)
        t_start = time.time()
        changed = True

        while changed and time.time() < max_time:
            changed = False
            new_active = []
            # Precompute union of all blocks to speed up check?
            # For each block, check if it is subset of union of others.
            # Union of others = Total Union - (Block - (Total Union - Block)) ? No.
            # Let U be union of all active.
            # Block i is redundant if U \ {Block i} == U.
            # i.e., Block i is subset of U \ {Block i}.
            
            # Efficient way:
            # 1. Compute total union mask.
            # 2. For each block i, compute mask of t-subs covered ONLY by i (within active set).
            # 3. If mask is empty, i is redundant.
            
            if len(active) <= 1:
                return active

            total_union = np.zeros(n_tsubs, dtype=np.uint8)
            for bi in active:
                total_union |= cov_matrix[bi]

            new_active = []
            for i, bi in enumerate(active):
                # Mask of t-subs covered by bi
                mask_bi = cov_matrix[bi]
                
                # Mask of t-subs covered by bi AND NOT covered by any other active block
                # This is mask_bi & ~(total_union - mask_bi) ? No.
                # total_union includes bi.
                # Let's compute union of others.
                # This is O(N^2) per pass.
                
                # Optimization:
                # If we iterate, we can maintain a count of how many blocks cover each t-sub.
                # But active set changes.
                
                # Given N is small (~150), O(N^2) is 22500 ops per pass.
                # Each op is bitwise OR on 3876 bytes. ~22500 * 4KB = 90MB of memory traffic.
                # This is fast in NumPy?
                # Let's just do it simply.
                
                other_union = np.zeros(n_tsubs, dtype=np.uint8)
                for j, bj in enumerate(active):
                    if j != i:
                        other_union |= cov_matrix[bj]
                
                # Check if bi is subset of other_union
                # i.e., (bi & ~other_union) is empty
                if (mask_bi & ~other_union).sum() == 0:
                    changed = True
                else:
                    new_active.append(bi)

            active = new_active

        return active

    def local_search_swap(selected, max_time):
        """
        Try to swap blocks to reduce overlap or enable further reduction.
        Strategy: For each block b, if it has 'unique' coverage (t-subs covered only by b),
        find a block b' that covers those unique t-subs and has lower overlap with the rest.
        """
        if not selected:
            return selected

        active = list(selected)
        t_start = time.time()
        
        # We will iterate a few times
        for _ in range(10): # Limit iterations
            if time.time() > max_time:
                break
            
            # 1. Compute total union and unique masks
            total_union = np.zeros(n_tsubs, dtype=np.uint8)
            for bi in active:
                total_union |= cov_matrix[bi]
            
            # Check for redundancy first (quick win)
            new_active = []
            for i, bi in enumerate(active):
                mask_bi = cov_matrix[bi]
                # Unique part
                # We need union of others
                # To save time, we only do full redundancy check if we didn't swap in this iteration?
                # Let's just do the swap logic.
                
                # Find unique t-subs for bi
                # unique_mask = mask_bi & ~(total_union - mask_bi)
                # Note: total_union - mask_bi is not a simple subtraction because of overlaps.
                # We need: t-subs covered by bi AND NOT covered by any other block in active.
                
                # This requires O(N) work per block to compute union of others.
                # Let's precompute "coverage count" for each t-sub in active set.
                pass
            
            # Simpler Swap Strategy:
            # For each block b in active:
            #   1. Calculate mask of t-subs covered ONLY by b (within active).
            #   2. If mask is not empty:
            #      a. Find candidates b' (not in active) that cover this mask.
            #      b. If found, pick b' with min overlap with (total_union - b).
            #      c. Swap b with b'.
            
            # To make this efficient, we compute unique masks for all blocks at once?
            # Hard to do in vector.
            
            # Let's just loop. N is small.
            improved = False
            new_active = list(active)
            
            # Compute total union once
            total_union = np.zeros(n_tsubs, dtype=np.uint8)
            for bi in active:
                total_union |= cov_matrix[bi]
                
            # For each block, find unique part
            # unique_mask[i] = cov_matrix[bi] & ~(Union of others)
            # Union of others = total_union - (cov_matrix[bi] - (cov_matrix[bi] & Union of others))
            # This is tricky.
            
            # Let's just compute Union of others for each block.
            for i, bi in enumerate(active):
                if time.time() > max_time:
                    break
                
                mask_bi = cov_matrix[bi]
                
                # Calculate union of others
                other_union = np.zeros(n_tsubs, dtype=np.uint8)
                for j, bj in enumerate(active):
                    if j != i:
                        other_union |= cov_matrix[bj]
                
                unique_mask = mask_bi & ~other_union
                unique_count = unique_mask.sum()
                
                if unique_count == 0:
                    # Redundant, remove it
                    new_active = [b for j, b in enumerate(new_active) if j != i]
                    improved = True
                    break # Restart loop with new set
            
            if improved:
                active = new_active
                continue
            
            # If no redundancy, try swaps
            # We need to be careful not to make it worse.
            # We'll try to swap a block that has high "waste" (overlap with others) 
            # for one that has less waste, provided it covers the unique parts.
            
            # Calculate "waste" for each block: overlap with total_union - mask_bi
            # Actually, let's just try to swap a random block with a candidate that covers its unique parts.
            
            # Pick a random block to try to swap
            if not active:
                break
                
            idx_to_swap = np.random.randint(len(active))
            bi = active[idx_to_swap]
            mask_bi = cov_matrix[bi]
            
            # Recalculate unique mask for this specific bi
            other_union = np.zeros(n_tsubs, dtype=np.uint8)
            for j, bj in enumerate(active):
                if j != idx_to_swap:
                    other_union |= cov_matrix[bj]
            
            unique_mask = mask_bi & ~other_union
            unique_count = unique_mask.sum()
            
            if unique_count == 0:
                # Should have been caught by redundancy check
                continue
            
            # Find candidates b' that cover unique_mask
            # Condition: (cov_matrix[b'] & unique_mask) == unique_mask
            # This is equivalent to: cov_matrix[b'] covers all 1s in unique_mask
            
            # Vectorized check:
            # intersection = cov_matrix & unique_mask[None, :]
            # We want rows where sum(intersection) == unique_count
            # AND the block is not in active.
            
            # This is expensive: 50388 x 3876 matrix multiplication-ish.
            # Let's optimize:
            # Only check if unique_count is small?
            
            # For now, let's just do a simple random swap if we find ANY candidate.
            # To speed up, we can pre-filter candidates based on total_cov?
            
            # Let's just try a few random candidates from the pool of unused blocks.
            # This is less systematic but faster.
            
            # Actually, let's stick to the redundancy removal which is already strong.
            # The "Swap" logic is complex to implement efficiently in Python without C extensions.
            # Given the time limit, I will focus on **Multiple Restarts** with **Diversity-Aware Greedy**.
            
            break # Exit swap loop for now, relying on restarts

        return active

    # Main search
    best_blocks = None
    best_count = float('inf')

    # 1. Quick Greedy
    t0 = time.time()
    quick_selected = greedy_cover(0, deadline - 10.0)
    t1 = time.time()
    
    if quick_selected:
        quick_active = remove_redundant(quick_selected, deadline - 5.0)
        if quick_active:
            best_blocks = [list(all_ksubs[idx]) for idx in quick_active]
            best_count = len(quick_active)

    # 2. Additional Restarts
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        n_restarts = min(100, int(remaining_time / 1.0)) # Estimate 1s per restart
        for r in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            seed = r + 1
            # Use a small time budget for greedy to allow many restarts
            # Greedy is fast, so we can do many.
            time_budget = min(deadline - time.time() - 2.0, 2.0)
            if time_budget < 1.0:
                break

            selected = greedy_cover(seed, time_budget)
            if not selected:
                continue

            active = remove_redundant(selected, deadline - 2.0)
            if active and len(active) < best_count:
                best_count = len(active)
                best_blocks = [list(all_ksubs[idx]) for idx in active]
                
                # If we hit the lower bound or a very good number, we can stop or continue
                if best_count <= 151:
                    # Try to polish further?
                    pass

    if best_blocks is None:
        best_blocks = [list(all_ksubs[idx]) for idx in quick_selected]

    return best_blocks
