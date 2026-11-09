"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 50.0  # safety margin before 60s kill (90s wall-clock)
    
    def elapsed():
        return time.time() - start_time
        
    N_SMALL_END = 128
    N_LARGE_END = 256
    
    small_nums = list(range(2, N_SMALL_END + 1))
    large_nums = list(range(N_SMALL_END + 1, N_LARGE_END + 1))
    
    n_small = len(small_nums)
    n_large = len(large_nums)
    
    # Weights
    small_weights = [1.0 / x for x in small_nums]
    large_weights = [1.0 / x for x in large_nums]
    
    # Precompute conflict masks
    # small_conflict_mask[i] : bitmask of small indices conflicting with i
    small_conflict_mask = [0] * n_small
    for i in range(n_small):
        for j in range(i + 1, n_small):
            a, b = small_nums[i], small_nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N_LARGE_END:
                small_conflict_mask[i] |= (1 << j)
                small_conflict_mask[j] |= (1 << i)
                
    # large_conflict_mask[i] : bitmask of large indices conflicting with small_nums[i]
    small_blocks_large = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        for j in range(n_large):
            b = large_nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N_LARGE_END:
                small_blocks_large[i] |= (1 << j)

    # large_large_conflict_mask[i] : bitmask of large indices conflicting with large_nums[i]
    large_large_conflict_mask = [0] * n_large
    for i in range(n_large):
        a = large_nums[i]
        for j in range(i + 1, n_large):
            b = large_nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N_LARGE_END:
                large_large_conflict_mask[i] |= (1 << j)
                large_large_conflict_mask[j] |= (1 << i)

    # Precompute sorted large indices by weight (descending)
    large_sorted_indices = sorted(range(n_large), key=lambda j: -large_weights[j])
    
    # Precompute for each large node, the list of large nodes it conflicts with (for quick check)
    large_conflict_neighbors = [0] * n_large
    for i in range(n_large):
        large_conflict_neighbors[i] = large_large_conflict_mask[i]
    
    # Cache for get_large_score to avoid redundant computations
    large_score_cache = {}
    
    def get_large_score(unblocked_mask):
        """Fast greedy with caching."""
        if unblocked_mask in large_score_cache:
            return large_score_cache[unblocked_mask]
        
        selected = 0
        score = 0.0
        for j in large_sorted_indices:
            if not (unblocked_mask & (1 << j)):
                continue
            if large_conflict_neighbors[j] & selected:
                continue
            selected |= (1 << j)
            score += large_weights[j]
        
        result = (score, selected)
        if len(large_score_cache) < 10000:
            large_score_cache[unblocked_mask] = result
        return result
    
    def get_large_score_upper_bound(unblocked_mask):
        # Upper bound: sum of all weights in unblocked_mask (ignores conflicts)
        s = 0.0
        m = unblocked_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += large_weights[idx]
            m ^= lsb
        return s

    def compute_blocked_large(small_mask):
        blocked_large = 0
        m = small_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            blocked_large |= small_blocks_large[idx]
            m ^= lsb
        return blocked_large
    
    # Precompute small weights as a list for fast access
    small_weight_arr = small_weights
    
    def get_total_score(small_mask):
        s_small = 0.0
        m = small_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s_small += small_weight_arr[idx]
            m ^= lsb
            
        # Calculate blocked large mask
        blocked_large = compute_blocked_large(small_mask)
        
        # Unblocked large mask
        all_large_mask = (1 << n_large) - 1
        unblocked_large = all_large_mask & ~blocked_large
        
        s_large, _ = get_large_score(unblocked_large)
        
        return s_small + s_large

    best_score = 0.0
    best_small_mask = 0
    best_large_selection_mask = 0

    def update_best(sm, score):
        nonlocal best_score, best_small_mask, best_large_selection_mask
        if score > best_score:
            best_score = score
            best_small_mask = sm
            # Re-calculate large selection to store it
            blocked_large = compute_blocked_large(sm)
            all_large_mask = (1 << n_large) - 1
            unblocked_large = all_large_mask & ~blocked_large
            _, best_large_selection_mask = get_large_score(unblocked_large)

    # Generate initial candidates
    initial_masks = []
    initial_masks.append(0)
    
    # Greedy by weight
    mask = 0
    for i in range(n_small):
        if not (small_conflict_mask[i] & mask):
            mask |= (1 << i)
    initial_masks.append(mask)

    # Greedy by weight (reverse order)
    mask = 0
    for i in range(n_small - 1, -1, -1):
        if not (small_conflict_mask[i] & mask):
            mask |= (1 << i)
    initial_masks.append(mask)

    # Randomized greedy - more restarts
    num_random_starts = 50
    for _ in range(num_random_starts):
        if elapsed() > DEADLINE - 5.0:
            break
        order = list(range(n_small))
        random.shuffle(order)
        mask = 0
        for i in order:
            if not (small_conflict_mask[i] & mask):
                mask |= (1 << i)
        initial_masks.append(mask)
    
    # Also try greedy with different weight orderings
    for _ in range(10):
        if elapsed() > DEADLINE - 5.0:
            break
        # Perturb weights slightly and do greedy
        perturbed_weights = [small_weights[i] * (0.8 + 0.4 * random.random()) for i in range(n_small)]
        order = sorted(range(n_small), key=lambda i: -perturbed_weights[i])
        mask = 0
        for i in order:
            if not (small_conflict_mask[i] & mask):
                mask |= (1 << i)
        initial_masks.append(mask)

    # Local Search with incremental blocked mask tracking
    for init_mask in initial_masks:
        if elapsed() > DEADLINE:
            break
            
        current_mask = init_mask
        current_score = get_total_score(current_mask)
        update_best(current_mask, current_score)
        
        # Precompute blocked_large for current mask
        current_blocked_large = compute_blocked_large(current_mask)
        
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            best_move_score = current_score
            best_new_mask = current_mask
            best_new_blocked_large = current_blocked_large
            
            # Try Add
            for i in range(n_small):
                if current_mask & (1 << i):
                    continue
                if small_conflict_mask[i] & current_mask:
                    continue
                
                new_mask = current_mask | (1 << i)
                # Incrementally update blocked_large
                new_blocked_large = current_blocked_large | small_blocks_large[i]
                
                all_large_mask = (1 << n_large) - 1
                unblocked_large = all_large_mask & ~new_blocked_large
                s_large, _ = get_large_score(unblocked_large)
                s_small = 0.0
                m = new_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    s_small += small_weight_arr[idx]
                    m ^= lsb
                new_score = s_small + s_large
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_new_mask = new_mask
                    best_new_blocked_large = new_blocked_large
            
            # Try Remove
            for i in range(n_small):
                if not (current_mask & (1 << i)):
                    continue
                
                new_mask = current_mask & ~(1 << i)
                # Recompute blocked_large (can't easily do incrementally for removal)
                new_blocked_large = compute_blocked_large(new_mask)
                
                all_large_mask = (1 << n_large) - 1
                unblocked_large = all_large_mask & ~new_blocked_large
                s_large, _ = get_large_score(unblocked_large)
                s_small = 0.0
                m = new_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    s_small += small_weight_arr[idx]
                    m ^= lsb
                new_score = s_small + s_large
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_new_mask = new_mask
                    best_new_blocked_large = new_blocked_large
            
            # Try Swap (limited)
            if elapsed() < DEADLINE - 3.0:
                # Get list of selected
                selected_indices = []
                m = current_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    selected_indices.append(idx)
                    m ^= lsb
                
                # Limit swap search to top candidates by weight difference
                potential_adds = []
                for j in range(n_small):
                    if current_mask & (1 << j):
                        continue
                    if small_conflict_mask[j] & current_mask:
                        continue
                    potential_adds.append(j)
                potential_adds.sort(key=lambda j: -small_weights[j])
                potential_adds = potential_adds[:30]  # limit to top 30
                
                for i in selected_indices:
                    mask_without_i = current_mask & ~(1 << i)
                    blocked_without_i = compute_blocked_large(mask_without_i)
                    for j in potential_adds:
                        if mask_without_i & (1 << j):
                            continue
                        if small_conflict_mask[j] & mask_without_i:
                            continue
                        new_mask = mask_without_i | (1 << j)
                        new_blocked_large = blocked_without_i | small_blocks_large[j]
                        
                        all_large_mask = (1 << n_large) - 1
                        unblocked_large = all_large_mask & ~new_blocked_large
                        s_large, _ = get_large_score(unblocked_large)
                        s_small = 0.0
                        m2 = new_mask
                        while m2:
                            lsb = m2 & -m2
                            idx = lsb.bit_length() - 1
                            s_small += small_weight_arr[idx]
                            m2 ^= lsb
                        new_score = s_small + s_large
                        
                        if new_score > best_move_score:
                            best_move_score = new_score
                            best_new_mask = new_mask
                            best_new_blocked_large = new_blocked_large
            
            if best_new_mask != current_mask:
                current_mask = best_new_mask
                current_score = best_move_score
                current_blocked_large = best_new_blocked_large
                improved = True
                update_best(current_mask, current_score)
            else:
                improved = False

    # Final polish pass: try single additions/removals on the best solution
    if elapsed() < DEADLINE - 2.0:
        final_mask = best_small_mask
        final_blocked = compute_blocked_large(final_mask)
        final_score = best_score
        
        all_large_mask = (1 << n_large) - 1
        
        # Try adding any small node
        for i in range(n_small):
            if final_mask & (1 << i):
                continue
            if small_conflict_mask[i] & final_mask:
                continue
            
            new_mask = final_mask | (1 << i)
            new_blocked = final_blocked | small_blocks_large[i]
            unblocked_large = all_large_mask & ~new_blocked
            s_large, _ = get_large_score(unblocked_large)
            s_small = 0.0
            m = new_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                s_small += small_weight_arr[idx]
                m ^= lsb
            new_score = s_small + s_large
            
            if new_score > final_score:
                final_score = new_score
                final_mask = new_mask
                final_blocked = new_blocked
                update_best(final_mask, final_score)
        
        # Try removing any small node
        for i in range(n_small):
            if not (final_mask & (1 << i)):
                continue
            
            new_mask = final_mask & ~(1 << i)
            new_blocked = compute_blocked_large(new_mask)
            unblocked_large = all_large_mask & ~new_blocked
            s_large, _ = get_large_score(unblocked_large)
            s_small = 0.0
            m = new_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                s_small += small_weight_arr[idx]
                m ^= lsb
            new_score = s_small + s_large
            
            if new_score > final_score:
                final_score = new_score
                final_mask = new_mask
                final_blocked = new_blocked
                update_best(final_mask, final_score)

    # Construct final result
    result = []
    m = best_small_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(small_nums[idx])
        m ^= lsb
        
    lm = best_large_selection_mask
    while lm:
        lsb = lm & -lm
        idx = lsb.bit_length() - 1
        result.append(large_nums[idx])
        lm ^= lsb
        
    result.sort()
    return result
# EVOLVE-BLOCK-END
