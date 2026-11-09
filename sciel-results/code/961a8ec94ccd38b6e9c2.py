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
    small_conflict_mask = [0] * n_small
    for i in range(n_small):
        for j in range(i + 1, n_small):
            a, b = small_nums[i], small_nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N_LARGE_END:
                small_conflict_mask[i] |= (1 << j)
                small_conflict_mask[j] |= (1 << i)
                
    small_blocks_large = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        for j in range(n_large):
            b = large_nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N_LARGE_END:
                small_blocks_large[i] |= (1 << j)

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

    # Pre-sort large indices by weight descending
    large_sorted_indices = sorted(range(n_large), key=lambda j: -large_weights[j])
    
    def get_large_score_and_mask(unblocked_mask):
        selected = 0
        score = 0.0
        for j in large_sorted_indices:
            if not (unblocked_mask & (1 << j)):
                continue
            if large_large_conflict_mask[j] & selected:
                continue
            selected |= (1 << j)
            score += large_weights[j]
        return score, selected

    all_large_mask = (1 << n_large) - 1

    def compute_blocked_large(small_mask):
        blocked = 0
        m = small_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            blocked |= small_blocks_large[idx]
            m ^= lsb
        return blocked

    # Cache for large selection results
    large_cache = {}
    
    def get_large_score_and_mask_cached(unblocked_mask):
        if unblocked_mask in large_cache:
            return large_cache[unblocked_mask]
        score, selected = get_large_score_and_mask(unblocked_mask)
        if len(large_cache) < 10000:
            large_cache[unblocked_mask] = (score, selected)
        return score, selected

    def get_total_score(small_mask):
        s_small = 0.0
        m = small_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s_small += small_weights[idx]
            m ^= lsb
        blocked_large = compute_blocked_large(small_mask)
        unblocked_large = all_large_mask & ~blocked_large
        s_large, _ = get_large_score_and_mask_cached(unblocked_large)
        return s_small + s_large

    best_score = 0.0
    best_small_mask = 0
    best_large_selection_mask = 0

    def update_best(sm, score):
        nonlocal best_score, best_small_mask, best_large_selection_mask
        if score > best_score:
            best_score = score
            best_small_mask = sm
            blocked_large = compute_blocked_large(sm)
            unblocked_large = all_large_mask & ~blocked_large
            _, best_large_selection_mask = get_large_score_and_mask_cached(unblocked_large)

    # Generate initial candidates
    initial_masks = [0]
    
    # Greedy by weight
    mask = 0
    for i in range(n_small):
        if not (small_conflict_mask[i] & mask):
            mask |= (1 << i)
    initial_masks.append(mask)

    # Randomized greedy with multiple restarts
    for _ in range(40):
        order = list(range(n_small))
        random.shuffle(order)
        mask = 0
        for i in order:
            if not (small_conflict_mask[i] & mask):
                mask |= (1 << i)
        initial_masks.append(mask)

    # Local Search with delta-based evaluation
    for init_mask in initial_masks:
        if elapsed() > DEADLINE:
            break
            
        current_mask = init_mask
        current_blocked = compute_blocked_large(current_mask)
        current_unblocked = all_large_mask & ~current_blocked
        current_large_score, _ = get_large_score_and_mask_cached(current_unblocked)
        
        s_small = 0.0
        m = current_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s_small += small_weights[idx]
            m ^= lsb
        current_score = s_small + current_large_score
        update_best(current_mask, current_score)
        
        # Precompute blocked_without_i for all i in current_mask for efficient swaps/removals
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            best_move_score = current_score
            best_new_mask = current_mask
            
            # Precompute: for each i in current_mask, compute blocked without i
            blocked_without_i_cache = {}
            m = current_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                mask_without_i = current_mask & ~(1 << idx)
                blocked_without_i = 0
                m2 = mask_without_i
                while m2:
                    lsb2 = m2 & -m2
                    idx2 = lsb2.bit_length() - 1
                    blocked_without_i |= small_blocks_large[idx2]
                    m2 ^= lsb2
                blocked_without_i_cache[idx] = blocked_without_i
                m ^= lsb
            
            # Try Add: adding small i adds small_weights[i] but may block some large nodes
            for i in range(n_small):
                if current_mask & (1 << i):
                    continue
                if small_conflict_mask[i] & current_mask:
                    continue
                
                # New blocked = current_blocked | small_blocks_large[i]
                new_blocked = current_blocked | small_blocks_large[i]
                new_unblocked = all_large_mask & ~new_blocked
                new_large_score, _ = get_large_score_and_mask_cached(new_unblocked)
                new_score = current_score - current_large_score + small_weights[i] + new_large_score
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_new_mask = current_mask | (1 << i)
            
            # Try Remove: removing small i removes small_weights[i] but unblocks some large nodes
            for i in range(n_small):
                if not (current_mask & (1 << i)):
                    continue
                
                new_blocked = blocked_without_i_cache[i]
                new_unblocked = all_large_mask & ~new_blocked
                new_large_score, _ = get_large_score_and_mask_cached(new_unblocked)
                new_score = current_score - small_weights[i] - current_large_score + new_large_score
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_new_mask = current_mask & ~(1 << i)
            
            # Try Swap: remove i, add j
            if elapsed() < DEADLINE - 3.0:
                selected_indices = []
                m = current_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    selected_indices.append(idx)
                    m ^= lsb
                
                for i in selected_indices:
                    mask_without_i = current_mask & ~(1 << i)
                    blocked_without_i = blocked_without_i_cache[i]
                    
                    for j in range(n_small):
                        if mask_without_i & (1 << j):
                            continue
                        if small_conflict_mask[j] & mask_without_i:
                            continue
                        
                        new_blocked = blocked_without_i | small_blocks_large[j]
                        new_unblocked = all_large_mask & ~new_blocked
                        new_large_score, _ = get_large_score_and_mask_cached(new_unblocked)
                        new_score = current_score - small_weights[i] - current_large_score + small_weights[j] + new_large_score
                        
                        if new_score > best_move_score:
                            best_move_score = new_score
                            best_new_mask = mask_without_i | (1 << j)
            
            if best_new_mask != current_mask:
                current_mask = best_new_mask
                current_blocked = compute_blocked_large(current_mask)
                current_unblocked = all_large_mask & ~current_blocked
                current_large_score, _ = get_large_score_and_mask_cached(current_unblocked)
                
                s_small = 0.0
                m = current_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    s_small += small_weights[idx]
                    m ^= lsb
                current_score = s_small + current_large_score
                improved = True
                update_best(current_mask, current_score)
            else:
                improved = False

    # Final polish: try single additions/removals on best solution
    if elapsed() < DEADLINE - 2.0:
        for _ in range(5):  # More polish passes
            if elapsed() > DEADLINE - 1.0:
                break
            improved = True
            while improved and elapsed() < DEADLINE - 1.0:
                improved = False
                current_mask = best_small_mask
                current_blocked = compute_blocked_large(current_mask)
                current_unblocked = all_large_mask & ~current_blocked
                current_large_score, _ = get_large_score_and_mask_cached(current_unblocked)
                
                s_small = 0.0
                m = current_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    s_small += small_weights[idx]
                    m ^= lsb
                current_score = s_small + current_large_score
                
                # Precompute blocked_without_i for removals
                blocked_without_i_cache = {}
                m = current_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    mask_without_i = current_mask & ~(1 << idx)
                    blocked_without_i = 0
                    m2 = mask_without_i
                    while m2:
                        lsb2 = m2 & -m2
                        idx2 = lsb2.bit_length() - 1
                        blocked_without_i |= small_blocks_large[idx2]
                        m2 ^= lsb2
                    blocked_without_i_cache[idx] = blocked_without_i
                    m ^= lsb
                
                # Try additions
                for i in range(n_small):
                    if current_mask & (1 << i):
                        continue
                    if small_conflict_mask[i] & current_mask:
                        continue
                    new_blocked = current_blocked | small_blocks_large[i]
                    new_unblocked = all_large_mask & ~new_blocked
                    new_large_score, _ = get_large_score_and_mask_cached(new_unblocked)
                    new_score = current_score - current_large_score + small_weights[i] + new_large_score
                    if new_score > current_score + 1e-12:
                        current_mask |= (1 << i)
                        improved = True
                        break
                
                # Try removals
                if not improved:
                    for i in range(n_small):
                        if not (current_mask & (1 << i)):
                            continue
                        new_blocked = blocked_without_i_cache[i]
                        new_unblocked = all_large_mask & ~new_blocked
                        new_large_score, _ = get_large_score_and_mask_cached(new_unblocked)
                        new_score = current_score - small_weights[i] - current_large_score + new_large_score
                        if new_score > current_score + 1e-12:
                            current_mask &= ~(1 << i)
                            improved = True
                            break
                
                if improved:
                    new_score = get_total_score(current_mask)
                    if new_score > best_score:
                        update_best(current_mask, new_score)

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
