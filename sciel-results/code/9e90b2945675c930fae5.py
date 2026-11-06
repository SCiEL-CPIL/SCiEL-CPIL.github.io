"""Rewritten program: Hybrid Exact/Heuristic Solver for Max Weight Independent Set."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n_max = 256
    # Partition: Small numbers (2-16) have high weight and dense conflicts.
    # Large numbers (17-256) have lower weight and sparser conflicts.
    small_nums = list(range(2, 17))
    large_nums = list(range(17, 257))
    
    n_small = len(small_nums)
    n_large = len(large_nums)
    
    # Map values to indices
    small_idx = {v: i for i, v in enumerate(small_nums)}
    large_idx = {v: i for i, v in enumerate(large_nums)}
    
    # Weights
    w_small = [1.0 / v for v in small_nums]
    w_large = [1.0 / v for v in large_nums]
    
    # Precompute conflicts
    # conf_small[i] is bitmask of small nodes conflicting with small_nums[i]
    conf_small = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        for j in range(i + 1, n_small):
            b = small_nums[j]
            if a * b // gcd(a, b) <= n_max:
                conf_small[i] |= (1 << j)
                conf_small[j] |= (1 << i)
                
    # conf_large[j] is bitmask of large nodes conflicting with large_nums[j]
    conf_large = [0] * n_large
    for i in range(n_large):
        a = large_nums[i]
        for j in range(i + 1, n_large):
            b = large_nums[j]
            if a * b // gcd(a, b) <= n_max:
                conf_large[i] |= (1 << j)
                conf_large[j] |= (1 << i)
                
    # block_large_by_small[i] is bitmask of large nodes blocked by small_nums[i]
    # i.e., large node j is blocked if lcm(small_nums[i], large_nums[j]) <= 256
    block_large_by_small = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        mask = 0
        for j in range(n_large):
            b = large_nums[j]
            if a * b // gcd(a, b) <= n_max:
                mask |= (1 << j)
        block_large_by_small[i] = mask

    # Pre-sorted indices of large nodes by weight descending (value ascending)
    large_order = sorted(range(n_large), key=lambda x: -w_large[x])
    
    # Fast Solver for Large Numbers
    # Input: free_mask (bitmask of large nodes that are NOT blocked by current small selection)
    # Returns: (score, selected_mask)
    def solve_large(free_mask):
        if free_mask == 0:
            return 0.0, 0
        
        # Greedy Initialization
        selected_mask = 0
        score = 0.0
        
        for idx in large_order:
            if not (free_mask & (1 << idx)):
                continue
            if (conf_large[idx] & selected_mask) == 0:
                selected_mask |= (1 << idx)
                score += w_large[idx]
        
        # Local Search: 1-swap and limited 2-swap
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1-swap: Try to swap out a selected node i and in an unselected node j
            m = selected_mask
            while m and not improved:
                lsb_i = m & -m
                i = lsb_i.bit_length() - 1
                m ^= lsb_i
                
                temp_mask = selected_mask & ~lsb_i
                best_j = -1
                best_gain = 0
                
                for j in large_order:
                    if w_large[j] <= w_large[i]:
                        break
                    if selected_mask & (1 << j):
                        continue
                    if not (free_mask & (1 << j)):
                        continue
                    if (conf_large[j] & temp_mask) == 0:
                        gain = w_large[j] - w_large[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_j = j
                
                if best_j != -1:
                    selected_mask = temp_mask | (1 << best_j)
                    score += best_gain
                    improved = True
                    break
            
            # 2-swap: Try to swap out two selected nodes i,k and in two unselected nodes j,l
            # Only do this if we have time and the set is not too large
            if not improved and time.time() < deadline:
                selected_list = []
                m = selected_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    selected_list.append(idx)
                    m ^= lsb
                
                # Limit 2-swap to smaller sets to avoid O(n^4) blowup
                if 2 <= len(selected_list) <= 30:
                    best_swap = None
                    best_swap_gain = 0
                    
                    for ai in range(len(selected_list)):
                        i = selected_list[ai]
                        for ak in range(ai + 1, len(selected_list)):
                            k = selected_list[ak]
                            temp_mask = selected_mask & ~(1 << i) & ~(1 << k)
                            
                            # Find best pair j, l to add
                            best_pair_gain = 0
                            best_pair = None
                            
                            for j in large_order:
                                if w_large[j] * 2 <= w_large[i] + w_large[k]:
                                    break
                                if selected_mask & (1 << j):
                                    continue
                                if not (free_mask & (1 << j)):
                                    continue
                                if (conf_large[j] & temp_mask) != 0:
                                    continue
                                
                                for l in large_order:
                                    if l == j:
                                        continue
                                    if w_large[l] <= w_large[j]:
                                        break
                                    if selected_mask & (1 << l):
                                        continue
                                    if not (free_mask & (1 << l)):
                                        continue
                                    if (conf_large[l] & temp_mask) != 0:
                                        continue
                                    if (conf_large[j] & (1 << l)) != 0:
                                        continue
                                    
                                    pair_gain = w_large[j] + w_large[l] - w_large[i] - w_large[k]
                                    if pair_gain > best_pair_gain:
                                        best_pair_gain = pair_gain
                                        best_pair = (j, l)
                            
                            if best_pair and best_pair_gain > best_swap_gain:
                                best_swap_gain = best_pair_gain
                                best_swap = (i, k, best_pair[0], best_pair[1])
                    
                    if best_swap:
                        i, k, j, l = best_swap
                        selected_mask = selected_mask & ~(1 << i) & ~(1 << k) | (1 << j) | (1 << l)
                        score += best_swap_gain
                        improved = True
        
        return score, selected_mask

    # B&B for Small Numbers
    best_total_score = 0.0
    best_small_mask = 0
    best_large_mask = 0
    
    # Sort small nodes by weight descending for B&B
    small_order = sorted(range(n_small), key=lambda x: -w_small[x])
    
    # Precompute suffix sums of small weights for upper bound
    suffix_sum = [0.0] * (n_small + 1)
    for i in range(n_small - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + w_small[small_order[i]]
    
    all_large_mask = (1 << n_large) - 1
    
    # Cache for greedy bounds to avoid recomputation
    greedy_bound_cache = {}
    
    def greedy_large_bound(free_mask):
        """Fast greedy upper bound for large part with caching."""
        if free_mask in greedy_bound_cache:
            return greedy_bound_cache[free_mask]
        
        if free_mask == 0:
            greedy_bound_cache[free_mask] = 0.0
            return 0.0
        
        selected = 0
        score = 0.0
        for idx in large_order:
            if not (free_mask & (1 << idx)):
                continue
            if (conf_large[idx] & selected) == 0:
                selected |= (1 << idx)
                score += w_large[idx]
        
        greedy_bound_cache[free_mask] = score
        return score

    def search_small(idx, current_mask, current_score, forbidden_large_mask):
        nonlocal best_total_score, best_small_mask, best_large_mask
        
        if time.time() > deadline:
            return
            
        free_large_mask = all_large_mask & ~forbidden_large_mask
        
        # If we've processed all small nodes, solve large exactly
        if idx == n_small:
            l_score, l_mask = solve_large(free_large_mask)
            total = current_score + l_score
            if total > best_total_score:
                best_total_score = total
                best_small_mask = current_mask
                best_large_mask = l_mask
            return
        
        # Upper bound calculation
        ub_small = suffix_sum[idx]
        
        # Use greedy bound for large part (tighter than sum)
        ub_large = greedy_large_bound(free_large_mask)
        
        ub_total = current_score + ub_small + ub_large
        
        if ub_total <= best_total_score + 1e-9:
            return
            
        # Branch on small_order[idx]
        i = small_order[idx]
        
        # Option 1: Include i (if compatible)
        include_possible = (conf_small[i] & current_mask) == 0
        
        if include_possible:
            new_forbidden = forbidden_large_mask | block_large_by_small[i]
            include_score = current_score + w_small[i] + suffix_sum[idx + 1] + greedy_large_bound(free_large_mask & ~block_large_by_small[i])
            exclude_score = current_score + suffix_sum[idx + 1] + greedy_large_bound(free_large_mask)
            
            if include_score >= exclude_score:
                search_small(idx + 1, current_mask | (1 << i), current_score + w_small[i], new_forbidden)
                search_small(idx + 1, current_mask, current_score, forbidden_large_mask)
            else:
                search_small(idx + 1, current_mask, current_score, forbidden_large_mask)
                search_small(idx + 1, current_mask | (1 << i), current_score + w_small[i], new_forbidden)
        else:
            search_small(idx + 1, current_mask, current_score, forbidden_large_mask)

    # Run the main B&B search
    search_small(0, 0, 0.0, 0)
    
    # Try forced exclusions to explore different basins
    forced_exclusions = [
        0,  # No exclusion (already done)
        (1 << 0),  # Exclude small_nums[0] = 2
        (1 << 1),  # Exclude small_nums[1] = 3
        (1 << 0) | (1 << 1),  # Exclude 2 and 3
    ]
    
    for excl in forced_exclusions[1:]:  # Skip first one, already done
        if time.time() > deadline:
            break
        
        # Create a modified search that forces exclusion of specified small nodes
        def search_small_forced(idx, current_mask, current_score, forbidden_large_mask, forced_excl):
            nonlocal best_total_score, best_small_mask, best_large_mask
            
            if time.time() > deadline:
                return
            
            free_large_mask = all_large_mask & ~forbidden_large_mask
            
            if idx == n_small:
                l_score, l_mask = solve_large(free_large_mask)
                total = current_score + l_score
                if total > best_total_score:
                    best_total_score = total
                    best_small_mask = current_mask
                    best_large_mask = l_mask
                return
            
            ub_small = suffix_sum[idx]
            ub_large = greedy_large_bound(free_large_mask)
            ub_total = current_score + ub_small + ub_large
            
            if ub_total <= best_total_score + 1e-9:
                return
            
            i = small_order[idx]
            
            # If this node is forced excluded, skip it
            if forced_excl & (1 << i):
                search_small_forced(idx + 1, current_mask, current_score, forbidden_large_mask, forced_excl)
                return
            
            include_possible = (conf_small[i] & current_mask) == 0
            
            if include_possible:
                new_forbidden = forbidden_large_mask | block_large_by_small[i]
                include_score = current_score + w_small[i] + suffix_sum[idx + 1] + greedy_large_bound(free_large_mask & ~block_large_by_small[i])
                exclude_score = current_score + suffix_sum[idx + 1] + greedy_large_bound(free_large_mask)
                
                if include_score >= exclude_score:
                    search_small_forced(idx + 1, current_mask | (1 << i), current_score + w_small[i], new_forbidden, forced_excl)
                    search_small_forced(idx + 1, current_mask, current_score, forbidden_large_mask, forced_excl)
                else:
                    search_small_forced(idx + 1, current_mask, current_score, forbidden_large_mask, forced_excl)
                    search_small_forced(idx + 1, current_mask | (1 << i), current_score + w_small[i], new_forbidden, forced_excl)
            else:
                search_small_forced(idx + 1, current_mask, current_score, forbidden_large_mask, forced_excl)
        
        search_small_forced(0, 0, 0.0, 0, excl)
    
    # Extract result
    result = []
    
    m = best_small_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        result.append(small_nums[i])
        m ^= lsb
        
    m = best_large_mask
    while m:
        lsb = m & -m
        j = lsb.bit_length() - 1
        result.append(large_nums[j])
        m ^= lsb
    
    # Ensure nonempty result
    if not result:
        result = [2]
    
    return result

# EVOLVE-BLOCK-END
