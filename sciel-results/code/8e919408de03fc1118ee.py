import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    
    # 1. Identify fixed elements: 1 and primes > n/2
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    large_primes = [p for p in primes if p > n // 2]
    
    fixed = [1] + large_primes
    fixed_set = set(fixed)
    
    # Candidates are the rest
    candidates = [i for i in range(2, n + 1) if i not in fixed_set]
    num_cands = len(candidates)
    
    # Map value to index in candidates
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Precompute conflict bitmasks
    # conflict_mask[i] is a bitmask where bit j is 1 if candidates[i] and candidates[j] are NOT coprime
    conflict_mask = [0] * num_cands
    for i in range(num_cands):
        vi = candidates[i]
        for j in range(i + 1, num_cands):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    def get_sum(mask):
        s = 0
        for i in range(num_cands):
            if (mask >> i) & 1:
                s += candidates[i]
        return s

    def greedy_initial(order):
        mask = 0
        for v in order:
            idx = val_to_idx[v]
            # Check if idx conflicts with any currently selected
            if (conflict_mask[idx] & mask) == 0:
                mask |= (1 << idx)
        return mask

    def local_search(initial_mask):
        current_mask = initial_mask
        current_sum = get_sum(current_mask)
        
        # Precompute sorted indices by value descending for quick access to "large" nodes
        # We will use this to limit the search space for "add" operations
        # indices_sorted_desc: list of indices sorted by candidates[index] descending
        indices_sorted_desc = sorted(range(num_cands), key=lambda i: candidates[i], reverse=True)
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            
            # --- 1-for-1 Swap ---
            best_gain = 0
            best_move_11 = None
            # Iterate over selected nodes to remove
            for j in range(num_cands):
                if not ((current_mask >> j) & 1):
                    continue
                val_j = candidates[j]
                # Mask without j
                mask_without_j = current_mask & ~(1 << j)
                
                # Try to add unselected i
                # Optimization: Only consider i > val_j for gain > 0? 
                # Not necessarily, but if val_i <= val_j, gain <= 0.
                # However, a swap with 0 gain might enable a better 2-for-2 later?
                # For now, we look for strictly positive gain.
                for i in range(num_cands):
                    if ((current_mask >> i) & 1):
                        continue
                    val_i = candidates[i]
                    if val_i <= val_j:
                        continue
                    # Check compatibility: i must not conflict with mask_without_j
                    if (conflict_mask[i] & mask_without_j) == 0:
                        gain = val_i - val_j
                        if gain > best_gain:
                            best_gain = gain
                            best_move_11 = (i, j)
                            
            if best_move_11:
                i, j = best_move_11
                current_mask &= ~(1 << j)
                current_mask |= (1 << i)
                current_sum += best_gain
                improved = True
                continue
            
            # --- 1-for-2 Swap ---
            # Remove j, add i1, i2
            best_gain_12 = 0
            best_move_12 = None
            # To keep it fast, we only consider top K unselected nodes for i1, i2
            K = 20
            # Get top K unselected
            unselected_indices = [i for i in indices_sorted_desc if not ((current_mask >> i) & 1)]
            top_unselected = unselected_indices[:K]
            
            for j in range(num_cands):
                if not ((current_mask >> j) & 1):
                    continue
                val_j = candidates[j]
                mask_without_j = current_mask & ~(1 << j)
                
                # Find compatible top_unselected
                comp_unselected = []
                for i in top_unselected:
                    if (conflict_mask[i] & mask_without_j) == 0:
                        comp_unselected.append(i)
                        
                # Try pairs from comp_unselected
                for a in range(len(comp_unselected)):
                    i1 = comp_unselected[a]
                    val_i1 = candidates[i1]
                    for b in range(a + 1, len(comp_unselected)):
                        i2 = comp_unselected[b]
                        val_i2 = candidates[i2]
                        # Check if i1, i2 are coprime
                        if math.gcd(candidates[i1], candidates[i2]) == 1:
                            gain = val_i1 + val_i2 - val_j
                            if gain > best_gain_12:
                                best_gain_12 = gain
                                best_move_12 = (i1, i2, j)
                                
            if best_move_12:
                i1, i2, j = best_move_12
                current_mask &= ~(1 << j)
                current_mask |= (1 << i1)
                current_mask |= (1 << i2)
                current_sum += best_gain_12
                improved = True
                continue
            
            # --- 2-for-1 Swap ---
            # Remove j1, j2, add i
            best_gain_21 = 0
            best_move_21 = None
            top_unselected_1 = top_unselected[:5] # Very large only
            
            # We need to iterate pairs of selected nodes to remove
            # This is O(N^2) * O(K)
            # N=231, N^2/2 ~ 26k. K=5. 130k checks. Acceptable.
            for j1 in range(num_cands):
                if not ((current_mask >> j1) & 1):
                    continue
                val_j1 = candidates[j1]
                for j2 in range(j1 + 1, num_cands):
                    if not ((current_mask >> j2) & 1):
                        continue
                    val_j2 = candidates[j2]
                    
                    mask_without_j1j2 = current_mask & ~(1 << j1) & ~(1 << j2)
                    
                    for i in top_unselected_1:
                        val_i = candidates[i]
                        # Check compatibility
                        if (conflict_mask[i] & mask_without_j1j2) == 0:
                            gain = val_i - (val_j1 + val_j2)
                            if gain > best_gain_21:
                                best_gain_21 = gain
                                best_move_21 = (i, j1, j2)
                                
            if best_move_21:
                i, j1, j2 = best_move_21
                current_mask &= ~(1 << j1)
                current_mask &= ~(1 << j2)
                current_mask |= (1 << i)
                current_sum += best_gain_21
                improved = True
                continue

            # --- 2-for-2 Swap ---
            # This is expensive. We will only do it if time permits and with strict limits.
            # Remove j1, j2, add i1, i2.
            # Gain = val_i1 + val_i2 - val_j1 - val_j2.
            # We restrict i1, i2 to top 5 unselected.
            # We restrict j1, j2 to "small" selected nodes? Or just iterate all pairs?
            # Iterating all pairs of selected (approx 100-150 nodes) is ~10k pairs.
            # For each pair, check top 5 unselected pairs (10 pairs).
            # 10k * 10 = 100k checks. Acceptable.
            
            best_gain_22 = 0
            best_move_22 = None
            top_unselected_2 = top_unselected[:5]
            
            # Precompute pairs of top_unselected_2 that are coprime
            valid_unselected_pairs = []
            for a in range(len(top_unselected_2)):
                i1 = top_unselected_2[a]
                for b in range(a + 1, len(top_unselected_2)):
                    i2 = top_unselected_2[b]
                    if math.gcd(candidates[i1], candidates[i2]) == 1:
                        valid_unselected_pairs.append((i1, i2))
                        
            for j1 in range(num_cands):
                if not ((current_mask >> j1) & 1):
                    continue
                val_j1 = candidates[j1]
                for j2 in range(j1 + 1, num_cands):
                    if not ((current_mask >> j2) & 1):
                        continue
                    val_j2 = candidates[j2]
                    
                    mask_without_j1j2 = current_mask & ~(1 << j1) & ~(1 << j2)
                    
                    for i1, i2 in valid_unselected_pairs:
                        # Check i1, i2 compatible with mask_without_j1j2
                        if (conflict_mask[i1] & mask_without_j1j2) == 0 and \
                           (conflict_mask[i2] & mask_without_j1j2) == 0:
                            gain = candidates[i1] + candidates[i2] - val_j1 - val_j2
                            if gain > best_gain_22:
                                best_gain_22 = gain
                                best_move_22 = (i1, i2, j1, j2)
                                
            if best_move_22:
                i1, i2, j1, j2 = best_move_22
                current_mask &= ~(1 << j1)
                current_mask &= ~(1 << j2)
                current_mask |= (1 << i1)
                current_mask |= (1 << i2)
                current_sum += best_gain_22
                improved = True
                continue

        return current_mask, current_sum

    # Main Loop: Multiple Restart
    best_mask = 0
    best_total_sum = sum(fixed)
    
    # Initial Orders
    orders = []
    # 1. Descending Value
    orders.append(sorted(candidates, reverse=True))
    # 2. Ascending Value
    orders.append(sorted(candidates))
    # 3. Random Shuffles
    rng = random.Random(42)
    for _ in range(10):
        rand_order = candidates.copy()
        rng.shuffle(rand_order)
        orders.append(rand_order)
        
    # Also try "Least Conflicts First"
    conflict_counts = [bin(conflict_mask[i]).count('1') for i in range(num_cands)]
    order_least_conflict = sorted(candidates, key=lambda v: (conflict_counts[val_to_idx[v]], -v))
    orders.append(order_least_conflict)
    
    for order in orders:
        if time.time() > deadline:
            break
        init_mask = greedy_initial(order)
        mask, sum_cands = local_search(init_mask)
        total = sum_cands + sum(fixed)
        if total > best_total_sum:
            best_total_sum = total
            best_mask = mask
            
    # Construct Result
    result = list(fixed)
    for i in range(num_cands):
        if (best_mask >> i) & 1:
            result.append(candidates[i])
            
    # Ensure types are int
    result = [int(x) for x in result]
    
    return result
