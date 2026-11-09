import time
from math import gcd
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    weights = [L // d for d in denominators]
    
    # Split indices
    mid = n // 2
    idx1 = list(range(mid))
    idx2 = list(range(mid, n))
    
    w1 = [weights[i] for i in idx1]
    w2 = [weights[i] for i in idx2]
    
    # Generate all subset sums for half 1
    # Store (sum, mask)
    sums1 = [(0, 0)]
    for i in range(len(w1)):
        wi = w1[i]
        sums1 += [(s + wi, m | (1 << i)) for s, m in sums1]
        
    # Sort by sum
    sums1.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    sums1_values = [s for s, m in sums1]
    
    # Precompute suffix sums for w2 for pruning
    suffix_sums = [0] * (len(w2) + 1)
    for i in range(len(w2) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + w2[i]
        
    max_s1 = sums1_values[-1] if sums1_values else 0
    
    best_deficit = L  # Max possible deficit
    best_mask1 = 0
    best_mask2 = 0
    
    # DFS for half 2 with pruning - explore include first for tighter bounds
    def dfs2(pos, current_sum, current_mask):
        nonlocal best_deficit, best_mask1, best_mask2
        
        if time.time() > deadline:
            return
            
        # Pruning: even with max possible, can't beat current best
        max_possible_total = current_sum + suffix_sums[pos] + max_s1
        if max_possible_total < L - best_deficit:
            return

        if pos == len(w2):
            if current_sum < L:
                target = L - current_sum
                idx = bisect_left(sums1_values, target)
                if idx > 0:
                    s1_val, m1 = sums1[idx - 1]
                    total_sum = s1_val + current_sum
                    if total_sum < L:
                        deficit = L - total_sum
                        if deficit < best_deficit:
                            best_deficit = deficit
                            best_mask1 = m1
                            best_mask2 = current_mask
            return
        
        # Option 1: Include w2[pos] first (explore larger sums to tighten bound)
        new_sum = current_sum + w2[pos]
        if new_sum < L:
            dfs2(pos + 1, new_sum, current_mask | (1 << pos))
            
        # Option 2: Exclude w2[pos]
        dfs2(pos + 1, current_sum, current_mask)
        
    dfs2(0, 0, 0)
    
    # Reconstruct indices from best masks
    result_indices = []
    for i in range(len(idx1)):
        if best_mask1 & (1 << i):
            result_indices.append(idx1[i])
            
    for i in range(len(idx2)):
        if best_mask2 & (1 << i):
            result_indices.append(idx2[i])
            
    if not result_indices:
        return [0]
    
    # Local search refinement: 1-swap and add/remove
    if time.time() < deadline - 10:
        current_set = set(result_indices)
        current_sum = sum(weights[i] for i in result_indices)
        current_deficit = L - current_sum
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding one element
            for add_idx in range(n):
                if add_idx in current_set:
                    continue
                new_sum = current_sum + weights[add_idx]
                if new_sum < L:
                    new_deficit = L - new_sum
                    if new_deficit < current_deficit:
                        current_set.add(add_idx)
                        current_sum = new_sum
                        current_deficit = new_deficit
                        improved = True
                        break
            if improved:
                continue
            
            # Try 1-swap
            for remove_idx in list(current_set):
                for add_idx in range(n):
                    if add_idx in current_set:
                        continue
                    new_sum = current_sum - weights[remove_idx] + weights[add_idx]
                    if new_sum < L:
                        new_deficit = L - new_sum
                        if new_deficit < current_deficit:
                            current_set.discard(remove_idx)
                            current_set.add(add_idx)
                            current_sum = new_sum
                            current_deficit = new_deficit
                            improved = True
                            break
                if improved:
                    break
            
            # Try 2-swap
            if not improved:
                removed = []
                for remove_idx in list(current_set):
                    for remove_idx2 in list(current_set):
                        if remove_idx2 <= remove_idx:
                            continue
                        for add_idx in range(n):
                            if add_idx in current_set:
                                continue
                            for add_idx2 in range(n):
                                if add_idx2 <= add_idx or add_idx2 in current_set:
                                    continue
                                new_sum = current_sum - weights[remove_idx] - weights[remove_idx2] + weights[add_idx] + weights[add_idx2]
                                if new_sum < L:
                                    new_deficit = L - new_sum
                                    if new_deficit < current_deficit:
                                        current_set.discard(remove_idx)
                                        current_set.discard(remove_idx2)
                                        current_set.add(add_idx)
                                        current_set.add(add_idx2)
                                        current_sum = new_sum
                                        current_deficit = new_deficit
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue
    
    # Rebuild result from local search
    if time.time() < deadline - 5:
        result_indices = sorted(current_set)
    
    if not result_indices:
        return [0]
        
    return result_indices
