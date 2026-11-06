import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    n = 5040
    target = 4979
    
    # 1. Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order to help find sums quickly
    divisors.sort(reverse=True)
    d_count = len(divisors)
    
    # 2. Determine minimum possible k
    # Max divisor is 5040. Target 4979.
    # 1 term: No divisor is 4979.
    # 2 terms: 5040 is too big. Next largest is 2520. 2520 + 2520 (distinct required, so next is 1680?) 
    # Actually, distinct divisors. Max sum of 2 distinct divisors < 4979?
    # Largest: 5040 (too big). 
    # If we don't use 5040, largest is 2520. Next is 1680. 2520+1680 = 4200 < 4979.
    # So k must be at least 3.
    
    min_k = 3
    
    # 3. Search for solution with increasing k
    # We use a recursive backtracking search
    # state: index in divisors, current sum, count selected, selected list
    
    def search(k, start_idx, current_sum, selected):
        if time.time() > deadline:
            return None
        
        if len(selected) == k:
            if current_sum == target:
                return selected
            return None
        
        # Pruning: if we have selected 'len(selected)' items, 
        # we need 'k - len(selected)' more items.
        # The maximum possible sum we can add is the sum of the largest 
        # 'k - len(selected)' divisors available from start_idx onwards.
        # But since divisors are sorted descending, the largest available 
        # are at start_idx, start_idx+1, ...
        # Actually, we can just check if current_sum + sum of next (k-len(selected)) largest 
        # possible divisors is < target. 
        # A tighter bound: The largest divisors are at the beginning. 
        # If we start from start_idx, the largest remaining divisors are divisors[start_idx], etc.
        
        items_needed = k - len(selected)
        
        # If we don't have enough divisors left
        if d_count - start_idx < items_needed:
            return None
        
        # Max possible sum addition: sum of the largest 'items_needed' divisors 
        # starting from start_idx.
        # Note: Since we are picking distinct divisors and moving forward in index,
        # the largest available are indeed the first 'items_needed' from start_idx.
        max_possible_add = sum(divisors[start_idx : start_idx + items_needed])
        
        if current_sum + max_possible_add < target:
            return None
        
        # If current_sum is already > target, prune (divisors are positive)
        if current_sum > target:
            return None
        
        # Iterate through possible next divisors
        # We can stop if current_sum + divisor[i] > target (since divisors are positive and sorted desc? 
        # No, sorted desc means larger first. If divisors[i] makes sum > target, 
        # smaller divisors later might still work? 
        # Wait, if divisors are sorted DESC, and current_sum + divisors[i] > target, 
        # then for any j > i, divisors[j] < divisors[i], so current_sum + divisors[j] might be <= target.
        # So we cannot break immediately on sum > target.
        # However, if current_sum + divisors[i] < target, we MUST pick something larger? 
        # No, we are picking one. If we pick divisors[i], the sum increases.
        
        for i in range(start_idx, d_count):
            d = divisors[i]
            
            # If current_sum + d > target, we can skip this d? 
            # Yes, because all subsequent divisors are smaller, but wait.
            # If we skip d, we might pick a smaller one later.
            # But if current_sum + d > target, then current_sum + d' (where d' < d) might be <= target.
            # So we cannot break. We just skip this specific d if it overshoots?
            # Actually, if we include d, and it overshoots, we don't include it.
            # So we just check if current_sum + d <= target. If not, we don't recurse with d.
            # But we still need to check other divisors.
            
            if current_sum + d > target:
                continue
                
            # Optimization: If we have items_needed left, and we pick d,
            # the max sum we can reach is current_sum + d + sum of next (items_needed-1) largest available.
            # This is already handled by the recursive call's pruning.
            
            new_selected = selected + [d]
            result = search(k, i + 1, current_sum + d, new_selected)
            if result is not None:
                return result
                
        return None

    # Try k = 3, 4, 5, ...
    for k in range(min_k, 11): # Up to 10 is safe
        # Reset time check? No, global deadline.
        sol = search(k, 0, 0, [])
        if sol is not None:
            return sol
            
    # Fallback: Greedy (just in case, though search should find it)
    # This should not be reached for k < 10 given the problem size.
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    if remainder == 0:
        return chosen
    return []
