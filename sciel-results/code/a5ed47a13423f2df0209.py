import time

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = sorted([i for i in range(1, n + 1) if n % i == 0], reverse=True)
    m = len(divisors)
    div_set = set(divisors)
    
    # Start time for safety
    start_time = time.time()
    time_limit = 50.0  # Leave margin for 60s limit
    
    # Search for minimum k
    # k=1 is impossible as 4979 is not a divisor
    # We start from k=2
    
    # Helper to check if a set of divisors sums to target
    # We use recursive backtracking with pruning for efficiency and clarity
    def find_combination(k, target_sum, start_idx, current_comb):
        # Pruning: if we can't fill remaining slots with max possible values
        # Max value is divisors[start_idx] (since sorted desc)
        # But distinctness makes simple max sum tricky. 
        # Simpler: just search. 60 divisors is small.
        
        if len(current_comb) == k:
            if sum(current_comb) == target_sum:
                return current_comb
            return None
        
        remaining_slots = k - len(current_comb)
        
        # If we have chosen all k, check sum
        # We iterate from start_idx to m - remaining_slots
        for i in range(start_idx, m - remaining_slots + 1):
            d = divisors[i]
            
            # Pruning: if current sum + d + (max possible sum of remaining) < target, continue?
            # Or if current sum + d > target, break (since divisors are positive and sorted desc, 
            # but we are picking smaller ones now, so sum will only decrease if we skip d? 
            # No, we are building sum. If d is too big, we break.
            # Since divisors are sorted descending, if we pick d, next picks are smaller.
            # If sum(current) + d > target, then any smaller d' will also result in sum < target? 
            # No, sum(current) + d' < sum(current) + d. 
            # Wait, if sum(current) + d > target, we might still need it if we subtract? No, all positive.
            # So if sum(current_comb) + d > target, we can break? 
            # No, because we might pick a smaller d later? 
            # But we are iterating i increasing, so d is decreasing.
            # If d is too big, smaller d's might be ok.
            # So we should NOT break if d is too big, we should just continue? 
            # Actually, if sum(current) + d > target, then d is too big. 
            # Since d is decreasing, subsequent d's are smaller, so they might be valid.
            # So we just skip this d.
            
            current_sum = sum(current_comb)
            if current_sum + d > target_sum:
                continue
                
            # Pruning: even if we pick the largest possible remaining divisors, can we reach target?
            # Max possible sum for remaining slots (remaining_slots - 1) is sum of next largest divisors.
            # This is expensive to calculate. 
            # Given m=60, k<=5, simple search is fast enough.
            
            new_comb = current_comb + [d]
            result = find_combination(k, target_sum, i + 1, new_comb)
            if result:
                return result
                
        return None

    # Try k=2
    res = find_combination(2, target, 0, [])
    if res:
        return res

    # Try k=3
    res = find_combination(3, target, 0, [])
    if res:
        return res

    # Try k=4
    res = find_combination(4, target, 0, [])
    if res:
        return res

    # Try k=5
    res = find_combination(5, target, 0, [])
    if res:
        return res
        
    # Fallback: Known solution
    return [2520, 1680, 560, 210, 9]
