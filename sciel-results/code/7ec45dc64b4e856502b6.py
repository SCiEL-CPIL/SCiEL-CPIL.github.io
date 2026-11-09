import time
from math import factorial

N = factorial(12)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Generate divisors of N
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            if i > 12:
                divisors.append(i)
            if N // i != i and N // i > 12:
                divisors.append(N // i)
    
    divisors = sorted(set(divisors))
    k = len(divisors)
    
    # Precompute max_prod[j][m] = product of largest m divisors in divisors[0..j]
    # j from 0 to k-1, m from 1 to 12
    max_prod = [[1] * 13 for _ in range(k)]
    for j in range(k):
        prod = 1
        for m in range(1, 13):
            if j - m + 1 >= 0:
                prod *= divisors[j - m + 1]
                max_prod[j][m] = prod
            else:
                # Not enough divisors
                max_prod[j][m] = 0 

    def can_solve(j):
        # DFS to find subset of divisors[0..j] that multiplies to N
        # returns list of factors or None
        
        def dfs(rem, start_idx, current_factors):
            if time.time() > deadline:
                return None
            
            if rem == 1:
                return current_factors
            
            if len(current_factors) >= 12:
                return None
            
            if start_idx > j:
                return None
                
            # Pruning 1: smallest divisor is too big
            if rem < divisors[start_idx]:
                return None
                
            # Pruning 2: largest possible product is too small
            slots_left = 12 - len(current_factors)
            # The largest possible product using distinct divisors from divisors[start_idx..j]
            # is the product of the largest slots_left divisors in that range.
            # These are divisors[j], divisors[j-1], ..., divisors[j-slots_left+1]
            # We need to ensure j - slots_left + 1 >= start_idx
            if slots_left > 0:
                if j - slots_left + 1 >= start_idx:
                    # Calculate product of divisors[j-slots_left+1 ... j]
                    # We can use max_prod if we precomputed it correctly.
                    # max_prod[j][slots_left] is product of largest slots_left in 0..j.
                    # But we are restricted to start_idx..j.
                    # If start_idx <= j - slots_left + 1, then the largest slots_left in start_idx..j
                    # are the same as the largest slots_left in 0..j.
                    # If start_idx > j - slots_left + 1, then we have fewer than slots_left divisors?
                    # No, we have j - start_idx + 1 divisors.
                    # If slots_left > j - start_idx + 1, we can't fill slots. But we check start_idx <= j.
                    # The max product is product of all available divisors if slots_left >= available count?
                    # No, we can pick at most slots_left.
                    # The max product is the product of the min(slots_left, available_count) largest divisors.
                    
                    available_count = j - start_idx + 1
                    if slots_left > available_count:
                         # We must use all available divisors? No, we can use fewer.
                         # But we want to maximize product to check if rem can be reached.
                         # If even the product of all available divisors is < rem, then fail.
                         # Actually, if we have to pick exactly slots_left? No, we can pick up to 12.
                         # But rem must be exactly formed.
                         # If we pick fewer, the product is smaller.
                         # So the max product is indeed the product of the largest min(slots_left, available_count) divisors.
                         # Wait, if we pick fewer, the remaining product is 1? No, rem must become 1.
                         # So we must pick exactly enough factors to make rem 1.
                         # So the number of factors is fixed by the decomposition? No.
                         # But if we pick a factor, rem decreases.
                         # The bound is: can we form rem with <= slots_left factors from the set?
                         # Max product with slots_left factors is the product of the largest slots_left.
                         # If rem > max_product, then impossible.
                         
                         pass # handled below
                    
                    # Let's just compute the product of the largest slots_left divisors in divisors[start_idx..j]
                    # If slots_left > available_count, we take all available.
                    m = min(slots_left, available_count)
                    if m > 0:
                        # Indices: j-m+1 to j
                        # But must be >= start_idx
                        start_bound = max(start_idx, j - m + 1)
                        if start_bound > j:
                            max_p = 1
                        else:
                            max_p = 1
                            for idx in range(start_bound, j + 1):
                                max_p *= divisors[idx]
                            # Wait, if m < available_count, we take the largest m.
                            # So indices j-m+1 to j.
                            # If j-m+1 < start_idx, we take start_idx to j.
                            # This logic is slightly off.
                            # The largest m divisors in start_idx..j are divisors[j-m+1]..divisors[j]
                            # IF j-m+1 >= start_idx.
                            # IF j-m+1 < start_idx, then m > available_count, which is handled by m=min(...).
                            # So if m == available_count, we take all.
                            # If m < available_count, we take the largest m, which are the last m in the range.
                            
                            # Correct logic:
                            # The largest m divisors in divisors[start_idx..j] are:
                            # divisors[j], divisors[j-1], ..., divisors[j-m+1]
                            # We need j-m+1 >= start_idx.
                            # Since m = min(slots_left, available_count), and available_count = j - start_idx + 1.
                            # If slots_left <= available_count, m = slots_left.
                            # j - slots_left + 1 >= j - available_count + 1 = start_idx.
                            # So j-m+1 >= start_idx holds.
                            # If slots_left > available_count, m = available_count.
                            # j - available_count + 1 = start_idx.
                            # So j-m+1 = start_idx.
                            # So in both cases, the indices are j-m+1 to j.
                            
                            # Let's re-calculate max_p properly.
                            max_p = 1
                            for idx in range(j - m + 1, j + 1):
                                max_p *= divisors[idx]
                            
                            if rem > max_p:
                                return None
                else:
                    # j - slots_left + 1 < start_idx
                    # This means slots_left > j - start_idx + 1 = available_count
                    # So we must use all available divisors to get the max product.
                    # Max product is product of all divisors in start_idx..j
                    max_p = 1
                    for idx in range(start_idx, j + 1):
                        max_p *= divisors[idx]
                    if rem > max_p:
                        return None
            
            for i in range(start_idx, j + 1):
                if rem % divisors[i] == 0:
                    res = dfs(rem // divisors[i], i + 1, current_factors + [divisors[i]])
                    if res is not None:
                        return res
            return None
        
        return dfs(N, 0, [])

    for j in range(k):
        if time.time() > deadline:
            break
        sol = can_solve(j)
        if sol is not None:
            return sol
            
    return [N]
