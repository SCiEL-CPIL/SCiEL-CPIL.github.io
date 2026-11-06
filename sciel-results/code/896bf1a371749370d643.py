import time
from math import factorial

def construct():
    deadline = time.time() + 85  # Safety margin under 90s kill
    target = factorial(12)
    
    # 1. Generate all divisors of 12!
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            other = target // i
            if other != i and other > 12:
                divisors.append(other)
    
    # 2. Sort divisors
    divisors.sort()
    
    # 3. Iterate through possible maximum factors M
    # We want to minimize M, so we check from smallest to largest
    for idx_m, m in enumerate(divisors):
        if time.time() > deadline:
            break
            
        # For each possible count k of factors
        for k in range(1, 13):
            if k == 1:
                if m == target:
                    return [m]
                else:
                    continue
            
            # If m does not divide target, it cannot be a factor
            if target % m != 0:
                continue
                
            remaining_product = target // m
            
            # We need to find k-1 distinct factors from divisors[0:idx_m] 
            # (i.e., divisors strictly less than m)
            # such that their product is remaining_product.
            
            candidates = divisors[:idx_m]
            
            # Heuristic check: 
            # The smallest possible product of k-1 distinct numbers from candidates
            # is the product of the first k-1 elements.
            if len(candidates) < k - 1:
                continue
                
            min_possible = 1
            for i in range(k - 1):
                min_possible *= candidates[i]
            if min_possible > remaining_product:
                continue
                
            max_possible = 1
            for i in range(len(candidates) - (k - 1), len(candidates)):
                max_possible *= candidates[i]
            if max_possible < remaining_product:
                continue

            # Backtracking search for subset
            # We want to pick indices i_1 < i_2 < ... < i_{k-1} from candidates
            # such that product(candidates[i_j]) == remaining_product
            
            found = False
            result_subset = []
            
            def backtrack(start, current_prod, count):
                nonlocal found, result_subset
                if found:
                    return
                if time.time() > deadline:
                    return
                    
                if count == k - 1:
                    if current_prod == remaining_product:
                        found = True
                    return
                    
                # Pruning:
                # If current_prod > remaining_product, impossible
                if current_prod > remaining_product:
                    return
                    
                # If we need more factors, check if it's possible
                # to reach remaining_product with the remaining choices.
                # The smallest possible next factor is candidates[start].
                # The largest possible product for the remaining (count + 1) factors
                # is the product of the largest (count + 1) factors available from start onwards.
                # But a simpler bound: 
                # Max product we can achieve with (k-1 - count) more factors from candidates[start:]
                # is the product of the last (k-1 - count) elements in candidates[start:].
                
                remaining_slots = k - 1 - count
                if remaining_slots > 0:
                    # Check if even the largest possible product is too small
                    # Or if the smallest possible product is too large
                    
                    # Smallest possible product for remaining_slots items starting from 'start'
                    min_prod = 1
                    for i in range(start, min(start + remaining_slots, len(candidates))):
                        min_prod *= candidates[i]
                    if min_prod * current_prod > remaining_product:
                        return
                        
                    # Largest possible product for remaining_slots items ending at len(candidates)-1
                    # We can pick any remaining_slots items from candidates[start:]
                    # The max product is picking the largest remaining_slots items.
                    max_prod = 1
                    for i in range(len(candidates) - remaining_slots, len(candidates)):
                        max_prod *= candidates[i]
                    # Note: This max_prod assumes we can pick the global largest ones.
                    # But we must pick from candidates[start:].
                    # If start is small, we can pick the global largest ones.
                    # If start is large, we can only pick from the tail.
                    # The largest remaining_slots items in candidates[start:] are candidates[-remaining_slots:]
                    # provided start <= len(candidates) - remaining_slots.
                    if start <= len(candidates) - remaining_slots:
                         pass # Max prod is indeed product of last 'remaining_slots' items
                    else:
                         # We can only pick from candidates[start:]
                         # The max product is product of candidates[start:] if len <= remaining_slots?
                         # No, we must pick exactly remaining_slots.
                         # If len(candidates) - start < remaining_slots, it's impossible (handled by count logic? No, we check indices)
                         # Actually, if we are at 'start', we have len(candidates) - start items left.
                         # We need to pick remaining_slots.
                         # If len(candidates) - start < remaining_slots, return.
                         pass
                         
                    # Let's just rely on the recursive call to handle bounds naturally, 
                    # but add a strong check:
                    # If current_prod * (max possible product of remaining_slots distinct factors from candidates[start:]) < remaining_product
                    # The max possible product is the product of the largest remaining_slots factors in candidates[start:].
                    # These are candidates[-remaining_slots:] if start is small enough.
                    # Let's compute it properly.
                    available = candidates[start:]
                    if len(available) < remaining_slots:
                        return
                    max_prod_available = 1
                    for i in range(len(available) - remaining_slots, len(available)):
                        max_prod_available *= available[i]
                    if current_prod * max_prod_available < remaining_product:
                        return

                # Iterate through candidates
                for i in range(start, len(candidates)):
                    val = candidates[i]
                    if current_prod * val > remaining_product:
                        break # Since candidates are sorted, further values will be larger
                    result_subset.append(val)
                    backtrack(i + 1, current_prod * val, count + 1)
                    result_subset.pop()
                    if found:
                        return

            backtrack(0, 1, 0)
            
            if found:
                # The solution is [m] + result_subset
                # We must ensure they are distinct. 
                # m is not in candidates (candidates < m), so distinct from m.
                # result_subset elements are distinct by construction (indices increasing).
                sol = [m] + result_subset
                # Verify
                prod = 1
                for v in sol:
                    prod *= v
                if prod == target and len(set(sol)) == len(sol) and all(v > 12 for v in sol):
                    return sol

    # Fallback: return single factor
    return [target]
