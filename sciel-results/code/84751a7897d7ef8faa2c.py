from math import factorial

def construct():
    N = factorial(12)
    
    # Get all divisors of N that are > 12
    divisors = set()
    for i in range(13, int(N**0.5) + 1):
        if N % i == 0:
            divisors.add(i)
            other = N // i
            if other > 12:
                divisors.add(other)
    if N > 12:
        divisors.add(N)
    # Also need to check divisors <= sqrt(N) that are > 12?
    # The loop starts at 13. So if N % i == 0, i is added.
    # What if a divisor d < 13? Not added. Correct, we need > 12.
    # What if a divisor d is between 13 and sqrt(N)? Added.
    # What if a divisor d is > sqrt(N)? Added as 'other' when i = N//d < sqrt(N).
    # So the set contains all divisors > 12.
    
    divisors = sorted(list(divisors))
    
    best_solution = None
    best_max = float('inf')
    
    # Max k is 7 based on 13*14*...*20 > 12!
    # Let's verify max k dynamically or just use 7.
    # 13*14*15*16*17*18*19 = 253955520 < 479001600
    # 13*...*20 = 5079110400 > 479001600
    # So max k is 7.
    
    def dfs(remaining, start_idx, count_left, current_factors, current_max):
        nonlocal best_solution, best_max
        
        if count_left == 0:
            if remaining == 1:
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(current_factors)
            return
        
        if current_max >= best_max:
            return
            
        # Iterate divisors in reverse order (largest to smallest)
        # to find compressed solutions (small max) quickly.
        # We must pick f > divisors[start_idx - 1] if start_idx > 0
        # But since we are iterating indices, we ensure distinctness and increasing order by index.
        # Wait, if we iterate indices in reverse, we pick larger values first.
        # The condition "distinct and increasing" is handled by the index constraint?
        # No. If we pick index i, then next must be index > i?
        # If we pick f = divisors[i], the next factor must be > f.
        # So the next index must be > i.
        # If we iterate i from high to low, we are picking large f first.
        # Then we recurse with start_idx = i + 1.
        # This ensures the next factor is at index > i, so value > f.
        # This maintains the increasing order.
        
        for i in range(len(divisors) - 1, start_idx - 1, -1):
            f = divisors[i]
            if f > remaining:
                # Since we are going downwards, smaller f might be <= remaining.
                # But if f > remaining, it can't divide remaining (unless remaining is 0, which it isn't).
                # Actually if f > remaining, remaining % f != 0 (unless remaining=0).
                # So we can continue.
                continue
            
            if remaining % f != 0:
                continue
                
            rem = remaining // f
            
            if count_left - 1 == 0:
                if rem == 1:
                    new_max = max(current_max, f)
                    if new_max < best_max:
                        best_max = new_max
                        best_solution = current_factors + [f]
                continue
            
            # Pruning:
            # Min product of count_left-1 distinct integers > f
            # is (f+1)*(f+2)*...*(f+count_left-1)
            min_prod = 1
            for j in range(1, count_left):
                min_prod *= (f + j)
                if min_prod > rem:
                    break
            
            if min_prod > rem:
                # Since we are iterating f downwards, smaller f means:
                # 1. rem increases
                # 2. min_prod decreases
                # So this condition might become false for smaller f.
                # So we cannot break. We must continue.
                continue

            dfs(rem, i + 1, count_left - 1, current_factors + [f], max(current_max, f))

    for k in range(1, 8):
        dfs(N, 0, k, [], 0)
        
    if best_solution is None:
        return [N]
        
    return best_solution
