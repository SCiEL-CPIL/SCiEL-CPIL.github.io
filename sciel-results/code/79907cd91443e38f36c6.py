"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for the 60s wall-clock limit

    # We need distinct denominators in [5, 10000000] whose reciprocal sum is exactly 1.
    # Minimize count. Theoretical minimum is 8 (since 7 terms max sum < 1).
    
    # Fast search using integer arithmetic.
    # State: (remaining_num, remaining_den, min_den, current_list)
    # We want remaining_num/remaining_den = sum of 1/d for d in remaining terms.
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Try to find 8-term solution with DFS and tight pruning
    # remaining = num/den (in lowest terms)
    # At each step, pick d >= min_den such that 1/d <= remaining
    # New remaining = remaining - 1/d
    
    def search(remaining_num, remaining_den, min_den, terms_left, current, deadline):
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            if remaining_num == 0:
                return list(current)
            return None
        
        if remaining_num == 0:
            return None  # need more terms but sum already reached
        
        if remaining_num < 0:
            return None
        
        if terms_left == 1:
            # Need 1/d = remaining_num/remaining_den
            # d = remaining_den / remaining_num must be integer
            if remaining_den % remaining_num == 0:
                d = remaining_den // remaining_num
                if d >= min_den and d >= 5 and d <= 10000000:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
            return None
        
        # Pruning: max achievable with terms_left terms starting from min_den
        # Sum of 1/(min_den + i) for i in range(terms_left)
        # Compute as fraction for exactness
        max_sum_num, max_sum_den = 0, 1
        for i in range(terms_left):
            d = min_den + i
            # max_sum += 1/d
            new_num = max_sum_num * d + max_sum_den
            new_den = max_sum_den * d
            g = gcd(new_num, new_den)
            max_sum_num, max_sum_den = new_num // g, new_den // g
        
        # Check: remaining <= max_sum
        # remaining_num/remaining_den <= max_sum_num/max_sum_den
        # remaining_num * max_sum_den <= max_sum_num * remaining_den
        if remaining_num * max_sum_den > max_sum_num * remaining_den:
            return None
        
        # Lower bound on d: 1/d <= remaining => d >= remaining_den / remaining_num
        # d >= ceil(remaining_den / remaining_num)
        d_lower = max(min_den, (remaining_den + remaining_num - 1) // remaining_num)
        
        # Upper bound on d: we need remaining - 1/d to be achievable with terms_left-1 terms
        # The max achievable with terms_left-1 terms starting from d+1 must be >= remaining - 1/d
        # Also, 1/d must be <= remaining, so d >= ceil(remaining_den/remaining_num) [already handled]
        # For upper bound: as d increases, 1/d decreases, so remaining - 1/d increases.
        # We need remaining - 1/d <= max_achievable(terms_left-1, d+1)
        # This gives a constraint that d can't be too small (which we have) but doesn't easily give upper bound.
        # However, d can't be too large either because we need enough remaining for the other terms.
        # The minimum sum with terms_left-1 terms starting from d+1 is very small, so d can be large.
        # But practically, d is bounded by the need to have remaining - 1/d > 0 and achievable.
        
        # Additional pruning: d must be such that remaining - 1/d > 0
        # 1/d < remaining => d > remaining_den/remaining_num (strict)
        # So d >= floor(remaining_den/remaining_num) + 1 if remaining_den/remaining_num is not integer
        # Actually d >= ceil(remaining_den/remaining_num) already ensures 1/d <= remaining
        
        # Try denominators
        for d in range(d_lower, 10000001):
            if time.time() > deadline:
                return None
            
            # 1/d <= remaining?
            # 1/d <= remaining_num/remaining_den => remaining_den <= d * remaining_num
            if d * remaining_num < remaining_den:
                continue
            
            # New remaining = remaining_num/remaining_den - 1/d
            # = (remaining_num * d - remaining_den) / (remaining_den * d)
            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d
            
            if new_num < 0:
                continue
            
            # Reduce fraction
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            # Check if new remaining can be achieved with terms_left-1 terms from d+1
            if terms_left - 1 == 0:
                if new_num == 0:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
                continue
            
            # Max achievable with terms_left-1 terms from d+1
            max_rest_num, max_rest_den = 0, 1
            for i in range(terms_left - 1):
                dd = d + 1 + i
                new_n = max_rest_num * dd + max_rest_den
                new_d = max_rest_den * dd
                gg = gcd(new_n, new_d)
                max_rest_num, max_rest_den = new_n // gg, new_d // gg
            
            # Check: new_num/new_den <= max_rest_num/max_rest_den
            # new_num * max_rest_den <= max_rest_num * new_den
            if new_num * max_rest_den > max_rest_num * new_den:
                break  # For larger d, new_num/new_den increases, so no point continuing
            
            current.append(d)
            result = search(new_num, new_den, d + 1, terms_left - 1, current, deadline)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    # Try 8 terms first (theoretical minimum)
    result = search(1, 1, 5, 8, [], deadline)
    if result is not None:
        # Verify
        total_num, total_den = 0, 1
        for d in result:
            total_num = total_num * d + total_den
            total_den = total_den * d
            g = gcd(total_num, total_den)
            total_num //= g
            total_den //= g
        if total_num == total_den and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
            return result
    
    # Try 9 terms
    if time.time() < deadline - 5:
        result = search(1, 1, 5, 9, [], deadline)
        if result is not None:
            total_num, total_den = 0, 1
            for d in result:
                total_num = total_num * d + total_den
                total_den = total_den * d
                g = gcd(total_num, total_den)
                total_num //= g
                total_den //= g
            if total_num == total_den and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Try 10 terms
    if time.time() < deadline - 3:
        result = search(1, 1, 5, 10, [], deadline)
        if result is not None:
            total_num, total_den = 0, 1
            for d in result:
                total_num = total_num * d + total_den
                total_den = total_den * d
                g = gcd(total_num, total_den)
                total_num //= g
                total_den //= g
            if total_num == total_den and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: greedy algorithm
    remaining_num, remaining_den = 1, 1
    result = []
    min_d = 5
    for _ in range(256):
        if remaining_num == 0:
            break
        if time.time() > deadline:
            break
        d = max(min_d, (remaining_den + remaining_num - 1) // remaining_num)
        while d * remaining_num < remaining_den:
            d += 1
        if d > 10000000:
            break
        result.append(d)
        remaining_num = remaining_num * d - remaining_den
        remaining_den = remaining_den * d
        g = gcd(remaining_num, remaining_den)
        if g > 1:
            remaining_num //= g
            remaining_den //= g
        min_d = d + 1
    
    if remaining_num == 0 and len(result) >= 1:
        return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
