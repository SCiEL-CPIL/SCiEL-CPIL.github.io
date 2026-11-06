import time
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    timeout = 85.0
    
    target = factorial(12)
    
    # Get all divisors
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
            
    # Filter valid divisors: > 12
    valid_divs = [d for d in divisors if d > 12]
    valid_divs.sort() # Ascending order for easier distinctness handling (pick increasing indices)
    
    n_divs = len(valid_divs)
    
    # Helper to check if a number is prime (for pruning)
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def can_decompose(M):
        # Find candidates <= M
        # valid_divs is sorted ascending
        # Find the rightmost index where valid_divs[idx] <= M
        # We can use bisect_right
        import bisect
        end_idx = bisect.bisect_right(valid_divs, M)
        candidates = valid_divs[:end_idx]
        
        if not candidates:
            return False
            
        # If target itself is in candidates, it's a valid solution (1 factor)
        # But we want to minimize max, so smaller M is better.
        # This check is implicit in the search.
        
        # Memoization key: (idx, remaining)
        # However, remaining can be large.
        # Let's use a set for failed states (idx, remaining) to prune.
        failed_states = set()
        
        def backtrack(idx, remaining, count):
            if remaining == 1:
                return True
            if count >= 12:
                return False
            if idx >= len(candidates):
                return False
            
            # Pruning:
            # 1. If remaining is prime, it must be one of the candidates.
            if is_prime(remaining):
                # Check if remaining is in candidates[idx:]
                # Since candidates is sorted, we can binary search
                pos = bisect.bisect_left(candidates, remaining, idx)
                if pos < len(candidates) and candidates[pos] == remaining:
                    return True
                return False
                
            # 2. If remaining is not divisible by any candidate in candidates[idx:], fail.
            # This is expensive to check fully, but we can do it in the loop.
            
            # 3. Max product of all remaining candidates?
            # If we pick all, product is huge. Not useful for lower bound.
            # Min product? 1.
            
            # 4. If remaining > product of all candidates[idx:], fail?
            # No, we don't have to pick all.
            
            # 5. If remaining < candidates[idx], fail?
            # No, we can skip candidates[idx].
            
            # 6. Important Pruning: 
            # If remaining is not divisible by candidates[i], skip.
            
            # 7. Another Pruning:
            # If we have `k` factors left to pick (count to 12), the minimum possible value for the next factor is candidates[idx].
            # The maximum possible value for the next factor is candidates[-1] (which is <= M).
            # This doesn't help much directly.
            
            # Let's try to force a solution if remaining is in candidates.
            # If remaining is in candidates[idx:], we can just take it and return True (if count < 12).
            # This is a very strong heuristic.
            pos = bisect.bisect_left(candidates, remaining, idx)
            if pos < len(candidates) and candidates[pos] == remaining:
                # We can pick this one and finish.
                # Is it distinct? Yes, because index > idx.
                # Is it <= M? Yes, because it's in candidates.
                # Is count < 12? Yes, checked above.
                return True

            for i in range(idx, len(candidates)):
                c = candidates[i]
                if c > remaining:
                    break # Since sorted ascending, no further candidates can divide remaining
                if remaining % c == 0:
                    if backtrack(i + 1, remaining // c, count + 1):
                        return True
            return False
        
        return backtrack(0, target, 0)

    # Binary search for M
    # Lower bound: 13
    # Upper bound: target
    low = 13
    high = target
    
    # We can narrow the search space.
    # For k=12 factors, max factor is at least (12!)^(1/12) ~ 7. So 13 is a good lower bound.
    # For k=1 factor, max factor is 12!.
    
    # Let's try a linear search from 13 to 1000 first, as solutions with small max factors are likely to have many factors.
    found_M = None
    for M in range(13, 1000):
        if time.time() - start_time > timeout - 10:
            break
        if can_decompose(M):
            found_M = M
            break
            
    if found_M is None:
        # Binary search in [1000, target]
        low = 1000
        high = target
        while low <= high:
            if time.time() - start_time > timeout - 10:
                break
            mid = (low + high) // 2
            if can_decompose(mid):
                found_M = mid
                high = mid - 1
            else:
                low = mid + 1

    if found_M is None:
        return [target]
        
    # Reconstruct solution
    def find_solution(M):
        import bisect
        end_idx = bisect.bisect_right(valid_divs, M)
        candidates = valid_divs[:end_idx]
        
        res = []
        
        def backtrack(idx, remaining, count):
            if remaining == 1:
                return True
            if count >= 12:
                return False
            if idx >= len(candidates):
                return False
            
            # Quick check: if remaining is in candidates[idx:]
            pos = bisect.bisect_left(candidates, remaining, idx)
            if pos < len(candidates) and candidates[pos] == remaining:
                res.append(remaining)
                return True

            for i in range(idx, len(candidates)):
                c = candidates[i]
                if c > remaining:
                    break
                if remaining % c == 0:
                    res.append(c)
                    if backtrack(i + 1, remaining // c, count + 1):
                        return True
                    res.pop()
            return False
        
        if backtrack(0, target, 0):
            return res
        return None

    sol = find_solution(found_M)
    
    if sol is None:
        return [target]
        
    return sol
