import time
from math import factorial
from functools import lru_cache
import bisect

def construct():
    start_time = time.time()
    timeout = 85.0
    
    target = factorial(12)  # 479001600
    
    # Get all divisors of 12!
    divisors = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter: must be > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    n_divs = len(valid_divs)
    
    # Precompute divisibility: for each valid divisor, which other valid divisors divide it
    # Actually, let's use a different approach: track prime exponent vector
    
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^2 * 11^1
    # Represent a number by its exponent vector (e2, e3, e5, e7, e11)
    
    def prime_exponents(n):
        """Return (e2, e3, e5, e7, e11) for n."""
        e = [0, 0, 0, 0, 0]
        for i, p in enumerate([2, 3, 5, 7, 11]):
            while n % p == 0:
                e[i] += 1
                n //= p
        return tuple(e)
    
    target_exp = prime_exponents(target)  # (10, 5, 2, 2, 1)
    
    # For each valid divisor, precompute its exponent vector
    div_exp = {}
    for d in valid_divs:
        div_exp[d] = prime_exponents(d)
    
    # Check if exp_a - exp_b is valid (all components >= 0)
    def can_subtract(a, b):
        return all(a[i] >= b[i] for i in range(5))
    
    def subtract_exp(a, b):
        return tuple(a[i] - b[i] for i in range(5))
    
    # Check if exponent vector represents 1
    def is_one(exp):
        return all(e == 0 for e in exp)
    
    # Check if exponent vector represents a prime
    def is_prime_exp(exp):
        total = sum(exp)
        if total <= 1:
            return False
        # A number is prime if its exponent vector has exactly one non-zero component equal to 1
        non_zero = [e for e in exp if e > 0]
        return len(non_zero) == 1 and non_zero[0] == 1
    
    # For the search, we'll use memoization on (remaining_exp, min_index, count)
    # But remaining_exp has many possibilities. Let's use a set for failed states.
    
    def can_decompose(M):
        """Check if we can decompose target into distinct divisors all <= M."""
        if time.time() - start_time > timeout - 5:
            return False
        
        # Get candidates: valid divisors <= M
        end_idx = bisect.bisect_right(valid_divs, M)
        candidates = valid_divs[:end_idx]
        
        if not candidates:
            return False
        
        # Quick check: if target itself is a candidate, we're done (1 factor)
        if target <= M:
            return True
        
        failed_states = set()
        
        def backtrack(idx, remaining_exp, count, used_set):
            if time.time() - start_time > timeout - 5:
                return False
            
            if is_one(remaining_exp):
                return True
            if count >= 12:
                return False
            if idx >= len(candidates):
                return False
            
            # Check memo
            state_key = (idx, remaining_exp, count)
            if state_key in failed_states:
                return False
            
            # Quick win: if remaining_exp corresponds to a divisor in candidates[idx:]
            # that is <= M, we can take it directly
            rem_val = 1
            for i, p in enumerate([2, 3, 5, 7, 11]):
                rem_val *= p ** remaining_exp[i]
            
            if rem_val > 12 and rem_val <= M:
                # Check if rem_val is in candidates[idx:]
                pos = bisect.bisect_left(candidates, rem_val, idx)
                if pos < len(candidates) and candidates[pos] == rem_val:
                    return True
            
            # Pruning: if remaining_exp represents a prime, it must be in candidates
            if is_prime_exp(remaining_exp):
                pos = bisect.bisect_left(candidates, rem_val, idx)
                if pos < len(candidates) and candidates[pos] == rem_val:
                    return True
                failed_states.add(state_key)
                return False
            
            # Try each candidate from idx onwards
            for i in range(idx, len(candidates)):
                c = candidates[i]
                c_exp = div_exp[c]
                
                # Check if c_exp can be subtracted from remaining_exp
                if not can_subtract(remaining_exp, c_exp):
                    continue
                
                new_remaining = subtract_exp(remaining_exp, c_exp)
                
                # Check if new_remaining can be formed from candidates[i+1:]
                # Basic pruning: if new_remaining is not 1, we need at least one more factor
                if not is_one(new_remaining) and i + 1 >= len(candidates):
                    continue
                
                if backtrack(i + 1, new_remaining, count + 1, used_set | {c}):
                    return True
            
            failed_states.add(state_key)
            return False
        
        return backtrack(0, target_exp, 0, frozenset())
    
    # Binary search for the minimum M
    # Lower bound: 13 (smallest valid divisor)
    # Upper bound: target (12! itself)
    
    # First, try a linear scan from 13 upward for a quick solution
    found_M = None
    for M in range(13, 500):
        if time.time() - start_time > timeout - 10:
            break
        if can_decompose(M):
            found_M = M
            break
    
    if found_M is None:
        # Binary search in [500, target]
        low = 500
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
        # Fallback: return [target]
        return [target]
    
    # Reconstruct the actual solution
    def find_solution(M):
        end_idx = bisect.bisect_right(valid_divs, M)
        candidates = valid_divs[:end_idx]
        
        res = []
        
        def backtrack(idx, remaining_exp, count):
            if time.time() - start_time > timeout - 5:
                return False
            
            if is_one(remaining_exp):
                return True
            if count >= 12:
                return False
            if idx >= len(candidates):
                return False
            
            # Quick win
            rem_val = 1
            for i, p in enumerate([2, 3, 5, 7, 11]):
                rem_val *= p ** remaining_exp[i]
            
            if rem_val > 12 and rem_val <= M:
                pos = bisect.bisect_left(candidates, rem_val, idx)
                if pos < len(candidates) and candidates[pos] == rem_val:
                    res.append(rem_val)
                    return True
            
            for i in range(idx, len(candidates)):
                c = candidates[i]
                c_exp = div_exp[c]
                
                if not can_subtract(remaining_exp, c_exp):
                    continue
                
                new_remaining = subtract_exp(remaining_exp, c_exp)
                
                res.append(c)
                if backtrack(i + 1, new_remaining, count + 1):
                    return True
                res.pop()
            
            return False
        
        if backtrack(0, target_exp, 0):
            return res
        return None
    
    sol = find_solution(found_M)
    
    if sol is None:
        return [target]
    
    # Verify
    prod = 1
    for x in sol:
        prod *= x
    assert prod == target, f"Product mismatch: {prod} != {target}"
    assert len(sol) == len(set(sol)), "Factors not distinct"
    assert all(x > 12 for x in sol), "Some factor <= 12"
    assert 1 <= len(sol) <= 12, "Wrong number of factors"
    
    return sol
