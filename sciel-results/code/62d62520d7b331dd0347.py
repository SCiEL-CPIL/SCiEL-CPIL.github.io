"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt
from functools import lru_cache

def construct():
    target = factorial(12)
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # Indices: 0:2, 1:3, 2:5, 3:7, 4:11
    target_exp = (10, 5, 2, 1, 1)
    primes = (2, 3, 5, 7, 11)
    
    # Precompute all divisors of 12!
    # A divisor is determined by exponents e_i where 0 <= e_i <= target_exp_i
    divisors = []
    def gen_divisors(idx, curr_exp, curr_val):
        if idx == len(primes):
            if curr_val > 12:
                divisors.append(curr_val)
            return
        for e in range(target_exp[idx] + 1):
            gen_divisors(idx + 1, curr_exp + (e,), curr_val * (primes[idx] ** e))
    
    gen_divisors(0, (), 1)
    
    # Sort divisors in descending order
    # We want to pick larger factors first to minimize the maximum factor?
    # Actually, if we are searching for a solution with max factor M,
    # we want to find *any* set of distinct divisors from [13, M] that multiply to target.
    # Sorting descending helps prune if we realize we can't reach the product?
    # Or sorting ascending?
    # Let's sort descending. If we pick a large factor, the remaining product is smaller,
    # which might be easier to decompose into smaller distinct factors?
    # Or maybe ascending is better to keep the "remaining" large for longer?
    # Let's try descending first, as it's a common heuristic for "minimize max" type problems
    # (greedy largest first).
    
    divisors.sort(reverse=True)
    
    # Map value to exponent vector for fast lookup
    # But we need to generate valid factors for a given M.
    # Let's pre-filter divisors > 12.
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort(reverse=True)
    
    # Map divisor to its exponent vector
    # We can compute this on the fly or precompute.
    # Precomputing a dict: value -> exp_tuple
    div_exp_map = {}
    for d in valid_divisors:
        exps = [0]*5
        temp = d
        for i, p in enumerate(primes):
            while temp % p == 0:
                exps[i] += 1
                temp //= p
        div_exp_map[d] = tuple(exps)
    
    # We will use a recursive backtracking search.
    # State: 
    #   remaining_exp: tuple of 5 ints
    #   max_allowed: int (M)
    #   used_set: frozenset of used divisors
    #   min_idx: index in valid_divisors to start from (to ensure distinctness and avoid re-visiting)
    #   Actually, since valid_divisors is sorted, we can just use an index.
    #   But we need to ensure we don't pick the same value twice.
    #   Since we iterate from idx to end, and values are distinct, this ensures distinctness.
    
    # Optimization: 
    # If remaining_exp is (0,0,0,0,0), return [].
    # If no more divisors, return None.
    
    # Pruning:
    # 1. If the product of the remaining prime powers is > 1, we need at least one more factor.
    # 2. The smallest possible factor we can pick is valid_divisors[last_idx].
    #    If remaining product < min_possible_factor, fail? No, remaining product is fixed.
    #    If remaining product is prime and not in the available divisors, fail.
    
    # Let's implement a helper to check if a number is prime (small numbers)
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i+2) == 0:
                return False
            i += 6
        return True
    
    # We will perform a binary search on M (the maximum allowed factor).
    # Lower bound: 13
    # Upper bound: max(valid_divisors) which is 12!
    
    # However, checking every M is expensive.
    # We can iterate M from 13 upwards?
    # The optimal M is likely not too large.
    # 12! = 479001600
    # If we use 1 factor, M = 479001600.
    # If we use 2 factors, min M is roughly sqrt(12!) ~ 21886.
    # If we use 3 factors, min M is roughly cbrt(12!) ~ 782.
    # If we use 4 factors, min M is roughly 4th root ~ 263.
    # If we use 5 factors, min M is roughly 5th root ~ 137.
    # If we use 6 factors, min M is roughly 6th root ~ 96.
    # If we use 7 factors, min M is roughly 7th root ~ 76.
    # If we use 8 factors, min M is roughly 8th root ~ 63.
    # If we use 9 factors, min M is roughly 9th root ~ 54.
    # If we use 10 factors, min M is roughly 10th root ~ 47.
    # If we use 11 factors, min M is roughly 11th root ~ 42.
    # If we use 12 factors, min M is roughly 12th root ~ 37.
    
    # Since factors must be > 12, and distinct, the minimum possible max factor is bounded by the k-th root.
    # We should try small M first.
    # Let's iterate M from 13 to, say, 1000.
    # If found, return.
    # If not found, binary search between 1000 and target.
    
    def can_decompose_with_max(M, record=False):
        """
        Returns a list of factors if a valid decomposition exists with all factors <= M, else None.
        """
        # Filter divisors <= M
        # valid_divisors is sorted descending.
        # Find the cutoff index where divisors <= M.
        # Since it's descending, divisors[:k] are > M, divisors[k:] are <= M.
        # We want divisors <= M, so we want the slice starting from the first element <= M.
        
        # Binary search for the first index where valid_divisors[idx] <= M
        lo, hi = 0, len(valid_divisors)
        while lo < hi:
            mid = (lo + hi) // 2
            if valid_divisors[mid] > M:
                lo = mid + 1
            else:
                hi = mid
        start_idx = lo
        
        if start_idx >= len(valid_divisors):
            return None if not record else []
        
        # The candidates are valid_divisors[start_idx:]
        # We will backtrack on this list.
        
        solution = []
        
        def backtrack(idx, remaining_exp):
            if remaining_exp == (0, 0, 0, 0, 0):
                return True
            
            # If we have run out of candidates
            if idx >= len(valid_divisors):
                return False
            
            # Pruning:
            # Calculate the remaining product value
            rem_val = 1
            for i in range(5):
                rem_val *= primes[i] ** remaining_exp[i]
                
            # If rem_val is 1, handled above.
            # If rem_val is prime, it must be one of the candidates.
            # Also, rem_val must be <= M (which is guaranteed if it's a divisor and we picked valid ones? No, rem_val is a divisor of target, so if it's prime, it's a specific number. If that number is > M, we fail. But we only pick candidates <= M. If rem_val is prime and > M, we fail. If rem_val is prime and <= M, we must pick it if it's available.)
            
            # If rem_val < valid_divisors[-1] (smallest candidate), we can't form it?
            # No, we can multiply smaller numbers.
            # But all candidates are >= valid_divisors[-1].
            # If rem_val < valid_divisors[-1], and rem_val != 1, we can't form rem_val using factors >= valid_divisors[-1] unless rem_val is 1.
            # Wait, if rem_val = 13, and min candidate is 13, we pick 13.
            # If rem_val = 14, and min candidate is 13, we can pick 13? No, 14/13 is not integer.
            # So if rem_val < min_candidate, fail.
            
            min_candidate = valid_divisors[-1]
            if rem_val < min_candidate:
                return False
            
            # Iterate candidates from idx to end
            # To improve search, we might want to try larger factors first?
            # The list is sorted descending. So valid_divisors[idx] is the largest available.
            # Trying larger factors first reduces the remaining product faster.
            
            for i in range(idx, len(valid_divisors)):
                d = valid_divisors[i]
                # d must divide remaining_exp
                # Check if d's exponents are <= remaining_exp
                d_exp = div_exp_map[d]
                
                valid = True
                for j in range(5):
                    if d_exp[j] > remaining_exp[j]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Subtract exponents
                new_exp = tuple(remaining_exp[j] - d_exp[j] for j in range(5))
                
                # Recurse
                # Note: we must use distinct factors. Since we iterate i increasing (in the subarray),
                # and each value is unique, we automatically ensure distinctness.
                # We pass i+1 to ensure we don't reuse d or any larger d (which are < idx in the full list, but we started at idx).
                # Wait, valid_divisors is global.
                # We started at start_idx.
                # Inside backtrack, we iterate from current idx.
                # So we pick d = valid_divisors[i].
                # Next call must start from i+1 to ensure we only pick smaller/equal divisors?
                # No, we need distinct integers.
                # If we pick d, we cannot pick d again.
                # Since all values in valid_divisors are distinct, picking index i means value d.
                # The next pick must be from indices > i to avoid picking d again?
                # Yes, because if we pick index j < i, we might pick a value already used?
                # No, we haven't used j yet if j < i?
                # Actually, if we sort descending, and we pick index i, the "used" set is {valid_divisors[k] for k in [start_idx, i-1]}?
                # No, we are doing a depth-first search.
                # Path: pick idx1, then idx2 > idx1, etc.
                # This ensures we pick a subset of indices in increasing order.
                # This corresponds to a set of distinct values.
                # So yes, recurse with i+1.
                
                solution.append(d)
                if backtrack(i + 1, new_exp):
                    return True
                solution.pop()
                
            return False
        
        if backtrack(start_idx, target_exp):
            if record:
                return solution
            else:
                return True
        else:
            if record:
                return None
            else:
                return False

    # Search strategy:
    # 1. Try M from 13 upwards up to a limit (e.g., 200).
    # 2. If not found, binary search between 200 and target.
    
    best_solution = None
    best_M = target
    
    # Phase 1: Linear scan for small M
    # This is fast because the candidate list is small.
    for M in range(13, 200):
        if time.time() > deadline:
            break
        if can_decompose_with_max(M, record=False):
            sol = can_decompose_with_max(M, record=True)
            if sol:
                best_solution = sol
                best_M = M
                break
    
    # Phase 2: Binary search for larger M if not found in Phase 1
    if best_solution is None:
        lo = 200
        hi = target
        while lo <= hi and time.time() < deadline:
            mid = (lo + hi) // 2
            if can_decompose_with_max(mid, record=False):
                best_M = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if best_M < target:
            sol = can_decompose_with_max(best_M, record=True)
            if sol:
                best_solution = sol
    
    # Fallback
    if best_solution is None:
        # If we still don't have a solution, return the single factor [target]
        # This is always valid: 1 factor, distinct, > 12.
        best_solution = [target]
    
    return best_solution

# EVOLVE-BLOCK-END
