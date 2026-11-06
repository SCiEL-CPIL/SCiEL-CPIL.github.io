"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def construct():
    target = math.factorial(12)  # 479001600
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (evaluator kills at 90s)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    primes = [2, 3, 5, 7, 11]
    target_exponents = (10, 5, 2, 1, 1)
    
    # Generate all divisors of target
    divisors = []
    def gen_divs(idx, current_val, exps_tuple):
        if idx == len(primes):
            if current_val > 12:
                divisors.append((current_val, exps_tuple))
            return
        p = primes[idx]
        max_exp = target_exponents[idx]
        val = 1
        for e in range(max_exp + 1):
            new_exps = list(exps_tuple)
            new_exps[idx] = e
            gen_divs(idx + 1, current_val * val, tuple(new_exps))
            val *= p
            
    gen_divs(0, 1, (0, 0, 0, 0, 0))
    
    # Sort divisors by value
    divisors.sort(key=lambda x: x[0])
    
    # Extract values and exponents
    divisor_vals = [d[0] for d in divisors]
    divisor_exps = [d[1] for d in divisors]
    
    # Map value to index for quick lookup if needed, but we iterate by index
    # We need to check if a number is a divisor? 
    # In the DFS, we only iterate over precomputed divisors, so no need for a set of all divisors for membership, 
    # but we do need to ensure we pick distinct factors. 
    # The "distinct" constraint is handled by picking factors in increasing order (index i+1 > i).
    
    # Binary search on the maximum factor M
    # Lower bound: smallest divisor > 12
    lo = divisor_vals[0] if divisor_vals else 13
    # Upper bound: target (trivial solution)
    hi = target
    
    best_result = [target]
    best_max = target
    
    # We want to find the smallest M such that a valid decomposition exists.
    # The predicate "can_decompose(M)" is monotonic: if it's true for M, it's true for M+1.
    # So we can binary search.
    
    def can_decompose(M):
        # Filter candidates <= M
        # Find the last index in divisor_vals that is <= M
        # Since divisor_vals is sorted, we can use bisect
        import bisect
        right_idx = bisect.bisect_right(divisor_vals, M)
        if right_idx == 0:
            return None
        
        # Candidates are divisor_vals[0:right_idx]
        # We will use a memoized DFS.
        # State: (remaining_exps, start_idx)
        # remaining_exps is a tuple of 5 integers.
        # start_idx is the index in the candidates list to start from.
        
        # To make memoization efficient, we map the state to a boolean.
        # However, the number of states might be large. 
        # Let's use lru_cache on a helper function.
        
        # We need to access the candidate lists inside the cached function.
        # Closures in lru_cache can be tricky if the lists change, 
        # but here they are fixed for a specific M.
        # We can define the function inside can_decompose to capture the filtered lists.
        
        cand_vals = divisor_vals[:right_idx]
        cand_exps = divisor_exps[:right_idx]
        n_cands = len(cand_vals)
        
        # If the product of all candidates is less than target, it's impossible? 
        # No, we don't have to use all. But if we can't form the target, it's false.
        # A quick check: if target > M, and we need at least 1 factor, 
        # and the largest factor is M, then if target > M and we can't use more factors to reduce the max?
        # Actually, if target > M, we MUST use multiple factors.
        # The product of factors is target. All factors <= M.
        # If M < target, we need at least 2 factors.
        
        @lru_cache(maxsize=None)
        def dfs(rem_exps, start_idx):
            # rem_exps is a tuple
            if all(e == 0 for e in rem_exps):
                return ()  # Empty tuple indicates success
            
            # If we have run out of candidates, fail
            if start_idx >= n_cands:
                return None
            
            # Pruning: 
            # The remaining product must be formable by distinct factors from cand_vals[start_idx:]
            # A simple lower bound check: 
            # The product of the remaining factors must be at least the product of the smallest 
            # available factors? No, that's not a tight bound.
            # However, we can check if the maximum possible product from remaining candidates is < target_remaining?
            # Or if the minimum possible product (using smallest available) > target_remaining?
            
            # Let's just try all candidates from start_idx to n_cands
            for i in range(start_idx, n_cands):
                f_val = cand_vals[i]
                f_exps = cand_exps[i]
                
                # Check if f_exps <= rem_exps
                valid = True
                for j in range(5):
                    if rem_exps[j] < f_exps[j]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Subtract
                new_rem = tuple(rem_exps[j] - f_exps[j] for j in range(5))
                
                # Recurse
                result = dfs(new_rem, i + 1)
                if result is not None:
                    return (f_val,) + result
            
            return None
        
        result = dfs(target_exponents, 0)
        return result
    
    # Binary search
    # We know that M=target is always a solution (the trivial one [target]).
    # We want the minimum M.
    
    # Optimization: Instead of standard binary search which might probe large M values first 
    # and waste time on complex DFS, we can try a few small M values first 
    # because the optimal solution is likely small (around 50-100).
    
    # Let's try a range of M values from low to high.
    # The theoretical lower bound for M with k factors is (target)^(1/k).
    # For k=12, M >= 35.
    # Let's test M values from 35 to 500 first.
    
    found = False
    # Test small M values
    for M in range(35, 500):
        if time.time() > deadline - 5.0:
            break
        # Only test M if it's a divisor? No, M is just a bound. 
        # But the factors must be divisors. 
        # If M is not a divisor, the set of candidates is the same as for the largest divisor < M.
        # So we can just iterate through the unique candidate counts.
        
        # Actually, just calling can_decompose(M) is fine. 
        # But to save time, we can skip M values that don't add new candidates.
        # The candidates are divisors. 
        # So we can just iterate through the divisor values as potential M bounds.
        
        # Let's modify the loop to iterate through divisor values directly.
        # This ensures we only test M when the candidate set changes.
        
        # However, the binary search logic is safer. 
        # Let's stick to binary search but with a good initial guess.
        
        # Let's just run the binary search.
        pass
    
    # Reset found
    found = False
    
    # To speed up, let's try to find a solution for a small M first.
    # We'll iterate through the sorted divisors as potential M bounds.
    # This is O(N_divs) calls to can_decompose, which might be too slow if N_divs is large (792).
    # But can_decompose is fast if M is small (few candidates).
    
    # Let's try M values corresponding to the divisors.
    # We'll go from smallest divisor to largest.
    # If we find a solution, that's our best (smallest M).
    
    # But wait, if we find a solution for M=100, we don't need to check M=101, 102... 
    # until we are sure no smaller M works. 
    # Since we are iterating from small to large, the first success is the minimum M.
    
    # Is it possible that a solution exists for M=100 but not for M=105? 
    # No, if a solution exists for M=100, the same factors work for M=105.
    # So the predicate is monotonic.
    
    # So we can just iterate M from low to high and return the first success.
    # This avoids the overhead of binary search probes that might fail or succeed unpredictably 
    # in terms of time cost (large M -> many candidates -> slow DFS).
    # Small M -> few candidates -> fast DFS.
    
    # So, iterate through unique divisor values as M.
    
    # Get unique sorted divisors
    unique_divs = sorted(set(divisor_vals))
    
    # We know M must be at least the smallest divisor.
    # We also know that if M < target, we need multiple factors.
    
    for M in unique_divs:
        if time.time() > deadline - 5.0:
            break
        
        # If M is very large, the DFS might be slow.
        # We can add a heuristic: if M > 1000, switch to binary search?
        # Or just let it run. The number of candidates for M=1000 is still small.
        
        result = can_decompose(M)
        if result is not None:
            best_result = result
            best_max = M
            found = True
            break
            
    if not found:
        # If no solution found in the loop (should not happen as M=target is a solution),
        # or if we timed out, return the best known.
        # The trivial solution is always valid.
        best_result = [target]
        best_max = target

    return best_result
# EVOLVE-BLOCK-END
