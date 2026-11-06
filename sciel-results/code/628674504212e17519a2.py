"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # 12! = 479001600
    # Prime factorization: 2^10 * 3^5 * 5^2 * 7^1
    # We need distinct integers > 12, product = 12!
    # Minimize the largest factor
    
    # Strategy: Use backtracking search over divisors of 12! that are > 12
    # We want to find a set of distinct divisors > 12 whose product is 12!
    # and whose maximum is minimized.
    
    # Key insight: The factors must be divisors of 12! that are > 12.
    # Let's think about what's possible.
    # 12! = 479001600
    # 
    # If we use 2 factors: a * b = 12!, a > 12, b > 12, a != b
    # Minimize max(a,b). Best case: a ≈ b ≈ sqrt(12!) ≈ 21886
    # But they must be distinct divisors > 12.
    #
    # If we use more factors, we can potentially get smaller maximums.
    #
    # Let's use a systematic approach:
    # 1. Generate all divisors of 12! that are > 12
    # 2. Search for combinations of distinct divisors whose product is 12!
    # 3. Minimize the maximum element
    
    # First, let's generate prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1
    
    # Generate all divisors of 12!
    def get_divisors(n):
        """Generate all divisors of n using prime factorization."""
        # For 12!, we know the factorization
        primes = [2, 3, 5, 7]
        exponents = [10, 5, 2, 1]
        
        divisors = [1]
        for p, e in zip(primes, exponents):
            new_divisors = []
            for d in divisors:
                for k in range(e + 1):
                    new_divisors.append(d * (p ** k))
            divisors = new_divisors
        return sorted(divisors)
    
    all_divisors = get_divisors(12)  # This is 12! = 479001600
    # Filter: divisors > 12
    valid_divisors = [d for d in all_divisors if d > 12]
    
    # 12! has (10+1)*(5+1)*(2+1)*(1+1) = 11*6*3*2 = 396 divisors
    # After filtering > 12, we have fewer
    
    # Now search for subsets of distinct valid divisors whose product is 12!
    # This is a subset product problem.
    
    target = 479001600
    
    # Use backtracking with pruning
    # Sort valid divisors in ascending order
    valid_divisors.sort()
    
    best_max = target  # Initial: single factor 12!
    best_set = [target]
    
    # For efficiency, we can limit the number of factors to at most 12
    # and each factor must be > 12
    
    # Let's use a recursive backtracking approach
    # State: remaining product, list of available divisors (indices), current set
    
    # To make this tractable, let's use the prime exponent vector representation
    # Each divisor corresponds to a vector (e2, e3, e5, e7)
    # We need to select distinct vectors that sum to (10, 5, 2, 1)
    
    # Map each divisor to its exponent vector
    divisor_to_vec = {}
    vec_to_divisor = {}
    for d in valid_divisors:
        n = d
        vec = []
        for p in [2, 3, 5, 7]:
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            vec.append(count)
        vec = tuple(vec)
        divisor_to_vec[d] = vec
        vec_to_divisor[vec] = d
    
    # Target vector: (10, 5, 2, 1)
    target_vec = (10, 5, 2, 1)
    
    # Sort divisors by value (ascending) for better pruning
    sorted_divisors = sorted(valid_divisors)
    
    # Backtracking search
    # We'll try to find a combination with minimal maximum
    # Use iterative deepening on the maximum value
    
    # First, find the minimum possible maximum by checking feasibility
    # Binary search on the maximum value M: can we form 12! using distinct divisors all <= M and > 12?
    
    # For a given M, the available divisors are those in valid_divisors with d <= M
    # We need to check if there's a subset whose product is 12!
    
    # This is still NP-hard in general, but 12! has only 396 divisors total,
    # and after filtering > 12 and <= M, it's much smaller.
    
    # Let's use memoization with the remaining exponent vector and the set of used divisors
    # But the set of used divisors is large. Instead, let's use a different approach.
    
    # Since the total number of divisors is small (396), and we need at most 12 factors,
    # let's use a smarter search.
    
    # Key observation: The factors must multiply to 12!. If we have k factors,
    # their geometric mean is (12!)^(1/k). To minimize the maximum, we want
    # factors to be as balanced as possible.
    
    # Let's try different numbers of factors from 2 to 12, and for each,
    # find the combination that minimizes the maximum.
    
    # For a fixed number of factors k, we want to partition the prime exponents
    # into k distinct divisor vectors, each corresponding to a divisor > 12.
    
    # This is a complex combinatorial problem. Let's use a practical approach:
    # 1. For small k (2, 3, 4, ...), enumerate possible combinations
    # 2. Use pruning based on the remaining product
    
    # Let's start with k=2: find two distinct divisors a, b > 12 with a*b = 12!
    # a < b, minimize b. So maximize a subject to a < sqrt(12!) and a | 12!, a > 12
    # sqrt(12!) ≈ 21886.1
    
    # Find the largest divisor of 12! that is < sqrt(12!) and > 12
    sqrt_target = math.isqrt(target)
    
    # For k=2:
    best_k2_max = None
    for d in sorted_divisors:
        if d > sqrt_target:
            break
        if d > 12 and target % d == 0:
            other = target // d
            if other > 12 and other != d:
                if best_k2_max is None or other < best_k2_max:
                    best_k2_max = other
                    best_k2_set = [d, other]
    
    if best_k2_max is not None:
        if best_k2_max < best_max:
            best_max = best_k2_max
            best_set = best_k2_set
    
    # For k=3: find three distinct divisors a, b, c > 12 with a*b*c = 12!
    # Minimize max(a,b,c). Assume a <= b <= c, minimize c.
    # a*b must be a divisor of 12!/c, and c = 12!/(a*b)
    
    # This is getting complex. Let's use a more systematic backtracking approach
    # with the exponent vectors.
    
    # State: (remaining_exponents, min_index, current_factors, current_max)
    # We want to find any valid completion and track the best max.
    
    # Use a priority queue or iterative approach.
    
    # Given the time constraint, let's implement a focused backtracking search.
    
    # Represent state as (rem_vec, start_idx, factors_list)
    # rem_vec is the remaining exponent vector to be distributed
    
    # To avoid exponential blowup, limit the depth to 12 factors max.
    
    # Let's use a different strategy: iterate over possible maximum values M in ascending order,
    # and check if a valid decomposition exists with all factors <= M.
    # The first M that works gives the optimal answer.
    
    # For a given M, available divisors are those d where 12 < d <= M and d | 12!
    # Check if we can select distinct such divisors whose product is 12!
    
    # This is a subset product problem, which is NP-hard, but with small numbers it's feasible.
    
    # Let's implement a backtracking solver for the subset product problem.
    
    def can_form_product(avail_divs, target_val):
        """Check if we can form target_val from distinct elements of avail_divs."""
        if target_val == 1:
            return []
        
        # Sort avail_divs in descending order for better pruning
        avail = sorted(avail_divs, reverse=True)
        
        # If the product of all available is less than target, impossible
        prod_all = 1
        for d in avail:
            prod_all *= d
            if prod_all >= target_val:
                break
        if prod_all < target_val:
            return None
        
        # Backtracking
        def backtrack(idx, remaining, chosen):
            if remaining == 1:
                return chosen
            if idx >= len(avail):
                return None
            
            # Pruning: if remaining is prime and not in avail[idx:], check directly
            # Try including avail[idx]
            d = avail[idx]
            if remaining % d == 0:
                # Check if d is distinct from all chosen
                if d not in chosen:
                    result = backtrack(idx + 1, remaining // d, chosen + [d])
                    if result is not None:
                        return result
            
            # Try excluding avail[idx]
            result = backtrack(idx + 1, remaining, chosen)
            if result is not None:
                return result
            
            return None
        
        return backtrack(0, target_val, [])
    
    # Now iterate M from 13 upward, checking feasibility
    # But this could be slow. Let's be smarter.
    
    # The minimum possible M is at least ceil((12!)^(1/12)) since we can have at most 12 factors
    # (12!)^(1/12) ≈ 479001600^(1/12) ≈ 8.5, so M >= 9, but factors must be > 12, so M >= 13
    
    # Let's try M values that are divisors of 12! or just above.
    
    # Actually, let's just do the backtracking search directly, trying to minimize the max.
    
    # Better approach: Use the fact that we have at most 396 divisors, and search for
    # the optimal solution using branch and bound.
    
    # Let's implement a cleaner backtracking that tracks the current maximum.
    
    # For time efficiency, let's focus on small numbers of factors first.
    
    # k=2 is already handled above.
    
    # For k>=3, let's use a different approach: 
    # Generate candidate factor sets by splitting the prime exponents.
    
    # Actually, let's just run a time-bounded search.
    
    # Simple approach: try to find solutions with 2, 3, 4, ... factors
    # For each k, use backtracking to find the solution with minimum max.
    
    # Given the complexity, let's use a heuristic search with time limit.
    
    # Let's try a different tactic: since the answer for k=2 is already found,
    # let's see if we can do better with more factors.
    
    # For the current best, let's also try to verify and improve.
    
    # Let's implement a focused search for k=3,4,5 factors.
    
    def search_for_k_factors(k, max_time=30.0):
        """Search for k distinct factors > 12 whose product is 12!, minimizing the max."""
        start = time.time()
        best = None
        best_max = float('inf')
        
        # Use backtracking over the sorted divisors
        # We need to pick k distinct divisors from valid_divisors
        # whose product is target
        
        # To make this tractable, use the following:
        # Assume factors are sorted: f1 <= f2 <= ... <= fk
        # f1 * f2 * ... * fk = target
        # f1 >= 13 (since > 12)
        # fk is what we minimize
        
        # For a given fk (the maximum), the remaining k-1 factors must multiply to target/fk
        # and each must be <= fk and > 12 and distinct from fk
        
        # Let's iterate fk in ascending order over divisors of target that are > 12
        # For each fk, check if we can form target/fk from k-1 distinct divisors in [13, fk]
        
        for fk in sorted_divisors:
            if time.time() - start > max_time:
                break
            
            if fk <= 12:
                continue
            if target % fk != 0:
                continue
            
            rem = target // fk
            if rem < 1:
                continue
            
            # Check if rem can be formed from k-1 distinct divisors in [13, fk]
            # These divisors must also divide rem
            # And be distinct from fk (but since fk is the max and we're looking at divisors <= fk,
            # and fk divides target, fk might not divide rem unless fk^2 divides target)
            
            # Get candidate divisors for the remaining product
            # They must be > 12, <= fk, divide rem, and be distinct from fk
            
            candidates = []
            for d in sorted_divisors:
                if d <= 12:
                    continue
                if d > fk:
                    break
                if d == fk:
                    # Allow only if fk^2 divides target (i.e., fk divides rem)
                    # But factors must be distinct, so we can't use fk again
                    continue
                if rem % d == 0:
                    candidates.append(d)
            
            if len(candidates) < k - 1:
                continue
            
            # Now check if we can pick k-1 distinct elements from candidates whose product is rem
            # This is a subset product problem with small k
            
            def can_form_k_minus_1(cands, target_rem, count):
                """Can we form target_rem from 'count' distinct elements of cands?"""
                if count == 0:
                    return [] if target_rem == 1 else None
                if count == 1:
                    if target_rem in cands:
                        return [target_rem]
                    return None
                
                # Backtracking
                def bt(idx, rem_val, chosen):
                    if rem_val == 1:
                        if len(chosen) == count:
                            return chosen
                        return None
                    if idx >= len(cands):
                        return None
                    if len(chosen) + (len(cands) - idx) < count:
                        return None
                    
                    d = cands[idx]
                    # Try including d
                    if rem_val % d == 0 and d not in chosen:
                        result = bt(idx + 1, rem_val // d, chosen + [d])
                        if result is not None:
                            return result
                    
                    # Try excluding d
                    result = bt(idx + 1, rem_val, chosen)
                    if result is not None:
                        return result
                    
                    return None
                
                return bt(0, target_rem, [])
            
            result = can_form_k_minus_1(candidates, rem, k - 1)
            if result is not None:
                factors = result + [fk]
                cur_max = max(factors)
                if cur_max < best_max:
                    best_max = cur_max
                    best = factors
                    # Since we're iterating fk in ascending order, this is likely optimal
                    # But let's continue a bit to see if we can do better
                    if time.time() - start > max_time * 0.5:
                        break
        
        return best, best_max
    
    # Try k=3, 4, 5 with time limits
    for k in [3, 4, 5]:
        if time.time() > deadline - 10:
            break
        max_time_for_k = min(20.0, max(1.0, deadline - time.time() - 5))
        sol, sol_max = search_for_k_factors(k, max_time_for_k)
        if sol is not None and sol_max < best_max:
            best_max = sol_max
            best_set = sol
    
    # Verify the best solution
    if best_set is None:
        best_set = [target]
        best_max = target
    
    # Final verification
    prod = 1
    for f in best_set:
        prod *= f
    assert prod == target, f"Product {prod} != {target}"
    assert all(f > 12 for f in best_set), "All factors must be > 12"
    assert len(set(best_set)) == len(best_set), "Factors must be distinct"
    assert 1 <= len(best_set) <= 12, "Number of factors must be 1-12"
    
    return best_set
# EVOLVE-BLOCK-END
