"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

P = {'n': 256}

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed_ok():
        return time.time() - start_time < DEADLINE

    n = P['n']
    
    # We want a set of distinct integers from 1..n such that every pair is coprime.
    # This is equivalent to a Maximum Weight Independent Set (MWIS) in a graph where
    # vertices are 1..n, and edges connect non-coprime pairs.
    # However, 1 is coprime to everything, so it's always beneficial to include it (weight 1).
    # For numbers > 1, two numbers are coprime iff they share no prime factors.
    
    # Strategy:
    # 1. Precompute prime factors for all numbers 2..n.
    # 2. Use Branch and Bound (B&B) to find the MWIS.
    #    - State: set of available primes, current sum, current set.
    #    - But B&B on primes might be tricky because a number consumes multiple primes.
    #    - Alternative: B&B on numbers.
    #      - Sort numbers by weight (value) descending.
    #      - At each step, try including the current number if it's coprime to all selected.
    #      - Upper bound: current_sum + sum of weights of all remaining numbers that are coprime to current selection?
    #        This is an optimistic bound. We can tighten it by considering that remaining numbers must also be pairwise coprime.
    #        A better bound: Sum of weights of remaining numbers that are coprime to current selection, but this ignores mutual conflicts among remaining.
    #        Even better: For each prime p, at most one number in the final set can be divisible by p.
    #        So, a valid upper bound is: current_sum + sum over all "free" primes p of (max weight of number divisible by p among remaining candidates).
    #        Wait, a number can have multiple primes. If we pick a number, we "use" all its primes.
    #        This looks like a set packing problem.
        
    # Let's refine the B&B approach:
    # Vertices: 2..n (we add 1 at the end).
    # Sort vertices by value descending: v_1, v_2, ..., v_m.
    # We decide include/exclude for each v_i.
    # State: index i, set of used primes (bitmask if possible, but primes up to 256 are ~54, so bitmask fits in 64-bit int? 
    #        Primes <= 256: 2, 3, 5, ..., 251. Count is 54. Yes, fits in a Python int (arbitrary precision, but fast bitwise ops).
    #        
    # Upper Bound Calculation:
    # Given current used_primes mask, and remaining candidates v_i..v_m:
    # A simple bound: Sum of v_j for j>=i if (v_j's primes & used_primes) == 0. 
    # This is weak because it ignores conflicts between v_j and v_k.
    # 
    # Stronger Bound:
    # For each prime p NOT in used_primes, let M_p = max { v_j | j>=i, p divides v_j }.
    # The sum of M_p over all free primes p is an upper bound on the additional weight we can get?
    # NO. Because one number v_j consumes multiple primes. If we sum M_p, we might count the same number multiple times if it has multiple primes.
    # Example: v=6 (2,3). M_2=6, M_3=6. Sum=12. But we can only pick 6 once.
    # So sum(M_p) is an overestimate, which is good for a bound (must be >= true max).
    # Is it a valid upper bound? Yes, because any valid set S of remaining numbers corresponds to a set of primes P_S.
    # For each p in P_S, there is at most one number in S divisible by p.
    # The weight of that number is <= M_p.
    # So Sum_{v in S} v = Sum_{v in S} Sum_{p|v} (v / count_p(v))? No.
    # Actually, simply: For any valid set S, let p_v be the "primary" prime of v? No.
    # Let's look at the conflict graph. It's an intersection graph of sets of primes.
    # The clique cover number or chromatic number can provide bounds.
    # 
    # Let's try a different bound:
    # Greedy coloring of the remaining conflict graph? Too slow.
    # 
    # Let's stick to the "Sum of Max per Prime" bound.
    # Bound = Current_Sum + Sum_{p not in used} MaxWeight[p]
    # Where MaxWeight[p] is the max value of any remaining candidate divisible by p.
    # Why is this an upper bound?
    # Let S be any valid set of remaining candidates.
    # For each v in S, let p_v be the smallest prime factor of v (or any prime factor).
    # The mapping v -> p_v is not necessarily injective?
    # If v1 and v2 both map to p, then both are divisible by p. But they must be coprime, so they can't both be in S.
    # So the map v -> p_v is injective for any valid set S?
    # Yes! If v1, v2 in S and p_v1 = p_v2 = p, then p|v1 and p|v2, so gcd(v1,v2) >= p > 1, contradiction.
    # Therefore, each v in S is assigned a unique prime p_v.
    # And v <= MaxWeight[p_v] (by definition of MaxWeight for the remaining set).
    # So Sum_{v in S} v <= Sum_{v in S} MaxWeight[p_v] <= Sum_{p not in used} MaxWeight[p].
    # This bound is VALID and tight enough?
    # It might be loose if a number has multiple primes and is the max for all of them.
    # But it's much better than summing all weights.
    
    # Implementation Plan:
    # 1. Identify primes <= n.
    # 2. Map each prime to an index 0..K-1.
    # 3. For each number x in 2..n, compute its prime mask.
    # 4. Sort numbers 2..n by value descending.
    # 5. B&B DFS:
    #    - i: current index in sorted list
    #    - used_mask: bitmask of used primes
    #    - current_sum
    #    - best_sum: global max found
    #    - best_set: global best set
    #    
    #    At node (i, used_mask, current_sum):
    #    - If i == m: update best if current_sum > best_sum.
    #    - Calculate Upper Bound:
    #      UB = current_sum
    #      For each prime p not in used_mask:
    #         Find max value of v_j (j>=i) such that p|v_j.
    #         UB += that max value.
    #      If UB <= best_sum: prune.
    #    - Option 1: Include v_i if (mask[v_i] & used_mask) == 0.
    #      Recurse(i+1, used_mask | mask[v_i], current_sum + v_i)
    #    - Option 2: Exclude v_i.
    #      Recurse(i+1, used_mask, current_sum)
    
    # Optimization:
    # Precomputing the "MaxWeight for remaining suffix" for each prime is O(K*m) per node, which is too slow.
    # We need to update MaxWeight efficiently or use a simpler bound.
    
    # Alternative Bound:
    # Sum of weights of remaining candidates that are coprime to used_mask.
    # This is O(m) per node. m=255. Nodes can be many.
    # Let's check if the "Sum of Max per Prime" can be precomputed.
    # Let M[i][p] = max value of v_j (j>=i) divisible by p.
    # We can precompute this table!
    # M has size m x K. m=255, K=54. 255*54 ~ 13k entries. Very fast.
    # Then UB calculation is:
    # UB = current_sum
    # for p in 0..K-1:
    #   if not (used_mask & (1<<p)):
    #      UB += M[i][p]
    # This is O(K) per node. Very fast.
    
    # Let's implement this.

    # 1. Primes and Masks
    def sieve_primes(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, flag in enumerate(is_prime) if flag]

    primes = sieve_primes(n)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    K = len(primes)

    # 2. Numbers and Masks
    candidates = list(range(2, n + 1))
    candidates.sort(reverse=True) # Descending order
    m = len(candidates)
    
    masks = []
    for x in candidates:
        mask = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        masks.append(mask)

    # 3. Precompute M[i][p]
    # M[i][p] = max(candidates[j] for j in range(i, m) if candidates[j] % p == 0) else 0
    # Note: candidates[j] is the value.
    # We can compute this backwards.
    # M[m][p] = 0 for all p.
    # M[i][p] = max(candidates[i] if p|candidates[i] else 0, M[i+1][p])
    
    M = [[0]*K for _ in range(m+1)]
    for i in range(m-1, -1, -1):
        val = candidates[i]
        mask = masks[i]
        for p_idx in range(K):
            if mask & (1 << p_idx):
                M[i][p_idx] = max(val, M[i+1][p_idx])
            else:
                M[i][p_idx] = M[i+1][p_idx]

    # 4. B&B
    best_sum = 0
    best_set = []
    
    # We include 1 always.
    base_sum = 1
    base_set = [1]
    
    # Global variables for recursion
    import sys
    sys.setrecursionlimit(10000)
    
    def bound(i, used_mask):
        ub = 0
        for p in range(K):
            if not (used_mask & (1 << p)):
                ub += M[i][p]
        return ub

    def dfs(i, used_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if not elapsed_ok():
            return
            
        if i == m:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return

        # Prune
        # Upper bound for the rest
        ub_rest = bound(i, used_mask)
        if current_sum + ub_rest <= best_sum:
            return

        # Option 1: Include candidates[i]
        if not (used_mask & masks[i]):
            # Check time
            if elapsed_ok():
                current_set.append(candidates[i])
                dfs(i+1, used_mask | masks[i], current_sum + candidates[i], current_set)
                current_set.pop()
        
        # Option 2: Exclude candidates[i]
        if elapsed_ok():
            dfs(i+1, used_mask, current_sum, current_set)

    # Start search
    dfs(0, 0, 0, [])

    # Result
    result = base_set + best_set
    # Ensure types are int and list
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
