"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    N = 256

    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Factorize each number 1..N into its set of distinct prime factors
    def factorize(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Build the set of all candidate numbers (1..256)
    candidates = list(range(1, N + 1))
    cand_factors = {}
    for c in candidates:
        if c == 1:
            cand_factors[c] = set()
        else:
            cand_factors[c] = factorize(c)

    # Two numbers are compatible if gcd(a,b) == 1, i.e., they share no prime factor
    # This is a Maximum Weight Independent Set problem on a graph where
    # vertices are numbers 1..256, edges connect incompatible pairs (gcd > 1),
    # and we want the max-weight independent set.
    
    # The graph is large (256 vertices), so exact MWS is hard in general.
    # However, the structure is special: incompatibility is determined by shared prime factors.
    # Each prime p "conflicts" all numbers divisible by p.
    
    # Key insight: We can use a greedy + local search approach,
    # but structured around prime-factor conflicts.
    
    # Alternative: Since the problem decomposes by prime factors,
    # we can use a different exact approach:
    # For each prime p, at most one selected number can be divisible by p.
    # This is a packing problem.
    
    # Let's use a different structural approach:
    # 1. Always include 1 (it conflicts with nothing, adds 1).
    # 2. For primes p where p^2 > N (i.e., p > 16), the only multiples in [1,N] 
    #    that are divisible by p are: p itself (and possibly p*2, p*3, etc. if <= N).
    #    Actually for p > 16: multiples are p, 2p, 3p, ... up to N.
    #    But 2p > 32, 3p > 48, etc. Many of these share factors with small primes.
    
    # Let me reconsider. The current DP approach is actually quite good for the
    # "small primes" part. Let me try a completely different structural idea:
    # 
    # Use Integer Linear Programming via a simple branch-and-bound with strong bounds,
    # but organized differently: process primes in descending order of "conflict potential."
    #
    # Actually, let me try a fundamentally different approach:
    # GREEDY WITH BACKTRACKING, processing numbers in descending order of value,
    # using bitmasks for prime factor conflicts.
    #
    # Since N=256, the number of distinct primes involved is small.
    # Primes up to 256: there are 54 primes.
    # But we can only select numbers that are pairwise coprime.
    # 
    # New approach: Maximum Weight Matching in a bipartite graph
    # where left side = "prime power slots" and right side = "composite slots."
    # This doesn't quite work either.
    
    # Let me try yet another approach: 
    # The problem is equivalent to: select a subset S of {1,...,256} such that
    # for every prime p, at most one element of S is divisible by p.
    # Maximize sum(S).
    #
    # This is a 0-1 integer program with 256 variables and 54 constraints.
    # We can solve it with a branch-and-bound that branches on primes.
    
    # For each prime p, the constraint is: sum_{n divisible by p} x_n <= 1
    # where x_n is 1 if n is selected.
    
    # Branch-and-bound:
    # - Sort primes by "impact" (number of multiples in [1,256])
    # - For each prime, decide which multiple (if any) to select
    # - Use upper bound = current_sum + sum of all remaining unconflicted numbers
    
    # Actually, let me try a simpler but structurally different approach:
    # Local search / simulated annealing starting from the DP solution.
    # This is a different optimization method (stochastic local search vs exact DP).
    
    # First, get the DP solution as a starting point (reuse current logic)
    small_p_list = [2, 3, 5, 7, 11, 13]
    n_small = len(small_p_list)
    full_mask = (1 << n_small) - 1
    
    mask_val = [0] * (1 << n_small)
    mask_valid = [False] * (1 << n_small)
    mask_val[0] = 0
    mask_valid[0] = True
    
    for mask in range(1, 1 << n_small):
        lsb = mask & -mask
        idx = (lsb.bit_length() - 1)
        p = small_p_list[idx]
        prev_mask = mask ^ lsb
        if mask_valid[prev_mask]:
            prod = mask_val[prev_mask] * p
            if prod <= N:
                mask_val[mask] = prod
                mask_valid[mask] = True
        else:
            mask_val[mask] = 0
            mask_valid[mask] = False
            
    dp = [-1] * (1 << n_small)
    dp[0] = 0
    
    for mask in range(1, 1 << n_small):
        sub = mask
        best = -1
        while sub:
            if mask_valid[sub]:
                if dp[mask ^ sub] != -1:
                    val = mask_val[sub] + dp[mask ^ sub]
                    if val > best:
                        best = val
            sub = (sub - 1) & mask
        dp[mask] = best

    def reconstruct(mask):
        if mask == 0:
            return []
        sub = mask
        best_val = -1
        best_sub = -1
        while sub:
            if mask_valid[sub]:
                if dp[mask ^ sub] != -1:
                    val = mask_val[sub] + dp[mask ^ sub]
                    if val > best_val:
                        best_val = val
                        best_sub = sub
            sub = (sub - 1) & mask
        if best_sub == -1:
            return []
        primes_in_sub = []
        for i in range(n_small):
            if best_sub & (1 << i):
                primes_in_sub.append(small_p_list[i])
        return [mask_val[best_sub]] + reconstruct(mask ^ best_sub)
    
    small_selection = reconstruct(full_mask)
    large_primes = [p for p in primes if p > 16]
    base_selection = list(large_primes)
    initial_selection = base_selection + small_selection + [1]
    initial_selection = list(set(initial_selection))
    
    # Now do local search: try to improve by swapping or adding elements.
    # Key idea: the DP only considers products of small primes.
    # But there might be better solutions using composite numbers that include
    # large primes combined with small ones, or prime powers of small primes.
    
    # Let me check: is it ever beneficial to use p^k (k>1) for a small prime p?
    # For p=2: 2^8=256. Value 256. Vs using 2 in a product. 
    # If we use 256, we consume prime 2 entirely. Value 256.
    # If we use 2*3*5*7=210, we consume 2,3,5,7. Value 210.
    # Then 11 and 13 are separate: 11+13=24. Total: 210+24=234.
    # Vs 256 + 3*5*7*11*13? 3*5*7*11*13=15015 > 256. No.
    # 256 + 3*5*7=105 + 11*13=143. 105+143=248. Total: 256+248=504.
    # Vs 2*3*5*7=210 + 11*13=143. Total: 353.
    # So 256 + 105 + 143 = 504 > 353!
    
    # This means the DP approach (which only considers square-free products of small primes)
    # is SUBOPTIMAL! It never considers prime powers like 256=2^8.
    
    # This is the key insight for a structural improvement:
    # We should ALSO consider prime powers of small primes as candidates.
    # For each small prime p, the candidates are: p, p^2, p^3, ..., p^k <= 256.
    # We need to decide, for each small prime, whether to use it as a prime power
    # or as part of a composite product.
    
    # Let's redo the DP to include prime powers as alternatives.
    # For each small prime p, the choices are:
    #   - Use p^k (max power <= 256) as a standalone number (consumes only p)
    #   - Use p as part of a composite product with other small primes
    #
    # This becomes a more complex DP. Let me think about it differently.
    #
    # Actually, the cleanest approach: enumerate ALL valid "groups" (subsets of 
    # small primes where the product <= 256, OR single prime powers), and find
    # the best partition.
    #
    # But this is getting complicated. Let me just do a full search over all
    # possible selections for the 6 small primes, where each prime can be:
    #   - Part of a composite group (with other small primes)
    #   - A standalone prime power
    #
    # Since there are only 6 small primes, and each can be in one of several states,
    # we can enumerate.
    
    # Better approach: enumerate all possible "assignments" of the 6 small primes
    # to groups, where each group is either:
    #   - A set of primes whose product <= 256 (square-free composite)
    #   - A single prime p with value p^k (max power <= 256)
    #
    # The number of such assignments is manageable for 6 primes.
    
    # Let's precompute all valid "items" for the small primes:
    # An item is (frozenset_of_primes, value)
    # where value is either the product of the primes (if square-free) or p^k (if single prime power).
    
    # Actually, for a single prime p, the best standalone value is p^k where p^k <= 256.
    # For a set of primes S with |S| >= 2, the value is prod(S) if prod(S) <= 256.
    
    # Let's enumerate all valid items:
    items = []  # list of (frozenset, value)
    
    # Single prime powers
    for p in small_p_list:
        max_power = p
        while max_power * p <= N:
            max_power *= p
        items.append((frozenset([p]), max_power))
    
    # Composite products (|S| >= 2)
    for mask in range(1, 1 << n_small):
        if mask_valid[mask]:
            n_bits = bin(mask).count('1')
            if n_bits >= 2:
                primes_in_mask = set()
                for i in range(n_small):
                    if mask & (1 << i):
                        primes_in_mask.add(small_p_list[i])
                items.append((frozenset(primes_in_mask), mask_val[mask]))
    
    # Now find the best partition of {2,3,5,7,11,13} into disjoint items
    # This is a set partition problem. With 6 elements, we can do it with DP over subsets.
    
    # dp[mask] = max total value for partitioning the primes in mask
    dp2 = [-1] * (1 << n_small)
    dp2[0] = 0
    
    # Precompute: for each mask, which items exactly cover it?
    # An item (frozenset S, value v) covers mask if S == set(mask)
    # But items are defined by their prime set. Let's map frozenset -> mask.
    
    fs_to_mask = {}
    for fs in [frozenset([p]) for p in small_p_list]:
        mask = 0
        for i, p in enumerate(small_p_list):
            if p in fs:
                mask |= (1 << i)
        fs_to_mask[fs] = mask
    
    # For composite items
    for mask in range(1, 1 << n_small):
        if mask_valid[mask] and bin(mask).count('1') >= 2:
            fs = frozenset(small_p_list[i] for i in range(n_small) if mask & (1 << i))
            fs_to_mask[fs] = mask
    
    # Now DP:
    for mask in range(1, 1 << n_small):
        best = -1
        # Try each item that is a subset of mask
        sub = mask
        while sub:
            # Check if sub corresponds to a valid item
            fs = frozenset(small_p_list[i] for i in range(n_small) if sub & (1 << i))
            if fs in fs_to_mask and fs_to_mask[fs] == sub:
                # Find the value for this item
                # It's either a prime power or a composite product
                # We stored it in items, but let's just recompute
                if len(fs) == 1:
                    p = list(fs)[0]
                    max_power = p
                    while max_power * p <= N:
                        max_power *= p
                    val = max_power
                else:
                    val = 1
                    for p in fs:
                        val *= p
                    if val <= N:
                        val = val
                    else:
                        val = -1  # invalid
                if val > 0:
                    if dp2[mask ^ sub] != -1:
                        total = val + dp2[mask ^ sub]
                        if total > best:
                            best = total
            sub = (sub - 1) & mask
        dp2[mask] = best
    
    max_small_sum_v2 = dp2[full_mask]
    
    # Reconstruct
    def reconstruct2(mask):
        if mask == 0:
            return []
        sub = mask
        best_val = -1
        best_sub = -1
        while sub:
            fs = frozenset(small_p_list[i] for i in range(n_small) if sub & (1 << i))
            if fs in fs_to_mask and fs_to_mask[fs] == sub:
                if len(fs) == 1:
                    p = list(fs)[0]
                    max_power = p
                    while max_power * p <= N:
                        max_power *= p
                    val = max_power
                else:
                    val = 1
                    for p in fs:
                        val *= p
                    if val > N:
                        val = -1
                if val > 0:
                    if dp2[mask ^ sub] != -1:
                        total = val + dp2[mask ^ sub]
                        if total > best_val:
                            best_val = total
                            best_sub = sub
            sub = (sub - 1) & mask
        if best_sub == -1:
            return []
        fs = frozenset(small_p_list[i] for i in range(n_small) if best_sub & (1 << i))
        if len(fs) == 1:
            p = list(fs)[0]
            max_power = p
            while max_power * p <= N:
                max_power *= p
            return [max_power] + reconstruct2(mask ^ best_sub)
        else:
            val = 1
            for p in fs:
                val *= p
            return [val] + reconstruct2(mask ^ best_sub)
    
    small_selection_v2 = reconstruct2(full_mask)
    
    # Compare with old approach
    old_sum = sum(small_selection)
    new_sum = sum(small_selection_v2)
    
    if new_sum >= old_sum:
        final_small = small_selection_v2
    else:
        final_small = small_selection
    
    final_selection = list(set(base_selection + final_small + [1]))
    
    # Verify pairwise coprimality
    valid = True
    for i in range(len(final_selection)):
        for j in range(i + 1, len(final_selection)):
            if math.gcd(final_selection[i], final_selection[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        result = [1]
        for p in primes:
            power = p
            while power * p <= N:
                power *= p
            result.append(power)
        return [int(x) for x in result]
    
    return [int(x) for x in final_selection]

# EVOLVE-BLOCK-END
