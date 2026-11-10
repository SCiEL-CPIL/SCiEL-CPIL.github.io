"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin before 60s kill
    
    n = 256
    
    # Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    num_primes = len(primes)
    
    # For each prime, find the largest power <= 256
    max_power = {}
    for p in primes:
        mp = p
        while mp * p <= n:
            mp *= p
        max_power[p] = mp
    
    # Precompute prime factor masks for all numbers 2..256
    # mask is a bitmask over the primes list
    def get_mask(x):
        mask = 0
        for idx, p in enumerate(primes):
            if x % p == 0:
                mask |= (1 << idx)
        return mask
    
    masks = [0] * (n + 1)
    for x in range(2, n + 1):
        masks[x] = get_mask(x)
    
    # Upper bound: sum of max_power for all primes + 1 (for the number 1)
    global_upper = sum(max_power.values()) + 1
    
    # Branch and bound
    # We process primes in order. At each step, we decide which number to assign
    # to the current prime's "slot". The number must be a power of that prime
    # OR a composite that includes that prime and no previously assigned primes.
    # Actually, a cleaner formulation:
    # Each prime p must be "covered" by exactly one number in the set that is divisible by p.
    # That number can be p^k, or p^k * q^a * ... where q are primes not yet assigned.
    # But wait, if a number covers multiple primes, it "uses up" those primes.
    #
    # Let's use a different B&B approach:
    # State: which primes are still "free" (not covered by any chosen number).
    # We choose numbers one by one. A valid number must have all its prime factors in the free set.
    # When we choose a number, we remove all its prime factors from the free set.
    # Goal: maximize sum of chosen numbers.
    #
    # Upper bound at any node: current_sum + sum of max_power[p] for all free primes p.
    # (This is valid because each free prime p can contribute at most max_power[p] to the sum,
    #  and these contributions are disjoint since each number uses at least one free prime.)
    # Actually this upper bound might be loose if a number uses multiple primes, but it's valid.
    
    # Initial free set: all primes
    all_free_mask = (1 << num_primes) - 1
    
    # Precompute for each possible number (2..256) its mask and value
    # Group numbers by their mask to avoid duplicates
    # For each mask, we only need the maximum value number with that mask
    mask_to_max_val = {}
    for x in range(2, n + 1):
        m = masks[x]
        if m not in mask_to_max_val or x > mask_to_max_val[m]:
            mask_to_max_val[m] = x
    
    # Also include the number 1 with mask 0
    # mask_to_max_val[0] = 1  # but 1 doesn't consume any prime
    
    # Sort candidate numbers by value descending for better pruning
    # Actually, for B&B we want to explore high-value options first
    candidates = []
    for m, v in mask_to_max_val.items():
        if m != 0:  # skip mask 0 (number 1) for now, handle separately
            candidates.append((m, v))
    # Sort by value descending
    candidates.sort(key=lambda x: -x[1])
    
    # Precompute max_power_sum for any subset of primes? Too expensive for 65 primes.
    # Instead, compute on the fly or precompute for the full set and subtract.
    # Let's precompute the contribution of each prime.
    prime_contrib = [0] * num_primes
    for idx, p in enumerate(primes):
        prime_contrib[idx] = max_power[p]
    
    # For the upper bound, we need sum of prime_contrib[i] for all free i.
    # We can maintain this incrementally.
    
    best_sum = 0
    best_set = [1]  # Always include 1
    
    # DFS with branch and bound
    # free_mask: bitmask of primes not yet covered
    # current_sum: sum of values chosen so far
    # chosen: list of chosen numbers
    
    # To speed up, we can precompute for each prime index the list of candidate indices
    # that include that prime. But let's start simple.
    
    def dfs(free_mask, current_sum, chosen):
        nonlocal best_sum, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if free_mask == 0:
            # All primes covered
            total = current_sum + 1  # add 1
            if total > best_sum:
                best_sum = total
                best_set = chosen + [1]
            return
        
        # Upper bound: current_sum + 1 + sum of prime_contrib for free primes
        ub = current_sum + 1
        fm = free_mask
        while fm:
            lowbit = fm & (-fm)
            idx = lowbit.bit_length() - 1
            ub += prime_contrib[idx]
            fm ^= lowbit
        
        if ub <= best_sum:
            return
        
        # Try each candidate that is a subset of free_mask
        # A candidate (m, v) is valid if (m & ~free_mask) == 0
        # i.e., all primes in m are in free_mask
        for m, v in candidates:
            if (m & ~free_mask) != 0:
                continue
            # This candidate is valid
            new_free = free_mask & ~m
            dfs(new_free, current_sum + v, chosen + [v])
            
            # Prune: if time is up, stop
            if time.time() > deadline:
                return
    
    dfs(all_free_mask, 0, [])
    
    # If no solution found (shouldn't happen), use the simple prime powers
    if not best_set or best_sum == 0:
        result = [1]
        for p in primes:
            result.append(max_power[p])
    else:
        result = best_set
    
    # Validation
    if len(result) != len(set(result)):
        return [1]
    if any(x < 1 or x > 256 for x in result):
        return [1]
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                return [1]
    
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
