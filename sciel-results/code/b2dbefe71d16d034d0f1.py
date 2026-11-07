import math

def construct():
    # 1. Identify primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(256)
    P_small = [p for p in primes if p <= 128]
    P_large = [p for p in primes if p > 128]
    
    # Map small primes to bit indices
    prime_to_idx = {p: i for i, p in enumerate(P_small)}
    
    # 2. Base sum: 1 and all large primes
    base_sum = 1 + sum(P_large)
    
    # 3. Construct list of remaining numbers R
    # Numbers 2..256 that are not in P_large
    # Note: Primes in P_large are excluded.
    # Any other number x in 2..256 must have all prime factors in P_small.
    # Why? If x had a prime factor p > 128, x would be a multiple of p.
    # Since p > 128, 2p > 256, so x must be p itself.
    # But we excluded P_large. So all remaining numbers have only small prime factors.
    
    items = []
    for x in range(2, 257):
        if x in P_large:
            continue
        # Compute mask for x
        mask = 0
        temp = x
        for p in P_small:
            if temp % p == 0:
                mask |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        # If temp > 1, it means there's a prime factor > 128?
        # This should not happen for x not in P_large.
        # But let's verify. If x=1, temp=1.
        # If x is prime > 128, it's in P_large and skipped.
        # So temp should always be 1.
        if temp > 1:
            # This case should be empty, but handle it just in case
            # Actually, if this happens, x has a large prime factor, so it conflicts only with itself?
            # No, if x has a large prime factor p, and x != p, then x is a multiple of p.
            # But 2p > 256. So x must be p.
            # So this block is dead code for x in 2..256 not in P_large.
            pass
            
        items.append((x, mask))
        
    # Sort items by value descending
    items.sort(key=lambda t: t[0], reverse=True)
    
    n_items = len(items)
    values = [v for v, m in items]
    masks = [m for v, m in items]
    
    # Precompute suffix sums for a loose bound
    suffix_sums = [0] * (n_items + 1)
    for i in range(n_items - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + values[i]
        
    best_sum = 0
    num_small_primes = len(P_small)
    
    def dfs(idx, used_mask, current_sum):
        nonlocal best_sum
        
        # Upper bound calculation
        # Loose bound: current_sum + suffix_sums[idx]
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Tighter bound: sum of top k compatible values
        # k = number of available primes
        # We can count bits not set in used_mask
        # available_primes = num_small_primes - bin(used_mask).count('1')
        # However, calculating top k requires sorting.
        # Given n_items <= 232, and depth <= 232, this might be okay.
        # But let's stick to the loose bound first to see if it's fast enough.
        # If not, we can optimize.
        # Actually, the loose bound might be too loose.
        # Let's use the tighter bound. It's safer for performance.
        
        if idx == n_items:
            if current_sum > best_sum:
                best_sum = current_sum
            return
        
        # Calculate tighter bound
        # Find compatible items from idx to end
        # And sum top 'available_primes' of them
        available_primes = num_small_primes - bin(used_mask).count('1')
        
        # Optimization: If available_primes is 0, we can't pick any more.
        if available_primes == 0:
            if current_sum > best_sum:
                best_sum = current_sum
            return
            
        # Collect compatible values
        compat_vals = []
        for j in range(idx, n_items):
            if (masks[j] & used_mask) == 0:
                compat_vals.append(values[j])
                if len(compat_vals) > available_primes:
                    # We only need top 'available_primes', but we don't know which are top yet.
                    # We can just collect all and sort later, or use a heap.
                    # Since n is small, collect all.
                    pass
                    
        # To speed up, we can just sum all compatible values as a bound?
        # No, we can pick at most 'available_primes' items.
        # So bound is sum of top 'available_primes' values in compat_vals.
        if compat_vals:
            compat_vals.sort(reverse=True)
            bound_add = sum(compat_vals[:available_primes])
        else:
            bound_add = 0
            
        if current_sum + bound_add <= best_sum:
            return
            
        # Option 1: Take item idx
        if (masks[idx] & used_mask) == 0:
            dfs(idx + 1, used_mask | masks[idx], current_sum + values[idx])
            
        # Option 2: Skip item idx
        dfs(idx + 1, used_mask, current_sum)
        
    dfs(0, 0, 0)
    
    return [1] + P_large + [0] * 0 # Placeholder, need to return the actual set?
    # Wait, the task asks to return a list of ints. The value is the sum.
    # "Return a list of ints. The raw value is their sum."
    # So I need to return the actual subset, not just the sum.
    # I need to modify the DFS to track the chosen items.
