import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 256
    # Primes
    primes = []
    for i in range(2, n + 1):
        is_prime = True
        for d in range(2, int(i**0.5) + 1):
            if i % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    # Use more small primes to get better optimization
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_small = len(small_primes)
    num_masks = 1 << num_small
    large_primes = [p for p in primes if p > 37]
    
    # Map small prime to bit index
    sp_index = {p: i for i, p in enumerate(small_primes)}
    
    # Precompute prime powers for small primes
    sp_powers = {}
    for p in small_primes:
        power = p
        while power * p <= n:
            power *= p
        sp_powers[p] = power
        
    # Precompute options for each large prime
    # Option: (small_mask, value)
    options = {}
    for q in large_primes:
        opts = []
        # Option 0: skip q
        opts.append((0, 0))
        
        # Option: use q itself
        opts.append((0, q))
        
        # Other multiples: only those where k has no large prime factors
        for k in range(2, n // q + 1):
            m = q * k
            # Check if k only has small prime factors
            temp = k
            for p in small_primes:
                while temp % p == 0:
                    temp //= p
            if temp != 1:
                # k has a large prime factor, skip
                continue
            # Find small prime factors of m (same as k since q has no small factors)
            mask = 0
            temp = k
            for p in small_primes:
                if temp % p == 0:
                    mask |= (1 << sp_index[p])
            opts.append((mask, m))
        options[q] = opts
        
    # DP over masks
    dp = [-10**9] * num_masks
    dp[0] = 0
    
    num_large = len(large_primes)
    history = [ [None]*num_masks for _ in range(num_large) ]
    
    for i, q in enumerate(large_primes):
        new_dp = [-10**9] * num_masks
        for mask in range(num_masks):
            if dp[mask] < 0:
                continue
            for j, (smask, val) in enumerate(options[q]):
                if (mask & smask) == 0:
                    new_mask = mask | smask
                    new_val = dp[mask] + val
                    if new_val > new_dp[new_mask]:
                        new_dp[new_mask] = new_val
                        history[i][new_mask] = (mask, j)
        dp = new_dp
        
    # Find best final mask
    best_mask = 0
    best_val = -1
    for mask in range(num_masks):
        if dp[mask] < 0: continue
        sum_free = 0
        for i, p in enumerate(small_primes):
            if not (mask & (1 << i)):
                sum_free += sp_powers[p]
        total = dp[mask] + sum_free + 1
        if total > best_val:
            best_val = total
            best_mask = mask
            
    # Backtrack to get choices
    result = [1]
    current_mask = best_mask
    for i in range(num_large - 1, -1, -1):
        entry = history[i][current_mask]
        if entry is None:
            continue
        prev_mask, opt_idx = entry
        q = large_primes[i]
        smask, val = options[q][opt_idx]
        if val > 0:
            result.append(val)
        current_mask = prev_mask
        
    # Add free small prime powers
    for i, p in enumerate(small_primes):
        if not (best_mask & (1 << i)):
            result.append(sp_powers[p])
            
    return [int(x) for x in result]
