import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline for safety
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need g(x) for x in [1001, 1000024]
    # g(x) = x with all prime factors >= 24 removed (i.e., keep only 23-smooth part)
    
    max_val = 1000000 + 24  # 1000024
    min_val = 1000 + 1      # 1001
    
    # Precompute g(x) for all x in [min_val, max_val]
    # For each x, divide out all factors of primes in `primes`
    
    g = [0] * (max_val + 1)
    
    for x in range(min_val, max_val + 1):
        v = x
        for p in primes:
            while v % p == 0:
                v //= p
        g[x] = v
    
    # Now search for n in [1000, 1000000] minimizing distinct count of g(n+1)..g(n+24)
    best_n = 1000
    best_count = float('inf')
    
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        # Count distinct g values
        vals = set()
        for i in range(1, 25):
            vals.add(g[n + i])
        cnt = len(vals)
        if cnt < best_count:
            best_count = cnt
            best_n = n
            if best_count <= 2:  # Can't do better than 2 (or 1?)
                # Actually minimum could be 1 if all 24 have same 23-smooth part
                # But that's extremely unlikely. Let's keep searching a bit more or break
                pass
    
    return best_n
