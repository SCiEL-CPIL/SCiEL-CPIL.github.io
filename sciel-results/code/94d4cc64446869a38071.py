import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    MAX_N = 1000000
    MIN_N = 10000
    limit = MAX_N + 1
    
    # SPF Sieve
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts for all n up to limit
    # This avoids recalculating in the main loop if we were doing it differently,
    # but here we just calculate on the fly or precompute array.
    # Precomputing array pp[0..limit] is O(N log N) and uses memory.
    # Given N=1e6, memory is fine (8MB for list of ints).
    # This allows O(1) lookup in the main loop.
    
    pp = [1] * (limit + 1)
    
    # Calculate pp for all numbers
    # Using the property: pp(n) = pp(n // p^e) * p^e if e >= 2 else pp(n // p)
    # This requires knowing p^e.
    # We can just factorize each n using spf.
    
    # To speed up, we can iterate n from 1 to limit
    # and use spf to strip factors.
    
    # Actually, the current program calculates pp(n) and pp(n+1) on the fly.
    # Precomputing the whole array might be faster because we can reuse sub-results?
    # No, pp(n) depends on pp(n/p^e), which is smaller.
    # So we can compute pp in increasing order.
    
    # Let's do it.
    for n in range(2, limit + 1):
        p = spf[n]
        # Find p^e
        x = n
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        
        if e >= 2:
            pp[n] = pp[x] * (p ** e)
        else:
            pp[n] = pp[x]
            
    best_n = MIN_N
    best_val = 0
    
    # Scan for max product
    # We need pp[n] * pp[n+1]
    # n ranges from MIN_N to MAX_N - 1
    
    for n in range(MIN_N, MAX_N):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
        if time.time() > deadline:
            break
            
    return best_n
