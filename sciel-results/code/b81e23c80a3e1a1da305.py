def construct():
    from math import isqrt
    
    LIMIT = 10**18
    powerful_set = set()
    
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(10**6)
    
    def gen(idx, current, max_val):
        powerful_set.add(current)
        if idx >= len(primes):
            return
        p = primes[idx]
        val = current * p * p
        while val <= max_val:
            gen(idx + 1, val, max_val)
            val *= p
    
    gen(0, 1, LIMIT)
    
    powerful_list = sorted(powerful_set)
    
    candidates = set()
    for i in range(0, 300):
        candidates.add(10**18 - i * 10**6)
    for a in powerful_list[:80]:
        for b in powerful_list[:80]:
            s = a + b
            if s <= LIMIT:
                candidates.add(s)
    
    candidates = list(candidates)
    
    best_N = 2
    best_count = 0
    best_pairs = []
    
    for N in candidates:
        if N < 2 or N > LIMIT:
            continue
        pairs = []
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                pairs.append([a, b])
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs
    
    return {'N': best_N, 'pairs': best_pairs}
