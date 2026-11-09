def construct():
    from collections import defaultdict
    import math
    
    LIMIT = 10**18
    
    def gen_powerful(max_val):
        results = set([1])
        
        def _gen(current, min_idx):
            for i in range(min_idx, len(primes)):
                p = primes[i]
                if current * p * p > max_val:
                    break
                val = current * p * p
                results.add(val)
                _gen(val, i)  # Allow same prime again for higher powers
                # Continue to next prime with current
            if min_idx + 1 < len(primes):
                _gen(current, min_idx + 1)
        
        max_prime = int(math.isqrt(max_val)) + 1
        sieve = [True] * (max_prime + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(max_prime)) + 1):
            if sieve[i]:
                for j in range(i*i, max_prime + 1, i):
                    sieve[j] = False
        primes = [i for i in range(2, max_prime + 1) if sieve[i]]
        
        _gen(1, 0)
        return sorted([x for x in results if x <= max_val])
    
    powerful = gen_powerful(LIMIT)
    powerful_set = set(powerful)
    n = len(powerful)
    
    sum_count = defaultdict(int)
    for i in range(n):
        a = powerful[i]
        for j in range(i, n):
            b = powerful[j]
            s = a + b
            if s > LIMIT:
                break
            sum_count[s] += 1
    
    best_N = max(sum_count.keys(), key=lambda n: sum_count[n])
    pairs = [[a, best_N - a] for a in powerful if 2 * a <= best_N and (best_N - a) in powerful_set]
    
    return {'N': best_N, 'pairs': pairs}
