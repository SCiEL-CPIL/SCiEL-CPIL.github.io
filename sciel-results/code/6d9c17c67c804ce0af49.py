import math

def construct():
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89]
    limit = 10**15
    
    exponents = [0] * len(primes)
    current_log_N = 0.0
    limit_log_N = math.log(limit)
    
    # Greedy
    for _ in range(100):
        best_gain = -1
        best_idx = -1
        for i in range(len(primes)):
            e = exponents[i]
            cost = math.log(primes[i])
            if current_log_N + cost > limit_log_N:
                continue
            old_term = math.log(e + 1)
            new_term = math.log(e + 2)
            gain = new_term - old_term
            ratio = gain / cost
            if ratio > best_gain:
                best_gain = ratio
                best_idx = i
        if best_idx == -1:
            break
        exponents[best_idx] += 1
        current_log_N += math.log(primes[best_idx])
        
    N = 1
    for i, p in enumerate(primes):
        N *= p ** exponents[i]
        
    # Find pairs
    pairs = []
    sqrt_N = math.isqrt(N)
    for x in range(1, sqrt_N + 1):
        x2 = x * x
        y2 = N - x2
        if y2 < x2:
            break
        y = math.isqrt(y2)
        if y * y == y2:
            pairs.append([x2, y2])
            
    return {'N': N, 'pairs': pairs}
