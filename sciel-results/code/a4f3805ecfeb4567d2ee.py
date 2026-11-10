import time

def construct():
    N = 123457
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    def get_smooth_numbers(primes, limit):
        """Generate all B-smooth numbers up to limit."""
        smooths = set()
        smooths.add(1)
        current = [1]
        for p in primes:
            new_vals = []
            for x in current:
                val = x
                while val * p <= limit:
                    val *= p
                    new_vals.append(val)
                    smooths.add(val)
            current.extend(new_vals)
        return smooths
    
    # Try increasing smoothness levels
    # Primes in order
    all_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for num_primes in range(2, len(all_primes) + 1):
        if time.time() > deadline:
            break
        primes_subset = all_primes[:num_primes]
        max_prime = primes_subset[-1]
        smooths = get_smooth_numbers(primes_subset, N)
        
        # Check if any pair sums to N
        found = False
        for a in smooths:
            b = N - a
            if 1 <= b <= N and b in smooths:
                return [a, b]
        
        # If we found smooth numbers but no pair, continue to next level
    
    # Fallback: Sieve for largest prime factor and find optimal pair
    if time.time() > deadline - 10:
        # Emergency: return a reasonable pair
        return [1, N - 1]
    
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    # Find the pair minimizing max(P(a), P(b))
    best_max = N + 1
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N // 2 + 1):
        if time.time() > deadline - 5:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max:
            best_max = m
            best_a = a
            best_b = b
    
    return [best_a, best_b]
