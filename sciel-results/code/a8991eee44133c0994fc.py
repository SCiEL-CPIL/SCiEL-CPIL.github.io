import sys
import time

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Primes up to 1000 (for smooth number generation)
    primes = []
    for x in range(2, 1000):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    
    # Generate y-smooth numbers in [N_MIN, N_MAX] for given max prime index
    def generate_smooth(max_prime_idx, lo, hi):
        """Generate all numbers in [lo, hi] whose prime factors are <= primes[max_prime_idx]"""
        result = set()
        # Use DFS to generate smooth numbers
        def dfs(idx, current):
            if current > hi:
                return
            if current >= lo:
                result.add(current)
            if idx >= max_prime_idx:
                return
            p = primes[idx]
            # Multiply by p^k for k >= 1
            val = current
            while val <= hi:
                val *= p
                if val > hi:
                    break
                dfs(idx, val)
                dfs(idx + 1, val)
        
        dfs(0, 1)
        return result
    
    # Start with small y and increase
    best_n = N_MIN
    best_val = float('inf')
    
    # First, compute LPF for all numbers using sieve to get initial best
    spf = list(range(N_MAX + 2))
    for i in range(2, int(N_MAX**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N_MAX + 2, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def get_lpf(x):
        if x <= 1:
            return 1
        max_p = 1
        while x > 1:
            p = spf[x]
            if p > max_p:
                max_p = p
            while x % p == 0:
                x //= p
        return max_p
    
    # Get initial best from full scan
    for n in range(N_MIN, N_MAX + 1):
        val = max(get_lpf(n), get_lpf(n + 1))
        if val < best_val:
            best_val = val
            best_n = n
    
    # Now try to find better solutions using smooth number generation
    # For each y (max prime allowed), generate y-smooth numbers and check consecutive pairs
    for max_prime_idx in range(len(primes)):
        if time.time() > deadline:
            break
        y = primes[max_prime_idx]
        if y >= best_val:
            break  # No improvement possible
        
        smooth = generate_smooth(max_prime_idx, N_MIN, N_MAX + 1)
        if not smooth:
            continue
        
        # Check consecutive pairs
        for m in sorted(smooth):
            if m + 1 in smooth and N_MIN <= m <= N_MAX:
                # Both m and m+1 are y-smooth, so max LPF <= y
                if y < best_val:
                    best_val = y
                    best_n = m
                    break  # Found optimal for this y
    
    return best_n
