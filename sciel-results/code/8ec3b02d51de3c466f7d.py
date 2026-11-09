def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Generate primes up to 10000 using simple sieve
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(limit + 1) if is_prime[i]]
    
    primes = sieve_primes(10000)
    
    def is_smooth(n, primes):
        """Check if n is smooth with respect to given primes (all prime factors in primes)."""
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                n //= p
        return n == 1
    
    def generate_smooth_numbers(lo, hi, primes):
        """Generate all P-smooth numbers in [lo, hi] where P is the largest prime in primes."""
        smooth = set()
        
        def gen(current, prime_idx):
            if current > hi:
                return
            if current >= lo:
                smooth.add(current)
            for i in range(prime_idx, len(primes)):
                p = primes[i]
                if current * p > hi:
                    break
                gen(current * p, i)
        
        gen(1, 0)
        return sorted(smooth)
    
    best_n = N_MIN
    best_val = float('inf')
    
    # Try progressively larger sets of small primes
    # Start with smallest primes and add more as needed
    for num_primes in range(1, len(primes) + 1):
        if time.time() > deadline:
            break
        
        small_primes = primes[:num_primes]
        max_p = small_primes[-1]
        
        # Generate all smooth numbers in [N_MIN, N_MAX+1]
        smooth_nums = generate_smooth_numbers(N_MIN, N_MAX + 1, small_primes)
        
        # Check for consecutive pairs
        found = False
        for i in range(len(smooth_nums) - 1):
            a = smooth_nums[i]
            b = smooth_nums[i + 1]
            if b - a == 1 and a >= N_MIN and a <= N_MAX:
                # Both a and a+1 are P-smooth, so LPF of product is at most max_p
                if max_p < best_val:
                    best_val = max_p
                    best_n = a
                found = True
                break
        
        if found:
            # We found a pair at this smoothness level
            # But we should check if a smaller n exists at the same level
            # Actually, we want to minimize the LPF value, so smaller max_p is better
            # But within the same max_p, any valid n works
            # Let's continue to see if we can find an even smaller max_p? No, we're increasing num_primes
            # So the first time we find a pair gives us the smallest max_p
            # But wait - we might find a pair at num_primes=5 (max_p=11) and also at num_primes=3 (max_p=5)
            # Since we're iterating from small to large num_primes, the first found is optimal
            break
    
    # If no smooth pair found (unlikely), fall back to linear scan with precomputed LPF
    if best_val == float('inf'):
        # Fallback: compute LPF via sieve for all numbers
        N = N_MAX + 1
        lpf = [0] * (N + 1)
        for i in range(2, N + 1):
            if lpf[i] == 0:
                for j in range(i, N + 1, i):
                    lpf[j] = i
        
        best_val = float('inf')
        best_n = N_MIN
        for n in range(N_MIN, N_MAX + 1):
            if time.time() > deadline:
                break
            v = max(lpf[n], lpf[n + 1])
            if v < best_val:
                best_val = v
                best_n = n
    
    return best_n
