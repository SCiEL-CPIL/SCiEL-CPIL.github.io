import time
from math import isqrt

def construct():
    N = 123457
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # We want to find min K such that there exist a,b with a+b=N, 
    # both having all prime factors <= K.
    # Generate K-smooth numbers for increasing K.
    
    # First, get primes up to N (we'll need them for smooth number generation)
    limit = N
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each K (prime), generate all K-smooth numbers up to N
    # A K-smooth number has all prime factors <= K
    # We use the list of primes <= K
    
    best_answer = (1, N - 1)  # fallback
    best_K = N  # worst case
    
    # Try K = 2, 3, 5, 7, 11, ... (primes in order)
    for idx, K in enumerate(primes):
        if time.time() - start_time > deadline:
            break
        
        # Generate all K-smooth numbers up to N
        # Use the primes up to K
        smooth_primes = primes[:idx + 1]  # primes[0..idx] are all <= K
        
        # Generate all numbers of form p1^a1 * p2^a2 * ... <= N
        smooth_nums = set()
        
        def gen_smooth(idx_p, current):
            if current > N:
                return
            smooth_nums.add(current)
            if idx_p >= len(smooth_primes):
                return
            p = smooth_primes[idx_p]
            # Multiply by p repeatedly
            val = current
            while val * p <= N:
                val *= p
                gen_smooth(idx_p, val)
                if time.time() - start_time > deadline:
                    return
            gen_smooth(idx_p + 1, current)
        
        gen_smooth(0, 1)
        
        # Now check if any two smooth numbers sum to N
        found = False
        for a in smooth_nums:
            b = N - a
            if b in smooth_nums and a >= 1 and b >= 1:
                found = True
                break
        
        if found:
            best_answer = (a, b)
            best_K = K
            break
    
    # Verify
    a, b = best_answer
    assert a > 0 and b > 0 and a + b == N, f"Invalid: {a}+{b}={a+b}"
    
    return (a, b)
