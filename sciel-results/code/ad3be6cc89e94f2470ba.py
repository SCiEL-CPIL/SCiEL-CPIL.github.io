import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 80  # Safety margin for 90s limit
    
    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000
    
    # 1. Sieve of Eratosthenes to get primes
    is_prime = bytearray(b'\x01') * (MAXN + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, int(MAXN**0.5) + 1) if is_prime[i]]
    
    # 2. Compute powerful part P(m) for all m using NumPy
    # P(m) = product of p^k where k = v_p(m) and k >= 2
    pp = np.ones(MAXN + 1, dtype=np.int64)
    
    # Only primes p where p^2 <= MAXN can contribute to the powerful part
    # For p > sqrt(MAXN), v_p(n) <= 1 for all n <= MAXN, so contribution is 1.
    for p in primes:
        if elapsed() > DEADLINE:
            break
            
        # Powers of p: p^2, p^3, ...
        p_power = p * p
        
        while p_power <= MAXN:
            # Multiply by p for all multiples of p_power
            # This accumulates p^(v_p(n)-1) for v_p(n) >= 2
            pp[p_power : MAXN + 1 : p_power] *= p
            
            p_power *= p
            
        # Multiply by p for all multiples of p^2
        # This adds the final factor to make it p^v_p(n)
        pp[p*p : MAXN + 1 : p*p] *= p

    # 3. Find max pp[n] * pp[n+1]
    # Vectorized scan
    # n ranges from 10000 to MAXN-1
    prod = pp[10000:MAXN] * pp[10001:MAXN+1]
    
    if len(prod) == 0:
        return 10000
        
    max_idx = np.argmax(prod)
    best_n = 10000 + max_idx
    
    return best_n
