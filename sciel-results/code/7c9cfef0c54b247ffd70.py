import time
import math

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N_MAX_VAL = 1000000
    N_MIN_VAL = 10000
    
    # 1. Get primes up to sqrt(10^6) = 1000
    primes = _get_primes(1000)
    
    # 2. Initialize powerful part array
    # pp[x] will store the powerful part of x
    pp = [1] * (N_MAX_VAL + 1)
    
    # 3. Sieve to compute powerful parts
    for p in primes:
        p_k = p * p
        k = 2
        while p_k <= N_MAX_VAL:
            next_p_k = p_k * p
            
            # Precompute p^k
            p_pow_k = p ** k
            
            # Iterate multiples of p_k
            # m is a multiple of p_k
            # If m is not a multiple of next_p_k (p^(k+1)), then v_p(m) = k
            # In that case, the contribution of p to P(m) is p^k
            # We multiply pp[m] by p^k
            
            # Optimization: 
            # Multiples of p_k are p_k, 2*p_k, 3*p_k, ...
            # We need to skip multiples of next_p_k.
            # next_p_k is a multiple of p_k.
            # Specifically, next_p_k = p * p_k.
            # So every p-th multiple of p_k is a multiple of next_p_k.
            # i.e., m = i * p_k. If i is a multiple of p, then m is a multiple of p^(k+1).
            
            i = 1
            m = p_k
            while m <= N_MAX_VAL:
                if i % p != 0:
                    pp[m] *= p_pow_k
                i += 1
                m += p_k
            
            p_k = next_p_k
            k += 1
            
            if time.time() > deadline - 5:
                break
    
    if time.time() > deadline - 5:
        # Fallback if sieve took too long (unlikely)
        return N_MIN_VAL

    # 4. Find the best n
    best_n = N_MIN_VAL
    best_score = 0
    
    # Iterate through valid n
    # n in [10000, 999999]
    # We can use numpy for speed if available, but list iteration is fast enough for 10^6
    # Let's try a simple loop first.
    for n in range(N_MIN_VAL, N_MAX_VAL):
        score = pp[n] * pp[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
        if time.time() > deadline - 2:
            break

    return best_n
