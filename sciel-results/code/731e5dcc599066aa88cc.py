import math
import time

def construct():
    start_time = time.time()
    LIMIT = 1000000
    MAX_N = LIMIT + 1  # We need n+1 up to 1000001
    
    # 1. Sieve of Eratosthenes to find primes up to MAX_N
    is_prime = [True] * (MAX_N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_N + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Compute powerful part for each number
    # pp[n] will store the powerful part of n
    pp = [1] * (MAX_N + 1)
    
    for p in primes:
        # Calculate powers of p
        # p^2, p^3, ...
        # We need to handle each exponent k >= 2
        # For a number m, if v_p(m) = k, we multiply pp[m] by p^k
        
        # Efficient way:
        # Iterate k from 2 upwards
        # For each k, let p_k = p^k
        # For each multiple m of p_k:
        #    Check if m is divisible by p^(k+1)
        #    If not, then v_p(m) == k, so multiply pp[m] by p_k
        #    If yes, do nothing (will be handled by higher k)
        
        p_k = p * p  # p^2
        k = 2
        while p_k <= MAX_N:
            p_next = p_k * p  # p^(k+1)
            
            # Iterate multiples of p_k
            # m = p_k, 2*p_k, 3*p_k, ...
            for m in range(p_k, MAX_N + 1, p_k):
                # Check if m is divisible by p_next
                # If m % p_next != 0, then the exponent of p in m is exactly k
                if m % p_next != 0:
                    pp[m] *= p_k
                # Else, the exponent is > k, so we skip
                # Note: If p_next > MAX_N, then for all m <= MAX_N, 
                # m cannot be divisible by p_next if m is a multiple of p_k 
                # and m < p_next? 
                # Actually, if p_next > MAX_N, then no m <= MAX_N is divisible by p_next.
                # So the condition m % p_next != 0 is always true.
                # We can optimize this.
                
            p_k = p_next
            k += 1
            
            # Time check
            if time.time() - start_time > 50:
                break

    if time.time() - start_time > 50:
        # Fallback if sieve is too slow (unlikely)
        return 10000

    # 3. Find the n that maximizes pp[n] * pp[n+1]
    best_n = 10000
    max_score = 0
    
    for n in range(10000, LIMIT + 1):
        score = pp[n] * pp[n+1]
        if score > max_score:
            max_score = score
            best_n = n
            
    return best_n
