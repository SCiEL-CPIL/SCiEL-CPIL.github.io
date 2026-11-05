import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    small_prime_set = set(small_primes)
    
    # Generate primes up to sqrt(MAX_VAL) ~ 1000
    # Actually, we need primes up to MAX_VAL to factorize? No, just to remove large prime factors.
    # We need to identify which primes are NOT in small_primes.
    # We can just iterate through all numbers and check if they are prime and not in small_primes.
    # Or generate primes up to MAX_VAL.
    # Sieve of Eratosthenes up to MAX_VAL is $O(N \log \log N) \approx 10^6 \times 3 = 3 \times 10^6$. Fast.
    
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_VAL + 1, i):
                is_prime[j] = False
                
    # Initialize L[m] = 1
    L = [1] * (MAX_VAL + 1)
    
    # For each prime q not in small_primes, update L[m] for multiples of q
    # We only need to process primes q >= 29
    for q in range(29, MAX_VAL + 1):
        if is_prime[q]:
            # Update L for multiples of q
            for m in range(q, MAX_VAL + 1, q):
                temp = m
                factor = 1
                while temp % q == 0:
                    factor *= q
                    temp //= q
                L[m] *= factor
                
    # Compute A[m] = m / L[m]
    A = [0] * (MAX_VAL + 1)
    for m in range(1, MAX_VAL + 1):
        A[m] = m // L[m]
        
    # Map distinct A values to IDs
    unique_A = {}
    A_ids = [0] * (MAX_VAL + 1)
    next_id = 0
    for m in range(1, MAX_VAL + 1):
        val = A[m]
        if val not in unique_A:
            unique_A[val] = next_id
            next_id += 1
        A_ids[m] = unique_A[val]
        
    # Search for n
    best_n = N_MIN
    best_count = WINDOW + 1
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
            
        # Count distinct A_ids in window
        # Using a set is simple and likely fast enough for 24 elements
        window_ids = A_ids[n+1 : n+WINDOW+1]
        count = len(set(window_ids))
        
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break
                
    return best_n
