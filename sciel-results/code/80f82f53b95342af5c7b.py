import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    N = 1000000
    LOW = 10000
    
    # Initialize powerful part array
    # We only need to store values for indices in [LOW, N], but 
    # computing for all 1..N is simpler and fast enough.
    # Using int64 to avoid overflow if possible, though Python ints are arbitrary precision.
    # Numpy int64 is faster. Max powerful part for N=1e6 is at most N itself.
    # Product of two such parts can be up to 1e12, which fits in int64.
    pow_part = np.ones(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to find primes up to sqrt(N) is not enough, 
    # we need primes up to N/2? No, if p > sqrt(N), p^2 > N, so p cannot be a factor 
    # with exponent >= 2. So we only need primes up to sqrt(N).
    limit = int(np.sqrt(N)) + 1
    
    # Generate primes up to limit
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # For each prime, update the powerful part of its multiples
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
            
        # We need to update all multiples of p2 in [1, N]
        # Start index: first multiple of p2 >= 1 is p2 itself.
        # But we only care about [LOW, N] for the final score, 
        # however, pow_part is used for n and n+1. 
        # n can be as low as 10000. n+1 can be 10001.
        # So we need pow_part for all i in [10000, 1000000].
        # It is safer to compute for all 1..N to avoid boundary issues, 
        # or just compute for LOW..N.
        # Let's compute for LOW..N to save some time, though negligible.
        
        # Find first multiple of p2 >= LOW
        start = LOW
        if start % p2 != 0:
            start += p2 - (start % p2)
            
        # Iterate through multiples of p2
        # For each multiple m, find the highest power p^e dividing m
        # and multiply pow_part[m] by p^e
        
        # Optimization: 
        # The exponent e is at least 2.
        # We can precompute powers of p: p^2, p^3, ...
        
        m = start
        while m <= N:
            # Determine exponent e
            # m is divisible by p2
            e = 2
            power = p2
            next_power = power * p
            # Check if m is divisible by higher powers
            # We can just divide m by p repeatedly? 
            # No, m is fixed. We check m % next_power == 0.
            while next_power <= N and m % next_power == 0:
                e += 1
                power = next_power
                next_power *= p
                
            # Multiply the powerful part by p^e
            pow_part[m] *= power
            
            # Move to next multiple of p2
            m += p2
            
        # Check deadline
        if time.time() > deadline:
            break
            
    # Find n in [LOW, N-1] that maximizes pow_part[n] * pow_part[n+1]
    # Note: n+1 can be up to N.
    # We need to evaluate for n from LOW to N-1.
    
    best_n = LOW
    best_score = 0
    
    # Vectorized approach for scoring
    # scores = pow_part[LOW:N] * pow_part[LOW+1:N+1]
    # This creates a large array, but N=1e6 is fine.
    # However, constructing the slice might be slow if done repeatedly. 
    # Doing it once is fine.
    
    # Ensure we don't go out of bounds
    # n ranges from LOW to N-1
    # pow_part[n] is valid for n in [1, N]
    # pow_part[n+1] is valid for n+1 in [1, N] => n in [0, N-1]
    # So n in [LOW, N-1] is safe.
    
    scores = pow_part[LOW:N] * pow_part[LOW+1:N+1]
    
    max_idx = np.argmax(scores)
    best_n = LOW + max_idx
    best_score = scores[max_idx]
    
    return best_n

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    N = 1000000
    LOW = 10000
    
    # Initialize powerful part array
    pow_part = np.ones(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to find primes up to sqrt(N)
    limit = int(np.sqrt(N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # For each prime, update the powerful part of its multiples
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
            
        # Find first multiple of p2 >= LOW
        start = LOW
        if start % p2 != 0:
            start += p2 - (start % p2)
            
        m = start
        while m <= N:
            # Determine exponent e
            e = 2
            power = p2
            next_power = power * p
            while next_power <= N and m % next_power == 0:
                e += 1
                power = next_power
                next_power *= p
                
            # Multiply the powerful part by p^e
            pow_part[m] *= power
            
            m += p2
            
        if time.time() > deadline:
            break
            
    # Find n in [LOW, N-1] that maximizes pow_part[n] * pow_part[n+1]
    scores = pow_part[LOW:N] * pow_part[LOW+1:N+1]
    max_idx = np.argmax(scores)
    best_n = LOW + max_idx
    
    return best_n
# EVOLVE-BLOCK-END
