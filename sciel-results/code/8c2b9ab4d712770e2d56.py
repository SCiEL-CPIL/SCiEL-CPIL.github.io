import numpy as np
import time

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    
    # We need to evaluate n up to 1,000,000.
    # So we need P(n) and P(n+1) for n=1,000,000.
    # Thus we need P up to 1,000,001.
    MAXN = 1_000_001
    
    # 1. Sieve of Eratosthenes for primes up to MAXN
    # Using a bytearray for memory efficiency, though bool is fine for 1e6
    is_prime = np.ones(MAXN + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.where(is_prime)[0]
    
    # 2. Compute h(x)
    # h[x] = product of p where v_p(x) == 1
    # P[x] = x / h[x]
    
    # Initialize h to 1 for all x
    h = np.ones(MAXN + 1, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        # Multiply by p for all multiples of p
        # This adds p to the "squarefree candidate" set
        h[p::p] *= p
        
        # Divide by p for all multiples of p^2
        # This removes p from the "squarefree candidate" set if p^2 divides x
        p2 = p * p
        if p2 <= MAXN:
            h[p2::p2] //= p
            
        # Time check
        if time.time() - t0 > DEADLINE:
            return 10000
            
    # 3. Compute P[x]
    # P[x] = x // h[x]
    # Note: x is 0..MAXN
    # h[0] is 1, 0//1 = 0. P[0]=0.
    # We only care about x >= 10000.
    
    x_vals = np.arange(MAXN + 1, dtype=np.int64)
    P = x_vals // h
    
    # 4. Maximize P[n] * P[n+1] for n in [10000, 1000000]
    # n ranges from 10000 to 1000000
    # So we look at P[10000] * P[10001] ... P[1000000] * P[1000001]
    
    start = 10000
    end = 1000000
    
    # Slice P from start to end+1 (inclusive)
    # P_slice contains P[start] ... P[end+1]
    # Length is (end+1) - start + 1 = end - start + 2
    P_slice = P[start:end+2]
    
    # products[i] = P_slice[i] * P_slice[i+1]
    # This corresponds to n = start + i
    # We want i such that start+i is in [start, end]
    # So i ranges from 0 to end - start
    
    # P_slice[:-1] are P[start] ... P[end]
    # P_slice[1:] are P[start+1] ... P[end+1]
    products = P_slice[:-1] * P_slice[1:]
    
    # Find index of max
    idx = np.argmax(products)
    best_n = int(start + idx)
    
    # Ensure best_n is in valid range
    if best_n < 10000 or best_n > 1000000:
        best_n = 10000
        
    return best_n
