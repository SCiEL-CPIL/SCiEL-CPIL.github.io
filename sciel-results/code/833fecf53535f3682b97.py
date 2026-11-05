import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds safety margin
    
    N = 1000000
    MIN_N = 10000
    
    # 1. Sieve for primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.where(is_prime)[0]
    
    if time.time() > deadline:
        return MIN_N
        
    # 2. Compute powerful part
    # powerful[n] is the product of p^e where e >= 2
    powerful = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        power = p * p
        while power <= N:
            max_j = N // power
            # j ranges from 1 to max_j
            # We want j such that j % p != 0
            # This ensures v_p(j * power) = v_p(power) + v_p(j) = k + 0 = k
            # If j % p == 0, then v_p(j * power) > k, so this p^k is not the FULL power.
            # Wait.
            # Definition: "product of FULL prime powers p^e with exponent e>=2"
            # If n = 12 = 2^2 * 3.
            # p=2, e=2. Full power is 4.
            # If n = 8 = 2^3.
            # p=2, e=3. Full power is 8.
            
            # My loop:
            # p=2.
            # k=2, power=4.
            # j=1 (1%2!=0) -> index 4. powerful[4] *= 4.
            # j=2 (2%2==0) -> skip. (index 8).
            # j=3 (3%2!=0) -> index 12. powerful[12] *= 4.
            # j=4 (4%2==0) -> skip. (index 16).
            
            # k=3, power=8.
            # j=1 (1%2!=0) -> index 8. powerful[8] *= 8.
            # j=2 (2%2==0) -> skip. (index 16).
            
            # So for n=8:
            # k=2 loop: j=1 (index 4), j=3 (index 12)... index 8 is skipped because j=2 is even.
            # k=3 loop: j=1 (index 8). powerful[8] *= 8.
            # Result: powerful[8] = 8. Correct.
            
            # For n=12:
            # k=2 loop: j=3 (index 12). powerful[12] *= 4.
            # k=3 loop: max_j = 12//8 = 1. j=1 (index 8).
            # Result: powerful[12] = 4. Correct.
            
            # For n=16:
            # k=2: j=4 (even) skip.
            # k=3: j=2 (even) skip.
            # k=4: power=16. max_j=1. j=1 (odd). powerful[16] *= 16.
            # Result: powerful[16] = 16. Correct.
            
            js = np.arange(1, max_j + 1, dtype=np.int64)
            mask = js % p != 0
            if np.any(mask):
                indices = js[mask] * power
                powerful[indices] *= power
                
            power *= p
            if power > N:
                break
                
    if time.time() > deadline:
        return MIN_N

    # 3. Vectorized scan for max product
    p1 = powerful[MIN_N:N]
    p2 = powerful[MIN_N+1:N+1]
    
    products = p1 * p2
    best_idx = int(np.argmax(products))
    best_n = MIN_N + best_idx
    
    return best_n
