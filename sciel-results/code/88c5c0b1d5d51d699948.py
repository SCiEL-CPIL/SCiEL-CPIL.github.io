import time
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 60.0  # seconds
    max_time_allowed = deadline - 5.0  # Safety margin

    N = 1000000
    LOWER = 10000
    UPPER = N
    
    # Step 1: Compute Smallest Prime Factor (SPF) for all numbers up to N
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Compute powerful part for each number
    # pow_part[i] is the product of p^e where e>=2 in factorization of i
    pow_part = [1] * (N + 2)
    
    # We compute pow_part for all i from 1 to N
    # To be efficient, we can process sequentially
    for i in range(2, N + 1):
        p = spf[i]
        m = i
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        
        # m is now i with all factors of p removed
        if e >= 2:
            pow_part[i] = pow_part[m] * (p ** e)
        else:
            pow_part[i] = pow_part[m]

    # Check time, if exceeded, return a default (though unlikely)
    if time.time() - start_time > max_time_allowed:
        return 10000

    # Step 3: Find n in [10000, 1000000] that maximizes pow_part[n] * pow_part[n+1]
    # Use NumPy for faster vectorized calculation
    pp = np.array(pow_part, dtype=np.int64)
    
    # We need to maximize pp[n] * pp[n+1] for n in [LOWER, N-1]
    # pp indices: 0 to N
    # n ranges from LOWER to N-1
    # So we look at pp[LOWER : N] and pp[LOWER+1 : N+1]
    
    # Ensure we don't go out of bounds
    # pp has length N+2, so pp[N+1] is valid (though 0/1)
    # n+1 goes up to N.
    # So we need pp[LOWER:N] * pp[LOWER+1:N+1]
    
    left = pp[LOWER:N]
    right = pp[LOWER+1:N+1]
    products = left * right
    
    # Find index of max product
    idx = np.argmax(products)
    best_n = LOWER + idx
    
    return int(best_n)
