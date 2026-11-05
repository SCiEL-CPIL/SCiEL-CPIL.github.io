import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 1000000
    START = 10000
    
    # Sieve for Smallest Prime Factor (SPF) using numpy for speed
    spf = np.arange(N + 1, dtype=np.int64)
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            j = i * i
            spf[j:N+1:i] = np.where(spf[j:N+1:i] == spf[j:N+1:i], spf[j:N+1:i], spf[i])
            # Actually, let me use a simpler approach - mark only if not yet marked
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number using SPF
    P = np.ones(N + 1, dtype=np.int64)
    
    for k in range(2, N + 1):
        x = k
        p_part = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                p_part *= (p ** e)
        P[k] = p_part
    
    # Find n in [10000, N-1] maximizing P[n] * P[n+1]
    # Use numpy for the product and argmax
    products = P[START:N] * P[START+1:N+1]
    best_idx = np.argmax(products)
    best_n = START + best_idx
    best_val = products[best_idx]
    
    if time.time() > deadline:
        pass  # Already have best_n
    
    return int(best_n)
