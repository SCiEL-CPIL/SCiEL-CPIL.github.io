import time
import math
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Set N based on available time or fixed value
    # Let's try N = 5_000_000 first, if fast, we can't easily increase in same run.
    # We'll fix N = 5_000_000 to be safe on memory and time.
    N = 5_000_000
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = array('I', range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute sigma(k) for all k
    # sigma[k] = sum of divisors of k
    sigma = array('Q', [0]) * (N + 1)
    sigma[0] = 0
    sigma[1] = 1
    
    # Using the property: sigma is multiplicative.
    # We can compute sigma for all k using the sieve.
    # Alternatively, use the formula: sigma(n) = product over p^a || n of (p^(a+1)-1)/(p-1).
    # This requires factorization. With SPF, factorization is fast.
    
    # Let's compute sigma using a dynamic approach or direct factorization.
    # Direct factorization for each k is O(log k) on average.
    # 5e6 * log(5e6) ~ 5e6 * 20 = 1e8 operations. Might be slow in Python.
    
    # Better: Use the sieve to compute sigma directly.
    # sigma[1] = 1
    # For i from 2 to N:
    #   if spf[i] == i (prime):
    #      sigma[i] = 1 + i
    #   else:
    #      p = spf[i]
    #      m = i // p
    #      if m % p == 0:
    #         # p divides m, so p^a || i, p^(a-1) || m
    #         # Let p^a || i, then i = p^a * r, gcd(r,p)=1
    #         # sigma(i) = sigma(p^a) * sigma(r)
    #         # sigma(m) = sigma(p^(a-1)) * sigma(r)
    #         # sigma(p^a) = sigma(p^(a-1)) * p
    #         # So sigma(i) = sigma(m) * p
    #         sigma[i] = sigma[m] * p
    #      else:
    #         # p does not divide m, so p || i
    #         # sigma(i) = sigma(p) * sigma(m) = (1+p) * sigma(m)
    #         sigma[i] = sigma[m] * (1 + p)

    for i in range(2, N + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            sigma[i] = sigma[m] * p
        else:
            sigma[i] = sigma[m] * (1 + p)

    # 3. Group k by M = k * sigma(k)
    solutions = {}
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            if M not in solutions:
                solutions[M] = []
            solutions[M].append(k)
        
        if time.time() > deadline:
            break

    # 4. Find best M
    best_M = 0
    best_k_list = []
    for M, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_M = M
            best_k_list = k_list
    
    # 5. Verify and return
    # The task requires distinct k_i. The list construction ensures this.
    # Check k_i <= isqrt(M)
    # As derived, k <= sigma(k) => k^2 <= k sigma(k) = M => k <= sqrt(M).
    # So all k in the list are valid.
    
    if not best_k_list:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_k_list}
