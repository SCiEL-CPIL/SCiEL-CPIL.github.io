import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    N = 10_000_000
    
    # Step 1: Compute smallest prime factor (SPF) sieve
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Compute sigma(n) for all n using SPF
    # sigma is multiplicative. For n = p1^a1 * p2^a2 * ...:
    # sigma(n) = prod (pi^(ai+1) - 1) / (pi - 1)
    # We compute incrementally:
    # Let ppow[n] = highest power of spf[n] dividing n
    # Let sig_ppow[n] = sigma(ppow[n])
    # Then sigma[n] = sigma[n // ppow[n]] * sig_ppow[n]
    
    ppow = [1] * (N + 1)
    sig_ppow = [1] * (N + 1)
    sigma = [0] * (N + 1)
    sigma[1] = 1
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # n = p^(k+1) * r, m = p^k * r
            ppow[n] = ppow[m] * p
            sig_ppow[n] = sig_ppow[m] + ppow[m]
            r = n // ppow[n]
            sigma[n] = sigma[r] * sig_ppow[n]
        else:
            # n = p * m, gcd(p, m) = 1
            ppow[n] = p
            sig_ppow[n] = 1 + p
            sigma[n] = sigma[m] * sig_ppow[n]
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return [11, 6]
    
    # Step 3: Search by k = |2a - 3b|
    # Error = k / (2b). We want small k and large b.
    # For each k, a = (3b + k) / 2 (assuming a > b, 2a - 3b = k)
    # Also consider 3b - 2a = k (b > a case), but a/b close to 1.5 means a > b
    
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5)  # = 1/3
    
    max_k_to_try = 5000
    
    for k in range(1, max_k_to_try + 1):
        if time.time() > deadline - 2:
            break
        
        # Pruning: minimum possible error for this k is k / (2 * N)
        # If this is >= best_err, no need to continue
        if k / (2.0 * N) >= best_err:
            break
        
        # a = (3b + k) / 2, so 3b + k must be even
        # Parity: if k is even, b must be even; if k is odd, b must be odd
        # a <= N => 3b + k <= 2N => b <= (2N - k) / 3
        # a >= 1 => 3b + k >= 2 => b >= (2 - k) / 3, so b >= 1 for k <= 1
        
        max_b = (2 * N - k) // 3
        if max_b < 1:
            continue
        
        # Adjust to correct parity
        start_b = max_b
        if (3 * start_b + k) % 2 != 0:
            start_b -= 1
        
        # Search b from max_b downward with step 2
        # We want to find the largest b where sigma(a) == sigma(b)
        # Limit search depth to avoid spending too much time on one k
        max_depth = 50000  # Check at most 50000 values of b for each k
        
        checked = 0
        for b_val in range(start_b, max(0, start_b - 2 * max_depth), -2):
            if time.time() > deadline - 1:
                break
            
            a_val = (3 * b_val + k) // 2
            if a_val > N or a_val < 1:
                continue
            
            if sigma[a_val] == sigma[b_val]:
                err = k / (2.0 * b_val)
                if err < best_err:
                    best_err = err
                    best_a = a_val
                    best_b = b_val
                    # Found a solution for this k with large b
                    # Since k is small and b is large, this is likely very good
                    # We can break to try next k (smaller k is better)
                    break
        
        # If we found a solution for k=1, it's extremely likely to be near-optimal
        # because error ~ 1/(2b) and b is large
        if k == 1 and best_err < 1.0 / (2 * N * 0.9):
            # Found a very good solution for k=1
            break
    
    return [int(best_a), int(best_b)]
