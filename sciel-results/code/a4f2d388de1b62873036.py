import time
import numpy as np

def construct():
    N = 10**7
    start_time = time.time()
    deadline = 80  # safety margin before 90s kill

    # Build smallest prime factor sieve and sigma array
    spf = np.zeros(N + 1, dtype=np.int64)
    spf[1] = 1
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    # Compute sigma(n) for all n <= N using the recurrence:
    # if n = p^k * m with p not dividing m:
    #   sigma(n) = sigma(m) * (p^(k+1)-1)/(p-1)
    sigma = np.ones(N + 1, dtype=np.int64)
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # n has higher power of p than m
            # sigma(n) = sigma(m) + p^k * sigma(m/p^k) ... 
            # Actually use: sigma(n) = sigma(m) * p + sigma(m / p^a) where m = p^a * rest
            # Simpler: sigma(n) = sigma(m) + p * (sigma(m) - sigma(m//p)) if m%p==0? No.
            # Correct recurrence: sigma(n) = sigma(m) + p * sigma(m) - p * sigma(m//p) ... 
            # Let me use the standard: if n = p * m, p | m:
            #   sigma(n) = sigma(m) + p * sigma(m) - p * sigma(m//p)
            # Actually the cleanest is:
            #   Let n = p^k * r, p ∤ r. sigma(n) = (1+p+...+p^k) * sigma(r)
            #   sigma(m) where m = n/p = p^(k-1) * r = (1+...+p^(k-1)) * sigma(r)
            #   So sigma(n) = sigma(m) + p^k * sigma(r) = sigma(m) + p * p^(k-1) * sigma(r)
            #   And p^(k-1)*sigma(r) = sigma(m) - sigma(r) ... not quite.
            #   sigma(m) = (1+p+...+p^(k-1)) * sigma(r)
            #   sigma(n) = (1+p+...+p^k) * sigma(r) = sigma(m) + p^k * sigma(r)
            #   p^k * sigma(r) = p * p^(k-1) * sigma(r)
            #   From sigma(m) = (1+...+p^(k-1))*sigma(r), we have sigma(r) = sigma(m) / (1+...+p^(k-1))
            # This is getting complex. Let me just use the sieve approach with numpy.
            pass
        else:
            # p does not divide m, so n = p * m with gcd(p,m)=1
            # sigma(n) = (1+p) * sigma(m)
            sigma[n] = (1 + p) * sigma[m]

    # The above is incomplete for the p|m case. Let me redo with a proper sieve.
    # Use the classic sieve: sigma[n] starts at 1, for each prime p, multiply contributions.
    # Actually, let me just use the divisor-sum sieve directly with numpy.
    
    # Recompute sigma properly
    sigma = np.ones(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                # multiply sigma[j] by (1 + i + i^2 + ... + i^k) where i^k || j
                k = 0
                temp = j
                while temp % i == 0:
                    temp //= i
                    k += 1
                # factor = 1 + i + ... + i^k = (i^(k+1)-1)/(i-1)
                factor = 0
                ipow = 1
                for _ in range(k + 1):
                    factor += ipow
                    ipow *= i
                sigma[j] *= factor

    # This is O(N log log N) but with Python loops over primes... might be slow.
    # Let me use a different approach: the standard sigma sieve.
    
    # Actually the fastest approach: sigma[n] = sum of divisors. Use the sieve:
    # Initialize sigma = ones, then for each d from 1 to N, add d to sigma[d], sigma[2d], ...
    # But that's O(N log N) which is too slow in Python.
    
    # Let me use numpy vectorized approach for the sigma computation.
    sigma = np.ones(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d

    # Now search for pairs with sigma[a] == sigma[b] minimizing |a/b - 3/2|
    # Error = |2a - 3b| / (2b)
    # For small delta = |2a - 3b|, scan b from large to small.
    
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5)  # 1/3

    for delta in range(1, 50000):
        # a = (3b + delta) / 2 or a = (3b - delta) / 2
        # Case 1: 2a = 3b + delta => a = (3b + delta)//2, need 3b + delta even
        # Case 2: 2a = 3b - delta => a = (3b - delta)//2, need 3b - delta even, a >= 1
        
        # Case 1: 3b + delta even
        # If delta even: b even. If delta odd: b odd.
        b_max_1 = min(N, (2 * N - delta) // 3)
        if b_max_1 < 1:
            continue
        min_err_1 = delta / (2.0 * b_max_1)
        if min_err_1 >= best_err:
            # Try case 2 too
            pass
        else:
            if delta % 2 == 0:
                b_start = b_max_1 if b_max_1 % 2 == 0 else b_max_1 - 1
            else:
                b_start = b_max_1 if b_max_1 % 2 == 1 else b_max_1 - 1
            if b_start >= 1:
                for b in range(b_start, 0, -2):
                    curr_err = delta / (2.0 * b)
                    if curr_err >= best_err:
                        break
                    a = (3 * b + delta) // 2
                    if a > N or a < 1:
                        continue
                    if sigma[a] == sigma[b]:
                        best_err = curr_err
                        best_a = a
                        best_b = b
                        break  # found for this delta, move on
        
        # Case 2: 2a = 3b - delta => a = (3b - delta)//2, need 3b - delta even, a >= 1
        # 3b - delta >= 2 => b >= (delta + 2)//3
        # 3b - delta even: same parity condition as case 1
        b_max_2 = min(N, (2 * N + delta) // 3)  # a <= N => (3b-delta)/2 <= N => b <= (2N+delta)/3
        b_min_2 = max(1, (delta + 2) // 3)
        if b_max_2 < b_min_2:
            continue
        min_err_2 = delta / (2.0 * b_max_2)
        if min_err_2 < best_err:
            if delta % 2 == 0:
                b_start = b_max_2 if b_max_2 % 2 == 0 else b_max_2 - 1
            else:
                b_start = b_max_2 if b_max_2 % 2 == 1 else b_max_2 - 1
            if b_start >= b_min_2:
                for b in range(b_start, b_min_2 - 1, -2):
                    curr_err = delta / (2.0 * b)
                    if curr_err >= best_err:
                        break
                    a = (3 * b - delta) // 2
                    if a > N or a < 1:
                        continue
                    if sigma[a] == sigma[b]:
                        best_err = curr_err
                        best_a = a
                        best_b = b
                        break
        
        if time.time() - start_time > deadline:
            break
    
    return [int(best_a), int(best_b)]
