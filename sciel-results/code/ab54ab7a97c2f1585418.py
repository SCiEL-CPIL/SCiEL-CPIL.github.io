import sys
import time
import math

# EVOLVE-BLOCK-START
MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Compute sigma(n) for all n <= MAX_N using a linear sieve
    # This is O(N) and much faster than O(N log N) divisor summation in Python.
    
    sigma = [0] * (MAX_N + 1)
    primes = []
    is_composite = [False] * (MAX_N + 1)
    # spf[i] will store the smallest prime factor of i
    # We can also store the "power part" of the prime factorization to compute sigma efficiently.
    # Let i = p^k * m where gcd(p, m) = 1.
    # sigma(i) = sigma(p^k) * sigma(m) = (p^(k+1)-1)/(p-1) * sigma(m)
    
    # To avoid storing large arrays for spf and power, we can just store:
    # sigma[i]
    # low_prime[i] : the smallest prime factor of i
    # low_power[i] : the highest power of low_prime[i] dividing i (i.e., p^k)
    # This allows O(1) update in linear sieve.
    
    low_prime = [0] * (MAX_N + 1)
    low_power = [0] * (MAX_N + 1)
    
    sigma[1] = 1
    low_prime[1] = 1
    low_power[1] = 1
    
    for i in range(2, MAX_N + 1):
        if not is_composite[i]:
            primes.append(i)
            low_prime[i] = i
            low_power[i] = i
            sigma[i] = 1 + i
        
        for p in primes:
            if i * p > MAX_N:
                break
            is_composite[i * p] = True
            
            if i % p == 0:
                # p is the smallest prime factor of i*p
                low_prime[i * p] = p
                # low_power[i] is p^k. Then low_power[i*p] is p^(k+1) = low_power[i] * p
                low_power[i * p] = low_power[i] * p
                # sigma(i*p) = sigma(i / p^k) * sigma(p^(k+1))
                # We don't store sigma(i / p^k) directly, but we can derive it?
                # Actually, standard linear sieve for sigma:
                # sigma(i*p) = sigma(i) * (1 + p + ... + p^(k+1)) / (1 + ... + p^k) ? No.
                # Let i = p^k * m. sigma(i) = sigma(p^k) * sigma(m).
                # i*p = p^(k+1) * m. sigma(i*p) = sigma(p^(k+1)) * sigma(m).
                # sigma(p^(k+1)) = sigma(p^k) * (1+p^(k+1))/(1+p^k) ? No.
                # sigma(p^k) = (p^(k+1)-1)/(p-1).
                # sigma(p^(k+1)) = (p^(k+2)-1)/(p-1).
                # Ratio is not integer necessarily.
                
                # Better approach:
                # Store sigma[i].
                # When i % p == 0:
                #   Let low_power[i] = p^k.
                #   sigma(i) = sigma(m) * sigma(p^k) where i = m * p^k.
                #   We want sigma(i*p) = sigma(m) * sigma(p^(k+1)).
                #   sigma(p^(k+1)) = sigma(p^k) + p^(k+1).
                #   So sigma(i*p) = sigma(i) + sigma(m) * p^(k+1).
                #   We need sigma(m).
                #   sigma(m) = sigma(i) / sigma(p^k).
                #   This requires division which might be slow or inexact if not careful? It is exact integer division.
                #   But we don't have sigma(p^k) stored separately.
                
                # Alternative: Store `sigma_prime_part` or just compute directly?
                # Let's use the property:
                # sigma(i*p) = sigma(i) * (p+1) if p does not divide i.
                # If p divides i, let i = p^k * m.
                # sigma(i) = sigma(p^k) * sigma(m).
                # sigma(i*p) = sigma(p^(k+1)) * sigma(m).
                # sigma(p^(k+1)) = sigma(p^k) + p^(k+1).
                # So sigma(i*p) = sigma(i) + sigma(m) * p^(k+1).
                # We need sigma(m).
                # m = i / low_power[i].
                # We can store `sigma_m`? No, too much memory.
                
                # Let's stick to a simpler O(N log N) but optimized with NumPy?
                # Or just accept the slower Python sieve if we can make the search very fast?
                # The previous sieve took ~60s.
                # Linear sieve in Python is tricky to get faster than optimized divisor summation?
                # Actually, `for p in primes` loop in linear sieve is fast in Python if list is local.
                
                # Let's try to implement the linear sieve correctly with the `sigma_m` trick.
                # We can store an auxiliary array `sigma_res` or just calculate.
                # Actually, we can just store `low_prime` and `low_power` and `sigma`.
                # To get sigma(m), we can do:
                # m = i // low_power[i]
                # sigma_m = sigma[i] // sigma[low_power[i]]
                # This requires sigma[low_power[i]] to be computed. Since low_power[i] < i, it is.
                
                lp = low_power[i]
                sigma_lp = sigma[lp]
                sigma_m = sigma[i] // sigma_lp
                sigma[i * p] = sigma[i] + sigma_m * (lp * p)
                # Check: sigma(p^(k+1)) = sigma(p^k) + p^(k+1).
                # sigma(m) * sigma(p^(k+1)) = sigma(m) * (sigma(p^k) + p^(k+1)) = sigma(m)*sigma(p^k) + sigma(m)*p^(k+1) = sigma(i) + sigma(m)*p^(k+1).
                # Correct.
                
            else:
                # p does not divide i
                low_prime[i * p] = p
                low_power[i * p] = p
                sigma[i * p] = sigma[i] * (1 + p)
                
        if time.time() > deadline:
            # Fallback if sieve is too slow (unlikely with linear sieve, but safe)
            return [11, 6]

    # 2. Search for best pair (a, b)
    # Minimize |a/b - 3/2| = |2a - 3b| / (2b)
    
    best_a, best_b = 11, 6
    best_err_num, best_err_den = 4, 12 # Error for 11/6 is 1/3 = 4/12
    best_score_val = 1/3.0
    
    # We iterate b downwards.
    # Pruning: if 1/(2*b) >= best_score_val, then for this and all smaller b,
    # even the best possible k=1 error is worse than current best.
    # Note: k=1 is only possible if 3b is odd. If 3b is even, min k is 2, error 1/b.
    # 1/b > 1/(2b). So the condition 1/(2*b) >= best_score_val is a safe pruning condition.
    
    for b in range(MAX_N, 0, -1):
        if time.time() > deadline:
            break
            
        if 1 / (2 * b) >= best_score_val:
            break
            
        sb = sigma[b]
        base = 3 * b // 2
        
        # Check candidates a near base
        # We want to minimize k = |2a - 3b|.
        # If 3b is odd, base is floor(3b/2).
        # a = base -> 2a - 3b = -1 (k=1)
        # a = base + 1 -> 2a - 3b = 1 (k=1)
        # If 3b is even, base is 3b/2.
        # a = base -> k=0 (invalid)
        # a = base +/- 1 -> k=2
        
        # We check a window [base-5, base+5] to be safe and simple.
        # Since we want smallest k, we should check in order of increasing k.
        # But checking 11 values is fast.
        
        found_k = 0
        for offset in range(-5, 6):
            a = base + offset
            if a < 1 or a > MAX_N:
                continue
            if sigma[a] == sb:
                k = abs(2 * a - 3 * b)
                if k == 0:
                    continue
                # Compare k/(2b) with best_err_num/best_err_den
                # k * best_err_den < best_err_num * 2b
                if k * best_err_den < best_err_num * (2 * b):
                    best_a, best_b = a, b
                    best_err_num, best_err_den = k, 2 * b
                    best_score_val = k / (2 * b)
                    found_k = k
                    # If k=1, we found the best possible for this b.
                    # Can we break inner loop? Yes, k cannot be smaller than 1.
                    if k == 1:
                        break

    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
