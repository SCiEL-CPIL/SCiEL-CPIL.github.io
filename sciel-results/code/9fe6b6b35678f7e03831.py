import time
import math

def construct():
    t_start = time.time()
    deadline = t_start + 55.0
    
    MAX_N = 10_000_000
    
    # 1. Linear Sieve for Smallest Prime Factor (SPF) and Sigma
    spf = [0] * (MAX_N + 1)
    sigma = [0] * (MAX_N + 1)
    primes = []
    sigma[1] = 1
    
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            sigma[i] = 1 + i
        for p in primes:
            if p > spf[i] or i * p > MAX_N:
                break
            spf[i * p] = p
            if i % p == 0:
                # p divides i, so p^k || i for some k >= 1
                # Let i = p^k * m where gcd(m, p) = 1
                # sigma(i*p) = sigma(p^{k+1}) * sigma(m)
                # sigma(i) = sigma(p^k) * sigma(m)
                # sigma(p^{k+1}) / sigma(p^k) = (p^{k+2}-1)/(p-1) / ((p^{k+1}-1)/(p-1)) = (p^{k+2}-1)/(p^{k+1}-1)
                # We can compute this iteratively or use the property:
                # If i = p^k * m, then i*p = p^{k+1} * m
                # sigma(i*p) = sigma(i) * (p^{k+2}-1)/(p^{k+1}-1) ? No.
                # sigma(i*p) = sigma(p^{k+1}) * sigma(m)
                # sigma(i) = sigma(p^k) * sigma(m)
                # Ratio = sigma(p^{k+1}) / sigma(p^k)
                # Let's find p^k.
                # Actually, a simpler recurrence:
                # Let i = p^k * m.
                # We know spf[i] = p.
                # We can find the highest power of p dividing i.
                # But in linear sieve, we usually just update.
                # sigma(i*p) = sigma(i) + p * sigma(i/p) ? No.
                # Standard linear sieve for sigma:
                # If p | i:
                #   Let i = p^k * m.
                #   sigma(i) = (1 + p + ... + p^k) * sigma(m)
                #   i*p = p^{k+1} * m
                #   sigma(i*p) = (1 + p + ... + p^{k+1}) * sigma(m)
                #   sigma(i*p) = sigma(i) + p^{k+1} * sigma(m)
                #   Note that i/p^{k-1} = p * m ? No.
                #   i = p^k m.
                #   i/p = p^{k-1} m.
                #   sigma(i/p) = (1 + ... + p^{k-1}) sigma(m).
                #   This is getting complicated to track k.
                
                # Alternative:
                # If p | i, let j = i/p.
                # If p | j, then p^2 | i.
                # It is easier to just compute sigma from spf in a separate pass if needed, 
                # but linear sieve can do it.
                
                # Let's stick to the previous plan: Compute SPF, then compute Sigma in a separate loop.
                # This is O(N log N) or O(N) if done carefully?
                # Computing sigma from SPF:
                # For each i, find prime factors.
                # Total cost is sum of number of prime factors, which is O(N log log N).
                # This is fast enough.
                
                pass
            else:
                sigma[i * p] = sigma[i] * (1 + p)
                
    # The logic above for the case p|i is incomplete/incorrect in the comment block.
    # Let's re-implement the sigma computation correctly after SPF is built.
    # Actually, the standard linear sieve for sigma is:
    # sigma[1] = 1
    # For i from 2 to N:
    #   if spf[i] == 0: prime
    #   for p in primes:
    #     if p > spf[i] or i*p > N: break
    #     spf[i*p] = p
    #     if i % p == 0:
    #        # p divides i
    #        # Let i = p^k * m
    #        # We need to update sigma[i*p]
    #        # sigma[i*p] = sigma[i] + p^k * sigma[m] * p ? No.
    #        # sigma[i*p] = sigma[i] * (1 + p) - sigma[i/p] * p ? No.
    #        
    #        # Let's use the property:
    #        # If i = p^k * m, gcd(m,p)=1
    #        # sigma(i) = (p^{k+1}-1)/(p-1) * sigma(m)
    #        # sigma(i*p) = (p^{k+2}-1)/(p-1) * sigma(m)
    #        # sigma(i*p) = sigma(i) + p^{k+1} * sigma(m)
    #        # Note that i = p^k * m => i/p^{k-1} = p * m ? No.
    #        # We can find p^k by dividing i by p until not divisible.
    #        # This makes it O(N log N).
    #        
    #        # Faster way:
    #        # Let j = i/p.
    #        # If j % p == 0, then p^2 | i.
    #        # This requires tracking the power of p.
    #        
    #        # Let's just compute sigma in a separate loop after SPF is ready.
    #        # It's simpler and likely fast enough (10^7 * small constant).
    #        pass
    #     else:
    #        sigma[i*p] = sigma[i] * (1+p)
        
    # Re-doing the sieve correctly for Sigma
    # We will just build SPF and then calculate Sigma.
    
    spf = [0] * (MAX_N + 1)
    primes = []
    for i in range(2, MAX_N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > MAX_N:
                break
            spf[i * p] = p
            
    if time.time() > deadline - 10:
        return [6, 11]

    # Precompute sigma
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    for i in range(2, MAX_N + 1):
        p = spf[i]
        n = i
        s = 1
        pk = 1
        while n % p == 0:
            n //= p
            pk *= p
            s += pk
        sigma[i] = s * sigma[n]
        
    if time.time() > deadline - 5:
        return [6, 11]

    best_err = float('inf')
    best_pair = (6, 11)
    
    for b in range(MAX_N, 0, -1):
        if time.time() > deadline - 2:
            break
            
        if 1.0 / (2.0 * b) >= best_err:
            break
            
        sb = sigma[b]
        
        if b % 2 == 0:
            a1 = (3 * b) // 2 - 1
            a2 = (3 * b) // 2 + 1
            candidates = [a1, a2]
        else:
            a1 = (3 * b - 1) // 2
            a2 = (3 * b + 1) // 2
            candidates = [a1, a2]
            
        for a in candidates:
            if a < 1 or a > MAX_N:
                continue
            if a == b:
                continue
            if sigma[a] == sb:
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue
                err = diff / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    
    return [int(best_pair[0]), int(best_pair[1])]
