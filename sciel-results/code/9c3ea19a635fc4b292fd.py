import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 10000000
    
    # Compute smallest prime factor for all n up to N
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    # Compute sigma(n) for all n using SPF
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    
    for n in range(2, N + 1):
        if time.time() > deadline - 5.0:
            break
        p = spf[n]
        m = n // p
        # Count power of p in n
        power = 1
        while m % p == 0:
            m //= p
            power *= p
        
        # sigma(n) = sigma(m) * (p^(k+1)-1)/(p-1) / (p^k - 1)/(p-1)... 
        # Actually: if n = m * p^k where gcd(m,p)=1, then sigma(n) = sigma(m) * (p^(k+1)-1)/(p-1)
        # But we need to be careful. Let's use:
        # sigma(n) = sigma(m) * (1 + p + p^2 + ... + p^k)
        # where m = n / p^k and gcd(m,p)=1
        
        # We already have power = p^k
        # sum = 1 + p + p^2 + ... + p^k = (p^(k+1)-1)/(p-1)
        # But we can compute it incrementally
        
        # Actually, let me just compute it directly
        # We have m and power = p^k
        # sigma(n) = sigma(m) * (1 + p + ... + p^k)
        
        # Compute geometric sum
        geom_sum = 1
        pk = 1
        for _ in range(int(np.log(power) / np.log(p)) + 1):  # This is messy
            break
        
        # Simpler: just use the formula
        # k = exponent of p in n
        # We can find k by dividing
        temp = n
        k = 0
        while temp % p == 0:
            temp //= p
            k += 1
        # Now temp = m, and p^k divides n
        # sigma(n) = sigma(temp) * (p^(k+1)-1)/(p-1)
        # But sigma(temp) might not be computed yet if temp > n... no, temp < n
        # Actually temp = n / p^k < n, so sigma[temp] is already computed
        
        # Compute (p^(k+1)-1)/(p-1)
        # This is 1 + p + p^2 + ... + p^k
        geom = 1
        pk = 1
        for i in range(k):
            pk *= p
            geom += pk
        
        sigma[n] = sigma[temp] * geom
    
    # Now group by sigma value
    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s not in groups:
            groups[s] = []
        groups[s].append(n)
    
    # Find best pair
    best_error = Fraction(1, 3)
    best_pair = (11, 6)
    
    group_sets = {s: set(lst) for s, lst in groups.items()}
    
    for s, lst in groups.items():
        if time.time() > deadline - 3.0:
            break
        if len(lst) < 2:
            continue
        sset = group_sets[s]
        for b in lst:
            a1 = (3 * b) // 2
            a2 = a1 + 1 if (3 * b) % 2 != 0 else a1
            for a in [a1, a2]:
                if a <= 0 or a > N or a == b:
                    continue
                if a in sset:
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    err = Fraction(num, den)
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)
    
    return [best_pair[0], best_pair[1]]
