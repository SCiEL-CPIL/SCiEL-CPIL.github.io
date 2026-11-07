import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    # Use a linear sieve to compute sigma(k) for k up to K
    # K = 10^7 should be fast enough with linear sieve
    K = 10 * 10**6

    # Linear sieve for sigma
    sigma = np.zeros(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    spf = np.zeros(K + 1, dtype=np.int64)  # smallest prime factor

    sigma[1] = 1

    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            spf[i] = i
            sigma[i] = 1 + i

        for p in primes:
            if p > spf[i] or i * p > K:
                break
            is_prime[i * p] = False
            spf[i * p] = p

            if i % p == 0:
                # p divides i
                # sigma(i*p) = sigma(i) * (1 + p + p^2 + ... + p^a) / (1 + p + ... + p^(a-1))
                # where p^a || i
                # More simply: if p^a || i, then p^(a+1) || i*p
                # sigma(i*p) = sigma(i) * (1 + p + ... + p^(a+1)) / (1 + p + ... + p^a)
                # = sigma(i) * (1 + p) ... no, that's not right.
                # Actually: sigma(n) = prod (p_i^(a_i+1) - 1)/(p_i - 1)
                # If p^a || i, then sigma(i) = sigma(i/p^a) * (1 + p + ... + p^a)
                # sigma(i*p) = sigma(i/p^a) * (1 + p + ... + p^(a+1))
                # So sigma(i*p) = sigma(i) * (1 + p + ... + p^(a+1)) / (1 + p + ... + p^a)
                # = sigma(i) * (1 + p) if a=0? No, a>=1 since p|i.
                # Let's compute: (1 + p + ... + p^(a+1)) / (1 + p + ... + p^a) = (p^(a+2)-1)/(p^(a+1)-1)
                # This is tricky. Let's use a different approach.
                # We can compute sigma multiplicatively.
                # Let i = p^a * m where gcd(m,p)=1
                # sigma(i) = (1+p+...+p^a) * sigma(m)
                # i*p = p^(a+1) * m
                # sigma(i*p) = (1+p+...+p^(a+1)) * sigma(m)
                # So sigma(i*p) = sigma(i) * (1+p+...+p^(a+1)) / (1+p+...+p^a)
                # = sigma(i) * (p^(a+2)-1)/(p-1) / ((p^(a+1)-1)/(p-1))
                # = sigma(i) * (p^(a+2)-1) / (p^(a+1)-1)
                
                # We need to find a. Since spf[i] = p, we can divide i by p until it's no longer divisible.
                # But this is expensive. Instead, let's store the power of the smallest prime factor.
                
                # Alternative: just compute sigma directly using the formula.
                # For now, let's use a simpler but slightly slower approach:
                # sigma[i*p] = sigma[i] * (1 + p) if p doesn't divide i, else more complex.
                # Since p divides i (p == spf[i]), we need the full formula.
                
                # Let's store for each number the power of its smallest prime factor.
                # This requires more memory. Let's try a different approach.
                
                # Actually, let's just use the standard linear sieve formula:
                # If p | i: sigma[i*p] = sigma[i] + i*p
                # Wait, that's not right either.
                
                # Let me look up the correct formula.
                # For multiplicative functions: if f is multiplicative and f(p^a) is known,
                # we can compute f(n) for all n.
                # sigma(p^a) = 1 + p + ... + p^a = (p^(a+1)-1)/(p-1)
                
                # In linear sieve, when p | i:
                # Let i = p^a * m, gcd(m,p)=1
                # sigma(i) = sigma(m) * (1+p+...+p^a)
                # sigma(i*p) = sigma(m) * (1+p+...+p^(a+1))
                # = sigma(i) * (1+p+...+p^(a+1)) / (1+p+...+p^a)
                
                # To compute this efficiently, we can store:
                # spf_power[i] = a where p^a || i for p = spf[i]
                # Then we can compute the ratio.
                
                # Let's add an array for this.
                pass  # We'll handle this below
                
            else:
                # p does not divide i
                # sigma(i*p) = sigma(i) * sigma(p) = sigma(i) * (1+p)
                sigma[i * p] = sigma[i] * (1 + p)
    
    # The above linear sieve is incomplete. Let's use a simpler approach:
    # Use the standard sieve for sigma.
    sigma = np.zeros(K + 1, dtype=np.int64)
    for i in range(1, K + 1):
        sigma[i::i] += i
    
    # Compute M values
    ks = np.arange(1, K + 1, dtype=np.int64)
    M_vals = ks * sigma[1:]
    
    # Find M with max count
    unique_Ms, counts = np.unique(M_vals, return_counts=True)
    
    if len(unique_Ms) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    best_M_idx = np.argmax(counts)
    best_M = int(unique_Ms[best_M_idx])
    
    # Get all k's for this M
    best_ks = ks[M_vals == best_M].tolist()
    
    return {'M': best_M, 'k': best_ks}
