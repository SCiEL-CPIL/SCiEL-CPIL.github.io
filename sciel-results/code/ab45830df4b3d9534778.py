import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Use a linear sieve to compute sigma(k) for k up to K
    # K = 5*10^7 should be fast enough with linear sieve
    K = 50 * 10**6

    # Linear sieve for sigma
    # sigma[n] = sum of divisors of n
    # We need to track the power of the smallest prime factor
    sigma = np.zeros(K + 1, dtype=np.int64)
    sigma[1] = 1
    spf = np.zeros(K + 1, dtype=np.int64)  # smallest prime factor
    spf_power = np.zeros(K + 1, dtype=np.int64)  # power of spf in n
    primes = []

    for i in range(2, K + 1):
        if spf[i] == 0:
            spf[i] = i
            spf_power[i] = 1
            sigma[i] = 1 + i
            primes.append(i)

        for p in primes:
            if p > spf[i] or i * p > K:
                break
            spf[i * p] = p

            if p == spf[i]:
                # p divides i
                # i = p^a * m, gcd(m,p)=1
                # i*p = p^(a+1) * m
                # sigma(i*p) = sigma(i) * (1 + p + ... + p^(a+1)) / (1 + p + ... + p^a)
                # = sigma(i) * (p^(a+2)-1)/(p^(a+1)-1)
                # But we can compute it as:
                # sigma(i*p) = sigma(m) * (1 + p + ... + p^(a+1))
                # sigma(i) = sigma(m) * (1 + p + ... + p^a)
                # So sigma(i*p) = sigma(i) * (1 + p + ... + p^(a+1)) / (1 + p + ... + p^a)
                
                # We need to compute the ratio. Let a = spf_power[i]
                a = spf_power[i]
                # (1 + p + ... + p^(a+1)) / (1 + p + ... + p^a)
                # = (p^(a+2)-1)/(p^(a+1)-1)
                # But we can also think: sigma(i*p) = sigma(i) + p^(a+1)*m*(1+p+...+p^a)/sigma(m)... 
                # Actually simpler: 
                # Let i = p^a * m
                # sigma(i) = sigma(m) * (1+p+...+p^a)
                # sigma(i*p) = sigma(m) * (1+p+...+p^(a+1))
                # So sigma(i*p) = sigma(i) * (1+p+...+p^(a+1)) / (1+p+...+p^a)
                
                # Compute (1+p+...+p^(a+1)) / (1+p+...+p^a) = (p^(a+2)-1)/(p^(a+1)-1)
                # For small a, we can compute directly
                # Actually, let's use: sigma(i*p) = sigma(i) + i*p * (1+p+...+p^a) / sigma(m) ... no
                
                # Better approach: 
                # sigma(i*p) = sigma(i) * (1 + p) if p^1 || i (a=1)
                # More generally: sigma(i*p) = sigma(i) * (p^(a+2)-1) / (p^(a+1)-1)
                
                # Let's compute p^(a+1) and p^(a+2)
                # p^(a+1) might be large but i*p <= K so p^(a+1) * m <= K
                # p^(a+1) <= K/m <= K
                p_power_a_plus_1 = p ** (a + 1)
                # sigma(m) = sigma(i) / (1+p+...+p^a)
                # This is getting complex. Let's use a different approach.
                
                # Alternative: store the "sigma contribution" of the spf part
                # Let's define: sigma_spf[i] = 1 + p + ... + p^a where p^a || i for p=spf[i]
                # Then sigma[i] = sigma[i / p^a] * sigma_spf[i]
                # sigma[i*p] = sigma[i / p^a] * (1 + p + ... + p^(a+1))
                # = sigma[i] * (1 + p + ... + p^(a+1)) / sigma_spf[i]
                
                # We need to store sigma_spf. Let me add that.
                # Actually, let's just compute it directly for now with a helper.
                
                # For efficiency, let's precompute or use the formula:
                # ratio = (p^(a+2)-1) / (p^(a+1)-1)
                # But integer division may not be exact in float.
                
                # Let's use: sigma(i*p) = sigma(i) + i * (1 + p + ... + p^a) / sigma(m) * p^(a+1)... 
                # This is getting too complex. Let me use a simpler O(n log n) approach but with larger K.
                
                # Actually, for the linear sieve, the standard formula is:
                # If p | i: sigma[i*p] = sigma[i] + i*p  ... NO, that's not right.
                
                # Let me look at this differently. Use the multiplicative property:
                # sigma is multiplicative.
                # If gcd(i,p)=1: sigma(i*p) = sigma(i)*sigma(p) = sigma(i)*(1+p)
                # If p|i: let i = p^a * m, gcd(m,p)=1
                #   sigma(i*p) = sigma(p^(a+1)) * sigma(m)
                #   sigma(i) = sigma(p^a) * sigma(m)
                #   sigma(i*p) = sigma(i) * sigma(p^(a+1))/sigma(p^a)
                #   = sigma(i) * (1+p+...+p^(a+1))/(1+p+...+p^a)
                
                # To compute this, we need a = v_p(i). We have spf_power[i] = a.
                # Let's compute the ratio as a float and round, or compute exactly.
                # p^(a+1) * m = i*p / p^a * p = ... 
                # Actually i = p^a * m, so m = i / p^a
                # sigma(p^a) = (p^(a+1)-1)/(p-1)
                # sigma(p^(a+1)) = (p^(a+2)-1)/(p-1)
                # ratio = (p^(a+2)-1)/(p^(a+1)-1)
                
                # For small p and a, this is manageable.
                # Let's compute p^(a+1) and p^(a+2)
                # Since i*p <= K and i = p^a * m, we have p^a * m * p <= K
                # So p^(a+1) <= K/m <= K, which fits in int64 for K=5*10^7
                
                p_pow_a1 = p ** (a + 1)
                p_pow_a2 = p ** (a + 2)
                # ratio = (p_pow_a2 - 1) / (p_pow_a1 - 1)
                # This should be exact integer division? Not necessarily.
                # (p^(a+2)-1)/(p^(a+1)-1) = 1 + p/(p^(a+1)-1) ... not integer in general.
                # Wait, let's check: a=1, p=2: (2^3-1)/(2^2-1) = 7/3, not integer!
                # So the ratio is NOT an integer. We need to compute sigma(i*p) differently.
                
                # Better: sigma(i*p) = sigma(m) * (1 + p + ... + p^(a+1))
                # sigma(m) = sigma(i) / (1 + p + ... + p^a)
                # So sigma(i*p) = sigma(i) * (1 + p + ... + p^(a+1)) / (1 + p + ... + p^a)
                # Let S_a = 1 + p + ... + p^a
                # sigma(i*p) = sigma(i) * (S_a + p^(a+1)) / S_a
                # = sigma(i) + sigma(i) * p^(a+1) / S_a
                # = sigma(i) + sigma(m) * p^(a+1)
                
                # So sigma(i*p) = sigma(i) + sigma(m) * p^(a+1)
                # where sigma(m) = sigma(i) / S_a
                
                # We need S_a = 1 + p + ... + p^a
                # And sigma(m) = sigma(i) / S_a (exact division)
                
                # Let's compute S_a
                # S_a = (p^(a+1) - 1) / (p - 1)
                S_a = (p_pow_a1 - 1) // (p - 1)
                sigma_m = sigma[i] // S_a  # This should be exact
                
                sigma[i * p] = sigma[i] + sigma_m * p_pow_a1
                spf_power[i * p] = a + 1
            else:
                # p does not divide i
                sigma[i * p] = sigma[i] * (1 + p)
                spf_power[i * p] = 1

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
    
    # Convert to Python ints
    best_ks = [int(x) for x in best_ks]
    
    return {'M': best_M, 'k': best_ks}
