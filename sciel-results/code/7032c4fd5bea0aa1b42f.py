import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Try multiple sieve sizes, starting large
    # M = k*sigma(k), k <= sqrt(M), M <= 10^18
    # For k up to K, M up to K*sigma(K) ~ K*K*6 (roughly)
    # So K ~ 10^7 gives M up to ~6*10^14, well within 10^18
    # K ~ 10^8 gives M up to ~6*10^16, still within 10^18
    
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3
    
    for K in [10**8, 5*10**7, 10**7, 5*10**6]:
        if time.time() > deadline:
            break
        
        try:
            # Linear sieve to compute sigma(k) for k=1..K
            # sigma(k) = sum of divisors
            # For prime p: sigma(p) = p+1
            # For p^a: sigma(p^a) = (p^(a+1)-1)/(p-1)
            # sigma is multiplicative
            
            sigma = [0] * (K + 1)
            sigma[1] = 1
            prime_powers = [0] * (K + 1)  # p^a for smallest prime p of k
            is_composite = [False] * (K + 1)
            primes = []
            
            # Linear sieve
            for i in range(2, K + 1):
                if not is_composite[i]:
                    primes.append(i)
                    sigma[i] = i + 1
                    prime_powers[i] = i
                for p in primes:
                    prod = i * p
                    if prod > K:
                        break
                    is_composite[prod] = True
                    if i % p == 0:
                        # p divides i, so p is the smallest prime of prod
                        pp = prime_powers[i]
                        new_pp = pp * p
                        # sigma(prod) = sigma(i) / (pp * (1 + p + ... + p^a)) * (pp*p * (1+...))
                        # Actually: if i = p^a * m where p doesn't divide m
                        # sigma(i) = (1+p+...+p^a) * sigma(m)
                        # prod = p^(a+1) * m
                        # sigma(prod) = (1+p+...+p^(a+1)) * sigma(m)
                        # sigma(prod) = sigma(i) * (1+p+...+p^(a+1)) / (1+p+...+p^a)
                        # = sigma(i) * (pp*p + 1) / (pp + 1) ... no
                        # Let's use: sigma(p^(a+1)*m) = sigma(p^a*m) * (p^(a+1)+1)/(p^a+1) ... no
                        # Better: sigma(p^(a+1)) = (p^(a+2)-1)/(p-1) = p*sigma(p^a) + 1
                        # So sigma(prod) = sigma(i) - sigma(m)*sigma(p^a) + sigma(m)*sigma(p^(a+1))
                        # This is getting complex. Let me use a different approach.
                        
                        # Simpler: sigma(prod) = sigma(i) + pp*p * sigma(i) // (pp + 1) * ... 
                        # Actually the cleanest: 
                        # Let i = p^a * m, gcd(p,m)=1
                        # sigma(i) = (1+p+...+p^a) * sigma(m)
                        # prod = p^(a+1) * m
                        # sigma(prod) = (1+p+...+p^(a+1)) * sigma(m)
                        # sigma(prod) = sigma(i) * (1+p+...+p^(a+1)) / (1+p+...+p^a)
                        # Let S_a = 1+p+...+p^a = pp + 1 ... no, S_a = (p^(a+1)-1)/(p-1)
                        # pp = p^a, so S_a = (p*pp - 1)/(p-1)
                        # S_{a+1} = (p^(a+2)-1)/(p-1) = (p*p*pp - 1)/(p-1)
                        # ratio = S_{a+1}/S_a = (p*p*pp - 1)/(p*pp - 1)
                        # This involves division which may not be exact...
                        
                        # Actually, let's just compute it directly:
                        # sigma(prod) = sigma(i) + pp*p * sigma(i) // (pp + 1)  -- NO this is wrong
                        
                        # Let me use the property: sigma is multiplicative
                        # If i = p^a * m with gcd(p,m)=1
                        # sigma(i) = sigma(p^a) * sigma(m)
                        # prod = p^(a+1) * m
                        # sigma(prod) = sigma(p^(a+1)) * sigma(m)
                        # sigma(p^(a+1)) = p * sigma(p^a) + 1  (since 1+p+...+p^(a+1) = p*(1+p+...+p^a) + 1)
                        # So sigma(prod) = (p * sigma(p^a) + 1) * sigma(m)
                        # And sigma(i) = sigma(p^a) * sigma(m)
                        # So sigma(prod) = (p * sigma(p^a) + 1) * sigma(m) = p * sigma(i) + sigma(m)
                        # But I need sigma(m)... I don't have it directly.
                        
                        # Alternative: sigma(prod) = sigma(i) + pp*p * sigma(m)
                        # where sigma(m) = sigma(i) / sigma(p^a) = sigma(i) / ((pp*p-1)/(p-1)) ... 
                        
                        # Simplest correct approach:
                        # sigma(p^(a+1)*m) = sigma(p^a*m) + p^(a+1)*sigma(m)
                        # = sigma(i) + pp*p * (sigma(i) / sigma(p^a))
                        # sigma(p^a) = (p^(a+1)-1)/(p-1) = (pp*p-1)/(p-1)
                        # So pp*p * sigma(i) / ((pp*p-1)/(p-1)) = pp*p*(p-1)*sigma(i)/(pp*p-1)
                        # This should be an integer.
                        
                        sp_a = (pp * p - 1) // (p - 1)  # sigma(p^a)
                        sig_m = sigma[i] // sp_a
                        sigma[prod] = sigma[i] + pp * p * sig_m
                        prime_powers[prod] = new_pp
                        break
                    else:
                        # p doesn't divide i, so prod = i * p where gcd(i,p)=1
                        sigma[prod] = sigma[i] * (p + 1)
                        prime_powers[prod] = p
            
            if time.time() > deadline:
                break
            
            # Now find collisions: M -> list of k
            m_to_k = defaultdict(list)
            for k in range(2, K + 1):
                M_val = k * sigma[k]
                if M_val <= 10**18:
                    m_to_k[M_val].append(k)
            
            if time.time() > deadline:
                break
            
            # Find M with most solutions
            best_M_local = None
            best_k_local = None
            best_count_local = 0
            for M_val, k_list in m_to_k.items():
                if len(k_list) > best_count_local:
                    best_count_local = len(k_list)
                    best_M_local = M_val
                    best_k_local = k_list
            
            if best_count_local > best_count:
                best_count = best_count_local
                best_M = best_M_local
                best_k_list = best_k_local[:1000]  # cap at 1000
                
        except MemoryError:
            continue
        except Exception:
            continue
    
    # Final safety check
    if time.time() > deadline - 2:
        pass
    
    # Validate and return
    result_k = [int(k) for k in best_k_list[:1000]]
    return {'M': int(best_M), 'k': result_k}
