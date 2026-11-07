<<<<<<< SEARCH
import time
import math

def construct():
    start_time = time.time()
    # Safety margin: 100 seconds
    deadline = start_time + 100.0
    
    # Fallback solution
    fallback_M = 333312
    fallback_ks = [336, 372, 434]
    
    # We will search for k in range [1, N]
    # Estimate N based on time.
    # Let's start with N = 2 * 10**6 and see if we can go higher.
    # Or just pick a fixed N that is safe.
    # 10**6 is very safe. 5*10**6 is likely safe.
    # Let's try 5 * 10**6.
    
    N = 5 * 10**6
    
    # Check if we have enough time for a larger N?
    # For now, fixed N to avoid complexity.
    
    # Compute sigma(k) for k = 1..N
    # sigma = [0] * (N + 1)
    # Using a list of ints.
    sigma = [0] * (N + 1)
    
    # Sieve for sigma
    # This is O(N log N)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    # Check time
    if time.time() > deadline - 10:
        return {'M': fallback_M, 'k': fallback_ks}
        
    counts = {}
    solutions = {}
    
    # Iterate k and compute M
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M > 10**18:
            continue
        if M not in counts:
            counts[M] = 1
            solutions[M] = [k]
        else:
            counts[M] += 1
            solutions[M].append(k)
            
    # Find the M with the most solutions
    max_count = 0
    best_M = fallback_M
    best_ks = fallback_ks
    
    for M, c in counts.items():
        if c > max_count:
            max_count = c
            best_M = M
            best_ks = solutions[M]
            
    # If we found a better solution, return it
    if max_count > len(fallback_ks):
        # Ensure types are correct
        return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
    else:
        return {'M': fallback_M, 'k': fallback_ks}
=======
import time
import math

def construct():
    start_time = time.time()
    # Safety margin: 12 seconds before 150s limit
    deadline = start_time + 138.0
    
    # Fallback solution
    fallback_M = 333312
    fallback_ks = [336, 372, 434]
    
    # Use linear sieve for sigma to go much further than 5M
    # Linear sieve is O(N), so we can try N = 30M or more
    N = 30 * 10**6
    
    # Linear sieve for sigma(k)
    # sigma[1] = 1
    sigma = [0] * (N + 1)
    sigma[1] = 1
    
    # For linear sieve, we track:
    # lp[i] = smallest prime factor of i
    # sp[i] = smallest prime of i (same as lp[i] for primes)
    # sigma[i] = sum of divisors
    # We need: for i = p^a * m where p doesn't divide m:
    #   sigma[i] = sigma[m] * (p^(a+1) - 1) / (p - 1)
    # Track: p_power[i] = p^(a+1) where p^a || i and p = smallest prime of i
    #         p_val[i] = p (the smallest prime)
    
    lp = [0] * (N + 1)  # smallest prime factor
    p_power = [0] * (N + 1)  # p^(a+1) for smallest prime p^a || i
    p_val = [0] * (N + 1)  # the smallest prime itself
    primes = []
    
    for i in range(2, N + 1):
        if lp[i] == 0:
            lp[i] = i
            p_val[i] = i
            p_power[i] = i * i  # p^(1+1) = p^2
            sigma[i] = 1 + i  # 1 + p
            primes.append(i)
        
        for p in primes:
            if p > lp[i]:
                break
            if i * p > N:
                break
            
            lp[i * p] = p
            if p == lp[i]:
                # p divides i, so p^a || i, now p^(a+1) || i*p
                p_val[i * p] = p
                p_power[i * p] = p_power[i] * p
                # sigma[i*p] = sigma[i] * p - p_power[i]
                # Because sigma[i] = sigma[m] * (p^(a+1)-1)/(p-1)
                # sigma[i*p] = sigma[m] * (p^(a+2)-1)/(p-1)
                # = sigma[i] * p - p^(a+1) * (p-1)/(p-1) ... let me derive:
                # sigma[i] = sigma[m] * (1 + p + ... + p^a)
                # sigma[i*p] = sigma[m] * (1 + p + ... + p^(a+1))
                # = sigma[i] + sigma[m] * p^(a+1)
                # = sigma[i] + (sigma[i] - p^(a+1) + 1) ... no
                # sigma[m] * p^(a+1) = (sigma[i] / (1+p+...+p^a)) * p^(a+1)
                # Simpler: sigma[i*p] = sigma[i] * p - p^(a+1)
                # Check: sigma[i] = sigma[m]*(p^(a+1)-1)/(p-1)
                # sigma[i]*p - p^(a+1) = sigma[m]*(p^(a+2)-p)/(p-1) - p^(a+1)
                # = sigma[m]*((p^(a+2)-p) - p^(a+1)*(p-1))/(p-1)
                # = sigma[m]*((p^(a+2)-p) - p^(a+2)+p^(a+1))/(p-1)
                # = sigma[m]*(p^(a+1)-p)/(p-1)
                # = sigma[m]*p*(p^a - 1)/(p-1) ... hmm that's not right either
                # Let me just use: sigma[i*p] = sigma[i] + sigma[i] - p_power[i] = 2*sigma[i] - p_power[i]? No.
                # Actually: sigma[i*p] where p^a || i:
                # i = p^a * m, gcd(p,m)=1
                # i*p = p^(a+1) * m
                # sigma(i*p) = sigma(m) * (1+p+...+p^(a+1))
                # sigma(i) = sigma(m) * (1+p+...+p^a)
                # sigma(m) = sigma(i) / (1+p+...+p^a) = sigma(i) * (p-1)/(p^(a+1)-1)
                # sigma(i*p) = sigma(i) * (p-1)/(p^(a+1)-1) * (p^(a+2)-1)/(p-1)
                # = sigma(i) * (p^(a+2)-1)/(p^(a+1)-1)
                # = sigma(i) * (1 + p^(a+1))
                # Wait: (p^(a+2)-1)/(p^(a+1)-1) = (p^(a+1)*p - 1)/(p^(a+1)-1)
                # Let x = p^(a+1): (xp-1)/(x-1) = 1 + (x-1)/(x-1) = 1 + 1? No.
                # (xp-1)/(x-1): xp-1 = x(p-1)+(p-1) = (p-1)(x+1)
                # So (xp-1)/(x-1) = (p-1)(x+1)/(x-1) ... that doesn't simplify nicely.
                
                # Let me just track it differently.
                # p_power[i] = p^(a+1) where p^a || i
                # sigma[i*p] = sigma[i] * p - p_power[i]
                # Verify with i=2, p=2: sigma[2]=3, p_power[2]=4, sigma[4]=7
                # 3*2 - 4 = 2. But sigma[4]=1+2+4=7. Wrong.
                
                # Try: sigma[i*p] = sigma[i] + p_power[i] * (sigma[i] / p_power[i] ... )
                # Let me just use the formula directly.
                # sigma[i] = sigma[m] * S(a) where S(a) = 1+p+...+p^a = (p^(a+1)-1)/(p-1)
                # sigma[i*p] = sigma[m] * S(a+1) = sigma[m] * (p^(a+2)-1)/(p-1)
                # sigma[i*p] / sigma[i] = S(a+1)/S(a) = (p^(a+2)-1)/(p^(a+1)-1)
                # With p_power[i] = p^(a+1):
                # ratio = (p * p^(a+1) - 1) / (p^(a+1) - 1) = (p*p_power[i] - 1) / (p_power[i] - 1)
                # This involves division which may not be exact in integer arithmetic... but it should be.
                # sigma[i*p] = sigma[i] * (p * p_power[i] - 1) // (p_power[i] - 1)
                # Check i=2,p=2: sigma[2]=3, p_power[2]=4
                # sigma[4] = 3 * (2*4-1) // (4-1) = 3*7//3 = 7. Correct!
                # Check i=4,p=2: sigma[4]=7, p_power[4]=8 (since 2^2||4, p^3=8)
                # sigma[8] = 7 * (2*8-1)//(8-1) = 7*15//7 = 15. sigma[8]=1+2+4+8=15. Correct!
                # Check i=3,p=3: sigma[3]=4, p_power[3]=9
                # sigma[9] = 4*(3*9-1)//(9-1) = 4*26//8 = 13. sigma[9]=1+3+9=13. Correct!
                
                sigma[i * p] = sigma[i] * (p * p_power[i] - 1) // (p_power[i] - 1)
            else:
                # p does not divide i, so p^1 || i*p
                p_val[i * p] = p
                p_power[i * p] = p * p  # p^2
                # sigma[i*p] = sigma[i] * (1 + p)
                sigma[i * p] = sigma[i] * (1 + p)
            
            # Time check every 1M iterations
            if (i * p) % (1000000) == 0:
                if time.time() > deadline - 15:
                    # Too late, use what we have so far
                    # Need to handle partial sigma array
                    N = i * p
                    break
        
        if time.time() > deadline - 15:
            N = i
            break
    
    # Check time
    if time.time() > deadline - 10:
        return {'M': fallback_M, 'k': fallback_ks}
    
    counts = {}
    solutions = {}
    
    # Iterate k and compute M
    for k in range(1, N + 1):
        if sigma[k] == 0:
            continue
        M = k * sigma[k]
        if M > 10**18:
            continue
        if M not in counts:
            counts[M] = 1
            solutions[M] = [k]
        else:
            counts[M] += 1
            solutions[M].append(k)
            
    # Find the M with the most solutions
    max_count = 0
    best_M = fallback_M
    best_ks = fallback_ks
    
    for M, c in counts.items():
        if c > max_count:
            max_count = c
            best_M = M
            best_ks = solutions[M]
            
    # If we found a better solution, return it
    if max_count > len(fallback_ks):
        # Ensure types are correct
        return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
    else:
        return {'M': fallback_M, 'k': fallback_ks}
>>>>>>> REPLACE
