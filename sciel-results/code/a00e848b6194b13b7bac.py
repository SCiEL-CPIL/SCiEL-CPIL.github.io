import math
import time

def sigma(n):
    """Calculate the sum of divisors of n."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            term = 1
            while temp % p == 0:
                temp //= p
                term *= p
            result *= (term * (p + 1) // (p - 1))
        p += 1
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for the 120s limit
    
    # We want M = p * q where p and q are of the form 6^a * 7^b.
    # For k = 6^a * 7^b, sigma(k) = (6^(a+1)-1)/5 * (7^(b+1)-1)/6.
    # We need p != q.
    
    # Generate candidates for p and q.
    # We want p and q to be large enough so that M <= 10^18.
    # p * q <= 10^18 => p, q can be up to ~10^9.
    # 6^11 ~ 3.6 * 10^8. 6^12 > 10^9.
    # So a can be up to 11.
    # 7^10 ~ 2.8 * 10^8. 7^11 > 10^9.
    # So b can be up to 10.
    
    candidates = []
    for a in range(12): # 0..11
        for b in range(11): # 0..10
            val = (6 ** a) * (7 ** b)
            if val < 10**9: # Keep p, q reasonable to allow M <= 10^18
                candidates.append(val)
                
    # We need distinct p, q.
    # Let's pick a few large ones to maximize M (which might allow more k? No, k <= sqrt(M). 
    # Actually, larger M allows larger k, but k must satisfy k*sigma(k)=M.
    # The structure of M determines the solutions.
    # If M = p*q, solutions are k such that k*sigma(k) = p*q.
    # We know k=p and k=q are solutions.
    # Are there others?
    # If k = 1, sigma(1)=1, M=1.
    # If k is prime, sigma(k)=k+1. k(k+1)=M.
    # If k = p*q? No, k <= sqrt(M).
    
    # Let's just compute M for a pair and search for k.
    # To maximize the number of solutions, we should probably look for M that is a "highly composite" number in terms of preimages under k*sigma(k).
    # However, finding such M is hard.
    # The prompt asks for a DIFFERENT strategy.
    # The current program returns 3 solutions.
    # Can we find M with 4 or more?
    
    # Let's try to find M with more solutions by searching small M?
    # No, M can be up to 10^18.
    
    # Let's try a different construction.
    # Consider M = LCM of some k*sigma(k).
    # If we pick k1, k2, ..., we want M to be a multiple of k_i * sigma(k_i) for all i.
    # Then k_i * sigma(k_i) | M. But we need k_i * sigma(k_i) == M.
    # So M must be exactly k_i * sigma(k_i).
    # This means k_i * sigma(k_i) must be the same for all i.
    # This is the "collision" problem.
    
    # Let's search for collisions among small k.
    # k*sigma(k) values.
    # If we find 4 k's with the same value, we win.
    
    seen = {}
    max_k_to_check = 100000 # Check up to 100k for k
    # This might be slow if we compute sigma naively for each.
    # But 100k is small.
    
    # To speed up, we can sieve sigma.
    # But 100k is fast enough with trial division?
    # sqrt(100k) ~ 316. 100k * 316 ops = 3*10^7. Doable in Python? Maybe 1-2 seconds.
    
    # Let's optimize sigma calculation with a sieve for small numbers.
    # Or just use the trial division for k up to 10^5.
    
    # Actually, let's just check the current best known collisions.
    # 336, 372, 434 -> 333312.
    # Are there others?
    # Let's scan k from 1 to 200000 and store k*sigma(k) in a dict.
    # If we find a value with count >= 4, we return it.
    
    # To be safe with time, we'll limit the scan.
    
    sigma_vals = [1] * (max_k_to_check + 1)
    for i in range(2, max_k_to_check + 1):
        for j in range(i, max_k_to_check + 1, i):
            sigma_vals[j] += i
            
    # Now sigma_vals[k] is sigma(k).
    # Compute k * sigma(k).
    collisions = {}
    for k in range(1, max_k_to_check + 1):
        val = k * sigma_vals[k]
        if val in collisions:
            collisions[val].append(k)
        else:
            collisions[val] = [k]
            
    best_M = None
    best_k = []
    
    for M_val, ks in collisions.items():
        if len(ks) > len(best_k):
            # Check if M_val <= 10^18
            if M_val <= 10**18:
                # Check if all k <= sqrt(M_val)
                sqrt_M = math.isqrt(M_val)
                valid_ks = [k for k in ks if k <= sqrt_M]
                if len(valid_ks) > len(best_k):
                    best_k = valid_ks
                    best_M = M_val
                    
    # If we found something better than 3, return it.
    # Otherwise, return the known 3.
    
    if best_M is not None and len(best_k) >= 3:
        # Ensure distinctness (they are distinct by construction of the loop)
        # Ensure types
        return {'M': int(best_M), 'k': [int(k) for k in best_k]}
    else:
        # Fallback to the seed
        return {'M': 333312, 'k': [336, 372, 434]}

# EVOLVE-BLOCK-END
