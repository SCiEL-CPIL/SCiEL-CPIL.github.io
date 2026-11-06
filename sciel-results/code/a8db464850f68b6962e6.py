import math
import time

def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return is_prime

def construct():
    start_time = time.time()
    deadline = start_time + 5.0 # 5s is plenty for a subset search
    
    is_prime = _sieve(256)
    
    # Fallback: Valid solution with 9 distinct primes
    fallback_A = [1, 7, 13]
    fallback_B = [4, 6, 30]
    
    # Search for AP-AP solutions
    # A = a, a+d, a+2d
    # B = b, b+e, b+2e
    
    # Limit search space to keep it fast
    # We don't need to check all 8000x8000.
    # Just a random sample or a structured subset?
    # Let's try a structured subset of small steps.
    
    found = None
    
    # Try small steps first, as they are more likely to form constellations?
    # Actually, large steps avoid small primes (which are rare in specific patterns).
    # Let's just iterate.
    
    ap_starts = range(1, 129)
    ap_steps = range(1, 64)
    
    # To speed up, we can pre-compute the set of sums for A and check against B?
    # No, just nested loops with early break.
    
    for a_s in ap_starts:
        for a_d in ap_steps:
            if a_s + 2*a_d > 128: continue
            A = (a_s, a_s + a_d, a_s + 2*a_d)
            
            for b_s in ap_starts:
                for b_d in ap_steps:
                    if b_s + 2*b_d > 128: continue
                    
                    # Check time
                    if time.time() > deadline:
                        return [fallback_A, fallback_B]
                        
                    B = (b_s, b_s + b_d, b_s + 2*b_d)
                    
                    # Compute 9 sums
                    s1 = A[0] + B[0]
                    if not is_prime[s1]: continue
                    
                    s2 = A[0] + B[1]
                    if not is_prime[s2]: continue
                    
                    s3 = A[0] + B[2]
                    if not is_prime[s3]: continue
                    
                    s4 = A[1] + B[0]
                    if not is_prime[s4]: continue
                    
                    s5 = A[1] + B[1]
                    if not is_prime[s5]: continue
                    
                    s6 = A[1] + B[2]
                    if not is_prime[s6]: continue
                    
                    s7 = A[2] + B[0]
                    if not is_prime[s7]: continue
                    
                    s8 = A[2] + B[1]
                    if not is_prime[s8]: continue
                    
                    s9 = A[2] + B[2]
                    if not is_prime[s9]: continue
                    
                    # Check distinctness
                    # If all are prime, are they distinct?
                    # s_i = s_j implies A[x]+B[y] = A[x']+B[y']
                    # If A, B are APs with distinct steps, they are distinct.
                    # But we should verify just in case.
                    sums = {s1, s2, s3, s4, s5, s6, s7, s8, s9}
                    if len(sums) == 9:
                        return [list(A), list(B)]
                        
    return [fallback_A, fallback_B]
