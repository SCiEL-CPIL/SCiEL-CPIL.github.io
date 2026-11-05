import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 1000001
    # Precompute largest prime factor for all numbers up to N
    # Using a list is faster in Python for this size than numpy array for the sieve update loop
    # because numpy scalar access is slower than list access in tight Python loops?
    # Actually, pure Python sieve is fine.
    
    lpf = [0] * (N + 1)
    
    # Optimization: Handle 2 separately
    for j in range(2, N + 1, 2):
        lpf[j] = 2
        
    # Sieve for odd primes
    for i in range(3, int(np.sqrt(N)) + 1, 2):
        if lpf[i] == 0:
            # i is prime
            # Mark multiples
            # Step by 2*i to only hit odd multiples? 
            # Wait, lpf[j] = i needs to be done for ALL multiples to get the LARGEST.
            # If we skip even multiples, lpf[even] might not be updated by larger odd primes?
            # Example: 6. lpf[6] set to 2. 
            # If i=3, 6 is a multiple. lpf[6] should become 3? 
            # Yes, 3 > 2.
            # So we must update ALL multiples.
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # Wait, the loop above only goes to sqrt(N).
    # If a number j has a prime factor p > sqrt(N), then j must have another factor q < sqrt(N).
    # But we want the LARGEST prime factor.
    # If j is prime, j > sqrt(N), lpf[j] is currently 0 (if not marked by smaller factors).
    # If j is composite and its largest prime factor is > sqrt(N), then the other factor is < sqrt(N).
    # The sieve above only updates lpf[j] for primes i <= sqrt(N).
    # So for a prime p > sqrt(N), lpf[p] remains 0.
    # For a composite j = p * q where p > sqrt(N) and q < sqrt(N), lpf[j] will be set to q (when i=q).
    # But p > q, so lpf[j] should be p.
    # The standard sieve `for i in range(2, N+1)` works because it goes up to N.
    # If I stop at sqrt(N), I miss the case where the number itself is the largest prime factor.
    
    # So I must iterate i up to N, or handle the "remaining" cases.
    # Actually, if lpf[j] is 0 after the sieve up to sqrt(N), it means j is prime.
    # So I can just do:
    for i in range(2, N + 1):
        if lpf[i] == 0:
            lpf[i] = i # It's prime, so its largest prime factor is itself
            
    # But wait, if j = 15. sqrt(15) ~ 3.8.
    # i=3. lpf[3]==0. Mark multiples of 3. lpf[15]=3.
    # i=5. 5 > sqrt(15). Loop stops.
    # lpf[15] is 3. But 5 is also a factor and 5>3.
    # So stopping at sqrt(N) is WRONG for finding the LARGEST prime factor.
    # We MUST iterate i up to N, or use a different method.
    
    # The original code iterates i up to N. This is correct.
    # I will stick to the original logic but clean it up.

    # Re-doing the sieve correctly
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i

    # Scan for best n
    best_n = 10000
    best_val = max(lpf[10000], lpf[100001])
    
    # Use numpy for the scan?
    # lpf is a list. Convert to numpy array?
    # lpf_np = np.array(lpf)
    # vals = np.maximum(lpf_np[10000:N], lpf_np[10001:N+1])
    # min_val = np.min(vals)
    # min_idx = np.argmin(vals) + 10000
    
    # This is much faster and cleaner.
    
    if time.time() > deadline:
        return best_n
        
    lpf_np = np.array(lpf)
    # n ranges from 10000 to 1000000
    # n+1 ranges from 10001 to 1000001
    # We need max(lpf[n], lpf[n+1])
    
    # Slice 1: lpf[10000 : 1000000]  (indices 10000 to 999999) -> length 990000
    # Slice 2: lpf[10001 : 1000001]  (indices 10001 to 1000000) -> length 990000
    # Wait, n goes up to 1000000.
    # So n is in [10000, 1000000].
    # n+1 is in [10001, 1000001].
    
    # lpf_np[10000:1000001] gives indices 10000 to 1000000. (Length 990001)
    # lpf_np[10001:1000002] gives indices 10001 to 1000001. (Length 990001)
    
    # We need to compare element-wise.
    # n_val = lpf_np[10000:1000001]
    # n1_val = lpf_np[10001:1000002]
    
    # Check bounds: N=1000001. Array size N+1 = 1000002. Max index 1000001.
    # Slice 1 ends at 1000000 (exclusive 1000001). So max index 1000000.
    # Slice 2 ends at 1000002 (exclusive). So max index 1000001.
    # This is correct.
    
    n_vals = lpf_np[10000:1000001]
    n1_vals = lpf_np[10001:1000002]
    
    max_vals = np.maximum(n_vals, n1_vals)
    
    min_val = np.min(max_vals)
    min_idx = np.argmin(max_vals)
    
    best_n = 10000 + min_idx
    best_val = min_val
    
    return best_n
