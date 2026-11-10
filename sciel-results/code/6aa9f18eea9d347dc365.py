import time
import math

def construct():
    start_time = time.time()
    deadline = 55  # Safety margin before 60s limit
    
    limit = 1000000
    max_val = limit + 1
    
    # Sieve to find largest prime factor
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (max_val + 1)
    
    # Standard sieve logic modified to store largest prime factor
    # Since we iterate p in increasing order, lpf[j] will be updated by smaller primes first,
    # and by larger primes later. Thus, the final value is the largest prime factor.
    
    # Optimization: Only iterate p up to sqrt(max_val) for the inner loop updates?
    # No, because if p > sqrt(max_val) and p is prime, its multiples are p, 2p, 3p...
    # But wait, if p is prime, lpf[p] is 0 initially.
    # If we only go up to sqrt(max_val), we miss primes p where sqrt(max_val) < p <= max_val.
    # For such p, the only multiple in range is p itself (since 2p > max_val).
    # So lpf[p] should be p.
    # If we don't update it, lpf[p] remains 0.
    # So we must handle primes > sqrt(max_val) or run the full loop.
    # Running the full loop is O(N log log N) which is fast enough for N=1e6.
    
    lpf_local = lpf
    for p in range(2, max_val + 1):
        if lpf_local[p] == 0:
            # p is prime
            for j in range(p, max_val + 1, p):
                lpf_local[j] = p
        
        # Time check inside sieve? Unnecessary for N=1e6, takes <0.1s usually.
        # But let's be safe.
        if time.time() - start_time > deadline:
            # Fallback: return a reasonable guess
            # We can't finish the sieve, so we can't compute LPF reliably.
            # Return 10000 as a safe default? Or try to compute LPF for a few numbers?
            # The task says "independently computed by exact trial division" in the description,
            # but the program just returns n. The evaluator computes the score.
            # So we just need to return a valid n in [10000, 1000000].
            return 10000

    # Scan for the minimum max(LPF(n), LPF(n+1))
    best_n = 10000
    min_score = float('inf')
    
    # We can optimize the scan by checking time periodically
    # The scan is O(N) = 1e6 iterations, which is very fast (<0.1s).
    
    for n in range(10000, limit + 1):
        score = max(lpf_local[n], lpf_local[n+1])
        if score < min_score:
            min_score = score
            best_n = n
            
        # Check time every 10000 iterations to avoid overhead
        if (n - 10000) % 10000 == 0 and time.time() - start_time > deadline:
            break

    return best_n
