"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial small scan to find promising M values
    # We start with a small N to quickly find collisions
    N = 200000
    sigma = list(range(N + 1))
    
    # Sieve for sigma
    for i in range(2, N + 1):
        if sigma[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                sigma[j] = sigma[j] // i * (i + 1)
                
    # Map M -> list of k
    m_to_k = defaultdict(list)
    for k in range(1, N + 1):
        m = k * sigma[k]
        if m <= 10**18:
            m_to_k[m].append(k)
            
    # Find the best M from the initial scan
    best_m = None
    best_count = 0
    for m, ks in m_to_k.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = m
            
    if best_m is None:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Adaptive extension: 
    # If we found collisions, we try to extend the range to find more k for the best_m.
    # We double the range until time runs out or we stop finding new solutions for best_m.
    
    # To extend, we need to compute sigma for new k values.
    # Recomputing the sieve from scratch for larger N is O(N log log N).
    # We can just re-run the sieve with a larger N.
    
    current_n = N
    best_ks = m_to_k[best_m][:]
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            break
            
        # Try to double the range
        next_n = current_n * 2
        # Estimate time: Sieve is roughly linear in N * log log N.
        # If current_n took T, next_n might take ~2 * (log log 2N / log log N) * T.
        # Let's just try it, but with a cap on next_n to avoid memory issues.
        if next_n > 5000000: # 5M is a reasonable cap for memory/time in 120s
             if elapsed > deadline - 5.0:
                 break
             # If we are close to deadline, don't start a huge new sieve
             if next_n > 2000000:
                 break
                 
        # Check if we have time for a larger sieve
        # Heuristic: if current_n took X seconds, next_n will take approx X * (next_n/current_n) * factor
        # We don't have exact time for current sieve, but we can guess.
        # Let's just attempt it.
        
        new_sigma = list(range(next_n + 1))
        for i in range(2, next_n + 1):
            if new_sigma[i] == i:
                for j in range(i, next_n + 1, i):
                    new_sigma[j] = new_sigma[j] // i * (i + 1)
                    
        # Update m_to_k for new k values
        # Note: We only care about best_m for the purpose of extending the list
        # But other M might become better? Unlikely if we started with a good seed.
        # However, to be safe, we should track the best.
        # But re-scanning all m_to_k is expensive. 
        # Strategy: Focus on best_m. If we find more, great. 
        # Also check if any other M surpasses best_count?
        # To keep it simple and fast, we just extend best_m's list.
        
        found_new = False
        for k in range(current_n + 1, next_n + 1):
            m = k * new_sigma[k]
            if m == best_m:
                best_ks.append(k)
                found_new = True
            elif m in m_to_k:
                m_to_k[m].append(k)
                # Check if this M is now better
                if len(m_to_k[m]) > len(best_ks):
                    # Switch best_m? This gets complicated. 
                    # Let's stick to the initial best_m unless we explicitly track all.
                    pass
            
        if not found_new:
            # No new solutions for best_m in this range. 
            # It's possible there are none at all, or they are much further out.
            # Given the distribution, if we didn't find any in [N, 2N], 
            # it's less likely to find many in [2N, 4N] without a specific structure.
            # But let's try one more step if we have time.
            if elapsed > deadline - 10.0:
                break
            # If we found 0 new, maybe stop? 
            # Actually, solutions can be sparse. 
            # Let's continue doubling unless we hit the cap.
            
        current_n = next_n
        
    # Final verification and formatting
    # Ensure k_i are distinct and <= sqrt(M)
    # k_i * sigma(k_i) = M implies k_i <= M / sigma(k_i) <= M / 1 = M.
    # But we need k_i <= sqrt(M).
    # If k > sqrt(M), then sigma(k) >= k+1 > k, so k*sigma(k) > k^2 > M.
    # So any k satisfying k*sigma(k)=M MUST be <= sqrt(M).
    # So the constraint k_i <= sqrt(M) is automatically satisfied for valid solutions.
    
    # Just to be safe, filter
    limit = math.isqrt(best_m)
    valid_ks = [k for k in best_ks if k <= limit]
    
    # Sort for consistency
    valid_ks.sort()
    
    return {'M': best_m, 'k': valid_ks}
# EVOLVE-BLOCK-END
