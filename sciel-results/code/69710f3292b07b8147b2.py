import time

def construct():
    N = 123457
    
    # Deadline guard
    start_time = time.time()
    deadline = 60.0  # 60 seconds
    
    # Precompute Largest Prime Factor (LPF) for all numbers up to N
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Sieve-like approach to fill lpf
    # Iterate potential prime factors p from 2 to N
    for p in range(2, N + 1):
        # If lpf[p] is 0, p is prime (no smaller prime factor found)
        if lpf[p] == 0:
            # Mark p as the largest prime factor for itself and all its multiples
            # Since we iterate p in increasing order, if p divides j, 
            # and p is the current prime, it will overwrite any smaller prime factor.
            # Thus, the last assignment for any j will be by its largest prime factor.
            for j in range(p, N + 1, p):
                lpf[j] = p
                # Optional: early break if time is tight, but N is small
                if time.time() > deadline - 5.0:
                    # Fallback if sieve takes too long (unlikely for N=123457)
                    # Just return a valid pair immediately
                    return [1, N - 1]
    
    # Find the pair (a, b) minimizing max(P(a), P(b))
    min_max_p = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        p_a = lpf[a]
        p_b = lpf[b]
        current_max = max(p_a, p_b)
        
        if current_max < min_max_p:
            min_max_p = current_max
            best_a = a
            best_b = b
            
            # If we found 1, we can't do better than 1 (since P(1)=1 and P(x)>=1)
            # But P(x)=1 only if x=1. So if max is 1, one is 1 and the other is 1? 
            # No, a+b=N. If N>2, we can't have both 1.
            # The minimum possible value for max(P(a), P(b)) is at least 2 (since one number is > 1).
            # Actually, if N is even, a=N/2, b=N/2.
            # If N is odd, one is even, one is odd.
            # We can stop if min_max_p == 2? 
            # P(a)=2 implies a is a power of 2.
            # P(b)=2 implies b is a power of 2.
            # Sum of two powers of 2 is a power of 2 only if they are equal? No.
            # If max is 2, both must be powers of 2.
            # 2^x + 2^y = 123457 (odd). Impossible since sum of two even numbers is even.
            # So one must be odd. P(odd) >= 3 (unless odd=1, P(1)=1).
            # So min_max_p cannot be 2.
            # Minimum possible is 3?
            # If a=1, P(a)=1. b=123456. P(123456) is likely large.
            # If we find a pair with max=3, it means both are of form 3^k or 1.
            # Sum of 3^k and 3^m?
            # We can just let the loop finish, it's fast.
            
        # Check time limit in loop
        if time.time() > deadline - 2.0:
            break
            
    return [best_a, best_b]
