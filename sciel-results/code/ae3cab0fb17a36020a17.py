"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find two positive integers a, b with a+b=123457 minimizing max(P(a), P(b)),
    where P(n) is the largest prime factor of n (P(1)=1).
    
    Strategy:
    1. Precompute the largest prime factor (LPF) for all numbers up to N=123457
       using a sieve-like approach.
    2. Iterate through all possible splits a from 1 to N//2.
    3. For each a, compute b = N - a.
    4. Compute cost = max(LPF[a], LPF[b]).
    5. Track the minimum cost and the corresponding (a, b).
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, 5s margin
    
    N = 123457
    
    # Precompute largest prime factor for all numbers up to N
    # lpf[i] will store the largest prime factor of i
    # Initialize with 0
    lpf = [0] * (N + 1)
    if N >= 1:
        lpf[1] = 1
    
    # Sieve for largest prime factor
    # For each i from 2 to N, if lpf[i] is 0, i is prime.
    # Mark all multiples of i with i. Since we iterate i in increasing order,
    # the last prime that divides a number will be its largest prime factor.
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_cost = float('inf')
    best_a = 1
    best_b = N - 1
    
    # We only need to check a from 1 to N//2, since (a,b) and (b,a) are symmetric
    # regarding the cost max(P(a), P(b)).
    half = N // 2
    for a in range(1, half + 1):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        cost = pa if pa > pb else pb
        
        if cost < best_cost:
            best_cost = cost
            best_a = a
            best_b = b
            
        # Early termination check: 
        # Since N is odd, one of a,b is even and the other is odd.
        # P(even) >= 2, P(odd) >= 3 (unless odd is 1, P(1)=1).
        # If we find a pair where max(P(a), P(b)) <= 3, it's the theoretical minimum.
        if best_cost <= 3:
            break
        
        # Check time periodically to ensure we don't exceed the deadline
        if a % 10000 == 0:
            if time.time() > deadline:
                break
    
    # Return the pair
    return [best_a, best_b]
# EVOLVE-BLOCK-END
