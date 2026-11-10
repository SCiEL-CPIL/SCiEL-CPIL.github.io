import time
import math

def construct():
    N = 123457
    deadline = time.time() + 80  # 80s deadline to stay well under 90s limit
    
    # Precompute largest prime factor (LPF) for all numbers up to N using sieve
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for p in range(2, N + 1):
        if lpf[p] == 0:
            for j in range(p, N + 1, p):
                lpf[j] = p
    
    # Strategy: Find the minimum M such that there exist M-smooth a, b with a+b=N.
    # M-smooth means all prime factors <= M.
    # We iterate M in increasing order (over distinct LPF values).
    # For each M, we check if there's a pair (a, b) with a+b=N, lpf[a]<=M, lpf[b]<=M.
    
    # Collect distinct LPF values that appear (these are the candidate M values)
    distinct_lpf = sorted(set(lpf[1:N+1]))
    
    best_score = float('inf')
    best_a = 1
    best_b = N - 1
    
    # For each candidate M (in increasing order), check if there exist M-smooth a and b with a+b=N
    # Key insight: once we find the smallest M where a valid pair exists, that's optimal.
    # But we also need to track the best pair for the current best M in case we time out.
    
    # Efficient approach: For each M, build the set of M-smooth numbers incrementally.
    # Actually, let's use a different approach:
    # Sort numbers by their LPF. Process in increasing LPF order.
    # Maintain a set of "available" numbers (those with LPF <= current M).
    # For each new number added, check if N - number is also in the set.
    
    # Let's sort indices by their LPF value
    # Create list of (lpf_value, number) pairs
    pairs = [(lpf[i], i) for i in range(1, N)]
    pairs.sort()
    
    available = set()
    found_optimal = False
    
    for lpf_val, num in pairs:
        if time.time() > deadline:
            break
        
        # Before adding 'num', check if N-num is already in available
        # If so, we found a pair where both have LPF <= lpf_val
        # (since all previously added numbers have LPF <= lpf_val, and num has LPF = lpf_val)
        complement = N - num
        if complement >= 1 and complement < N and complement in available:
            # We found a valid pair with max(P(a), P(b)) <= lpf_val
            # Since we're processing in increasing LPF order, this is the minimum possible
            return [num, complement]
        
        # Also check: is num in available? No, we haven't added it yet.
        # But what if complement == num? Then we'd need 2*num = N, which requires N even.
        # N = 123457 is odd, so num != complement always.
        
        # Add num to available
        available.add(num)
    
    # If we didn't find a pair via the incremental approach (shouldn't happen for valid N),
    # fall back to direct scan
    best_score = float('inf')
    best_a = 1
    best_b = N - 1
    
    mid = N // 2
    for a in range(1, mid + 1):
        if time.time() > deadline:
            break
        b = N - a
        score = max(lpf[a], lpf[b])
        if score < best_score:
            best_score = score
            best_a = a
            best_b = b
    
    return [best_a, best_b]
