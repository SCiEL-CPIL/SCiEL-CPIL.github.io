import time
from math import isqrt

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    # n = u^2 * v^3
    # v^3 <= limit => v <= limit^(1/3)
    max_v = int(limit ** (1/3)) + 2
    for v in range(1, max_v + 1):
        v3 = v ** 3
        if v3 > limit:
            break
        # u^2 <= limit / v^3 => u <= sqrt(limit / v^3)
        max_u = isqrt(limit // v3)
        for u in range(1, max_u + 1):
            n = (u * u) * v3
            if n <= limit:
                powerful.add(n)
    return powerful

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Start with a moderate limit
    # 10^10 gives ~10^4 numbers, ~5*10^7 pair checks
    # 10^11 gives ~3*10^4 numbers, ~4.5*10^8 pair checks
    # Let's try 10^10 first
    
    limits_to_try = [10**10, 10**11, 10**12]
    
    best_N = 2
    best_pairs = []
    best_count = 0
    
    for limit in limits_to_try:
        if time.time() > deadline - 30:  # Leave 30s for counting
            break
            
        P = _generate_powerful_numbers(limit)
        P_list = sorted(P)
        K = len(P_list)
        
        # Estimate time for counting
        # ~ K^2 / 2 operations
        # Python ~ 10^7 ops/sec
        est_time = (K * K) / 2 / 10e6
        if est_time > 120 - (time.time() - start_time):
            continue # Skip if too slow
            
        counts = {}
        # Optimization: only a <= b
        for i in range(K):
            a = P_list[i]
            # b must be >= a
            # And a+b <= 2*limit (always true for our generated set)
            for j in range(i, K):
                b = P_list[j]
                s = a + b
                counts[s] = counts.get(s, 0) + 1
                
        # Find max
        if counts:
            max_count = max(counts.values())
            if max_count > best_count:
                best_count = max_count
                # Find N
                for N, c in counts.items():
                    if c == max_count:
                        best_N = N
                        # Reconstruct pairs
                        # This is expensive if we do it naively, but we only do it for the best
                        best_pairs = []
                        P_set = set(P_list)
                        for a in P_list:
                            if a > best_N // 2:
                                break
                            b = best_N - a
                            if b in P_set and b >= a:
                                best_pairs.append([a, b])
                        break
                        
        # If we used a lot of time, break
        if time.time() > deadline - 10:
            break

    return {'N': best_N, 'pairs': best_pairs}
