import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        for v in range(1, max_v + 1):
            if not is_square_free(v):
                continue
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    # Generate powerful numbers up to 10^9
    # This yields ~30,000 numbers.
    # Pair counting ~4.5 * 10^8 operations.
    # Estimated time: 30-50 seconds.
    
    L = 10**9
    P = generate_powerful(L)
    P_set = set(P)
    len_P = len(P)
    
    counts = {}
    best_N = 0
    best_count = 0
    
    # Local variables for speed
    P_list = P
    counts_get = counts.get
    
    # We iterate i from 0 to len_P - 1
    # For each i, we iterate j from i to len_P - 1
    # Since P is sorted, sums increase as j increases.
    # We can break the inner loop if the sum exceeds a certain threshold?
    # No, we don't know the threshold.
    # But we can break the OUTER loop if time is low.
    
    for i in range(len_P):
        ai = P_list[i]
        # Optimization: if ai is very large, the sums will be large.
        # If we have already found a good best_count, and the minimum possible sum
        # for this i is ai + ai, and we suspect larger sums have lower counts...
        # Actually, let's just run it.
        
        for j in range(i, len_P):
            s = ai + P_list[j]
            c = counts_get(s, 0) + 1
            counts[s] = c
            if c > best_count:
                best_count = c
                best_N = s
                
        if time.time() > deadline:
            break

    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Collect pairs for best_N
    pairs = []
    for a in P_list:
        if a > best_N:
            break
        b = best_N - a
        if b < a:
            continue
        if b in P_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
