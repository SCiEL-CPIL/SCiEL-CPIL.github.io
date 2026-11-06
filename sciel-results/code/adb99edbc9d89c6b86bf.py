import time
import math

def construct():
        start_time = time.time()
        deadline = start_time + 140  # 10s safety margin
        
        # Determine L
        # We want to maximize N, which correlates with L.
        # Let's try L = 3 * 10^6
        L = 3 * 10**6
        
        # 1. Generate powerful numbers <= L
        powerful = set()
        # y is square-free
        # x^2 y^3 <= L
        # y <= L^(1/3)
        max_y = int(L ** (1/3)) + 1
        for y in range(1, max_y + 1):
            # Check if y is square-free
            # Simple check: no p^2 divides y
            # Since y is small, we can just check primes up to sqrt(y)
            # Or precompute square-free
            # For speed, simple check is fine for y up to 1400
            is_sq_free = True
            temp_y = y
            d = 2
            while d * d <= temp_y:
                if temp_y % (d * d) == 0:
                    is_sq_free = False
                    break
                d += 1
            if not is_sq_free:
                continue
                
            y_cubed = y ** 3
            if y_cubed > L:
                break
            max_x = int(math.isqrt(L // y_cubed))
            for x in range(1, max_x + 1):
                val = x * x * y_cubed
                powerful.add(val)
        
        P = sorted(powerful)
        K = len(P)
        
        # Check time
        if time.time() > deadline:
             # Fallback to small L if generation was slow (unlikely)
             pass

        # 2. Count sums
        # Max sum is 2*L
        # counts array size 2*L + 1
        # 6 * 10^6 size list is fine.
        counts = [0] * (2 * L + 1)
        
        # Optimize inner loop
        # Using local variables
        P_list = P
        K_val = K
        limit_sum = 2 * L
        counts_list = counts
        
        for i in range(K_val):
            pi = P_list[i]
            # j starts from i
            # We can find the max j using bisect
            # max_j = bisect_right(P_list, limit_sum - pi) - 1
            # But bisect is log(K). Doing it K times is K log K.
            # The inner loop runs K/2 times on average.
            # So bisect is cheaper than the loop iterations?
            # No, bisect is ~15 ops. Loop is ~1 op per iteration.
            # Loop is much faster if we just break.
            
            for j in range(i, K_val):
                s = pi + P_list[j]
                if s > limit_sum:
                    break
                counts_list[s] += 1
                
                # Time check inside loop? Too slow.
                # We rely on L being small enough.
        
        # 3. Find max
        max_count = 0
        best_N = 0
        # Iterate counts
        # counts is a list, max() is C-speed, fast.
        max_count = max(counts_list)
        best_N = counts_list.index(max_count)
        
        # 4. Retrieve pairs
        pairs = []
        for i in range(K_val):
            a = P_list[i]
            b = best_N - a
            if b < a:
                continue
            if b > limit_sum:
                break
            # Check if b is in P
            # Since P is sorted, we can use binary search or a set
            # Set lookup is O(1)
            if b in powerful:
                pairs.append([a, b])
        
        return {'N': best_N, 'pairs': pairs}
