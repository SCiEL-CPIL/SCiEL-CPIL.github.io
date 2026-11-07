import time
import numpy as np
import bisect

def construct():
    t0 = time.time()
    LIMIT = 60.0
    SAFETY = 0.9
    hard_stop = t0 + LIMIT * SAFETY
    
    def time_left():
        return hard_stop - time.time()

    N = 10000000
    # Fallback: sigma(11)=12, sigma(6)=12. 11/6 = 1.8333. |1.8333 - 1.5| = 0.3333.
    # Another one: sigma(7)=8, sigma(3)=4 (no).
    # sigma(4)=7, sigma(6)=12.
    # Let's find a good fallback.
    # sigma(1)=1, sigma(2)=3, sigma(3)=4, sigma(4)=7, sigma(5)=6, sigma(6)=12, sigma(7)=8, sigma(8)=15, sigma(9)=13, sigma(10)=18, sigma(11)=12.
    # Pairs with same sigma:
    # sigma(6)=12, sigma(11)=12. Pair (6, 11) or (11, 6).
    # 11/6 = 1.833, err = 0.333.
    # 6/11 = 0.545, err = 0.954.
    # Better: sigma(10)=18, sigma(15)=24 (no).
    # sigma(12)=28, sigma(18)=39.
    # sigma(14)=24, sigma(15)=24. Pair (14, 15).
    # 15/14 = 1.07, err = 0.42.
    # 14/15 = 0.93, err = 0.56.
    # sigma(16)=31.
    # sigma(20)=42, sigma(21)=32.
    # sigma(22)=36, sigma(24)=60.
    # sigma(24)=60, sigma(27)=40.
    # sigma(28)=56, sigma(30)=72.
    # sigma(30)=72, sigma(42)=96.
    # Let's use (11, 6) as fallback.
    
    fallback = [11, 6]
    
    try:
        # 1. Compute Sigma using vectorized sieve
        # sigma[n] = sum of divisors of n
        # sigma = np.zeros(N + 1, dtype=np.int64)
        # for d in range(1, N + 1):
        #     sigma[d::d] += d
        
        # This loop is O(N log N). N=10^7.
        # 10^7 * ln(10^7) ~ 1.6 * 10^8 additions.
        # NumPy can do ~10^8-10^9 ops/sec.
        # Should take ~1-2 seconds.
        
        sigma = np.zeros(N + 1, dtype=np.int64)
        # Use a list of primes? No, just iterate d.
        # Optimization: Only iterate d if it contributes? No, every d contributes to multiples.
        
        # We can try to speed this up by only iterating up to N//2? No, d=N contributes to N.
        # But d > N/2 only contributes to itself.
        # sigma[d] += d for d in N//2 + 1 .. N.
        # We can do:
        sigma[1:] += np.arange(1, N + 1, dtype=np.int64) # This is just sigma[d] = d for now? No.
        # Wait, the loop `sigma[d::d] += d` adds d to sigma[d], sigma[2d], etc.
        # So for d, it adds d to sigma[d].
        # So after all loops, sigma[n] is sum of all d such that d|n.
        
        # Let's just run the loop.
        for d in range(1, N + 1):
            sigma[d::d] += d
            
            # Time check
            if (d & 0xFFFFF) == 0 and time_left() < 10.0:
                return fallback

        if time_left() < 5.0:
            return fallback

        # 2. Group by Sigma
        # We want indices 1..N
        idx = np.arange(1, N + 1, dtype=np.int32)
        vals = sigma[1:]
        
        # Sort indices by sigma value
        order = np.argsort(vals, kind='stable')
        svals = vals[order]
        
        # Find group boundaries
        M = order.size
        change = np.empty(M, dtype=bool)
        change[0] = True
        change[1:] = svals[1:] != svals[:-1]
        starts = np.nonzero(change)[0]
        ends = np.append(starts[1:], M)
        
        best_val = float('inf')
        best_pair = None
        # max_b is not strictly needed if we check a,b <= N, but a,b are from 1..N.
        # We want a/b approx 1.5. So a approx 1.5 b.
        # If b is large, a must be <= N. So 1.5 b <= N => b <= 2N/3.
        # If b is small, a can be up to N.
        
        # Iterate groups
        for gi in range(starts.size):
            if time_left() < 2.0:
                break
                
            s0 = int(starts[gi])
            e0 = int(ends[gi])
            length = e0 - s0
            if length < 2:
                continue
                
            # Extract and sort the group elements (they are already in 'order' which is sorted by sigma, 
            # but we need them sorted by value for bisect)
            # idx[order[s0:e0]] are the numbers in this group.
            gn = np.sort(idx[order[s0:e0]])
            gn_list = gn.tolist()
            L = len(gn_list)
            
            # For each b in group, find best a
            # We want a/b approx 1.5 => a approx 1.5 b
            # Error = |a/b - 1.5| = |2a - 3b| / (2b)
            
            # To optimize, we can iterate b and use bisect.
            # However, if L is large, this is O(L log L).
            # Total sum of L is N.
            
            for bi in range(L):
                b = gn_list[bi]
                # If b is too large, 1.5b > N, so a must be close to N.
                # But a must be in gn_list.
                
                target = 1.5 * b
                j = bisect.bisect_left(gn_list, target)
                
                # Check j
                if j < L:
                    a = gn_list[j]
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2.0 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)
                
                # Check j-1
                if j > 0:
                    a = gn_list[j-1]
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2.0 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)

        if best_pair is None:
            return fallback
            
        a, b = int(best_pair[0]), int(best_pair[1])
        if not (1 <= a <= N and 1 <= b <= N):
            return fallback
        if 2 * a == 3 * b:
            return fallback
            
        return [a, b]

    except Exception:
        return [11, 6]
