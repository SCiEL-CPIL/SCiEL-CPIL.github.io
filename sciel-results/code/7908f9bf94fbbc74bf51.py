import time
import random
import math
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)
        
    weights = [L // d for d in denominators]
    
    # Fallback heuristic
    def heuristic_fallback():
        # Simple greedy
        indices = []
        curr_sum = Fraction(0)
        # Sort by denominator descending (smaller reciprocal) to pack more?
        # Or ascending?
        # Greedy: pick largest possible?
        # Let's just use the old heuristic logic briefly
        order = list(range(n))
        for i in order:
            ns = curr_sum + Fraction(1, denominators[i])
            if ns < 1:
                indices.append(i)
                curr_sum = ns
        return indices

    # Try Meet in the Middle
    try:
        n1 = n // 2
        n2 = n - n1
        
        w1 = weights[:n1]
        w2 = weights[n1:]
        
        # Generate subsets for half 1
        map1 = {}
        curr_sum = 0
        prev_gray = 0
        size1 = 1 << n1
        for i in range(size1):
            if time.time() > deadline - 5:
                raise TimeoutError("Time limit approaching")
            if i > 0:
                gray = i ^ (i >> 1)
                diff = gray ^ prev_gray
                bit = diff.bit_length() - 1
                if gray & diff:
                    curr_sum += w1[bit]
                else:
                    curr_sum -= w1[bit]
                prev_gray = gray
            
            if curr_sum not in map1:
                map1[curr_sum] = gray

        # Generate subsets for half 2
        map2 = {}
        curr_sum = 0
        prev_gray = 0
        size2 = 1 << n2
        for i in range(size2):
            if time.time() > deadline - 5:
                raise TimeoutError("Time limit approaching")
            if i > 0:
                gray = i ^ (i >> 1)
                diff = gray ^ prev_gray
                bit = diff.bit_length() - 1
                if gray & diff:
                    curr_sum += w2[bit]
                else:
                    curr_sum -= w2[bit]
                prev_gray = gray
            
            if curr_sum not in map2:
                map2[curr_sum] = gray

        sums2 = sorted(map2.keys())
        
        best_total = -1
        best_mask1 = 0
        best_mask2 = 0
        
        for s1, m1 in map1.items():
            if s1 >= L:
                continue
            cap = L - 1 - s1
            pos = bisect.bisect_right(sums2, cap)
            if pos > 0:
                s2 = sums2[pos - 1]
                total = s1 + s2
                if total > best_total:
                    best_total = total
                    best_mask1 = m1
                    best_mask2 = map2[s2]
        
        res = []
        m = best_mask1
        bit = 0
        while m:
            if m & 1:
                res.append(bit)
            m >>= 1
            bit += 1
        
        m = best_mask2
        bit = n1
        while m:
            if m & 1:
                res.append(bit)
            m >>= 1
            bit += 1
            
        if not res:
            res = [0]
        return res

    except (TimeoutError, MemoryError):
        return heuristic_fallback()

if __name__ == "__main__":
    res = construct()
    denominators = list(range(2, 49))
    s = Fraction(0)
    for idx in res:
        s += Fraction(1, denominators[idx])
    print(f"Indices: {res}")
    print(f"Sum: {s}")
    print(f"Deficit: {1 - s}")
