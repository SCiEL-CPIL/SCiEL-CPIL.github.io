import time
import sys
from collections import defaultdict
import bisect

def construct():
        start_time = time.time()
        deadline = start_time + 55.0 # 5s margin
        
        N = 10_000_000
        
        # Sieve
        sigma = [1] * (N + 1)
        for i in range(2, N + 1):
            if time.time() > deadline:
                # Fallback to a known good pair or best found so far?
                # If we haven't finished sigma, we can't group.
                # We should check time before heavy loops.
                pass
            if sigma[i] == 1:
                for j in range(i, N + 1, i):
                    p_power = 1
                    sum_p = 1
                    temp = j
                    while temp % i == 0:
                        temp //= i
                        p_power *= i
                        sum_p += p_power
                    sigma[j] *= sum_p
                    if time.time() > deadline:
                         # Break out?
                         # This is tricky.
                         pass
        
        # If we hit deadline during sieve, we might not have valid sigma.
        # However, 10^7 sieve is ~2s. Grouping ~1s. Search ~1s.
        # Total ~4s. Deadline 55s is very safe.
        
        groups = defaultdict(list)
        for n in range(1, N + 1):
            groups[sigma[n]].append(n)
            
        best_err = float('inf')
        best_a = 1
        best_b = 1
        
        for S, lst in groups.items():
            if time.time() > deadline:
                break
            if len(lst) < 2:
                continue
            lst.sort()
            for b in lst:
                target = 1.5 * b
                idx = bisect.bisect_left(lst, target)
                if idx < len(lst):
                    a = lst[idx]
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                if idx > 0:
                    a = lst[idx - 1]
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                        
        return [best_a, best_b]
