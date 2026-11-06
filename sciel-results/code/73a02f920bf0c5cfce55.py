import time
import math
import heapq
from math import factorial, isqrt

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin under 90s kill
    
    N = factorial(12)  # 479001600
    
    # Generate all divisors of N that are > 12
    divs = []
    i = 1
    while i * i <= N:
        if N % i == 0:
            if i > 12:
                divs.append(i)
            other = N // i
            if other != i and other > 12:
                divs.append(other)
        i += 1
    
    divs.sort()
    divs_set = set(divs)
    
    # Cache for factor pairs of a number: pairs (a, b) with a < b, a*b = n, a > 12, b > 12
    pairs_cache = {}
    def get_pairs(n):
        if n in pairs_cache:
            return pairs_cache[n]
        res = []
        r = isqrt(n)
        for a in range(1, r + 1):
            if n % a == 0:
                b = n // a
                if a > 12 and b > 12 and a != b:
                    if a < b:
                        res.append((a, b))
                    else:
                        res.append((b, a))
        pairs_cache[n] = res
        return res
    
    best_max = N
    best_set = frozenset([N])
    
    # PQ: (max_factor, count, frozenset)
    visited = {frozenset([N])}
    pq = [(N, 1, frozenset([N]))]
    
    while pq:
        if time.time() - t0 > DEADLINE:
            break
        
        m, c, fs = heapq.heappop(pq)
        
        if m > best_max:
            break
        
        if m < best_max:
            best_max = m
            best_set = fs
        
        # Can't improve further if next PQ entry has max >= best_max
        if pq and pq[0][0] >= best_max:
            break
        
        if c >= 12:
            continue
        
        # Split the largest element
        largest = max(fs)
        
        for (a, b) in get_pairs(largest):
            # b >= largest/2, and since a < b, b > sqrt(largest)
            # New max would be max(second_largest, a, b)
            # But b < largest (since a > 1), so b < m
            # Check distinctness
            newfs = set(fs)
            newfs.remove(largest)
            if a in newfs or b in newfs:
                continue
            newfs.add(a)
            newfs.add(b)
            nset = frozenset(newfs)
            
            if nset in visited:
                continue
            
            newm = max(nset)
            if newm >= best_max:
                continue
            
            visited.add(nset)
            heapq.heappush(pq, (newm, c + 1, nset))
        
        # Periodic time check inside loop
        if time.time() - t0 > DEADLINE:
            break
    
    return sorted(best_set)
