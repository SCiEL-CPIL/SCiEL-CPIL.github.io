from fractions import Fraction
import math
import itertools
import bisect

def construct():
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    D = 1
    for i in range(1, 31):
        D = lcm(D, i)
        
    w = [D // (i + 1) for i in range(30)]
    w1 = w[:15]
    w2 = w[15:]
    
    def generate_sums_dict(weights):
        d = {}
        n = len(weights)
        for tup in itertools.product([-1, 0, 1], repeat=n):
            s = 0
            for i in range(n):
                c = tup[i]
                if c != 0:
                    s += c * weights[i]
            if s not in d:
                d[s] = tup
        return d

    # Generate and store
    dict1 = generate_sums_dict(w1)
    dict2 = generate_sums_dict(w2)
    
    # We need a sorted list of keys for binary search
    keys2 = sorted(dict2.keys())
    
    best_abs = float('inf')
    best_s1 = None
    best_s2 = None
    
    for s1 in dict1.keys():
        target = -s1
        idx = bisect.bisect_left(keys2, target)
        
        candidates = []
        if idx < len(keys2):
            candidates.append(keys2[idx])
        if idx > 0:
            candidates.append(keys2[idx-1])
            
        for s2 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_abs:
                best_abs = abs_total
                best_s1 = s1
                best_s2 = s2
                
    if best_s1 is None:
        return [0]*30
        
    c1 = list(dict1[best_s1])
    c2 = list(dict2[best_s2])
    
    return c1 + c2
