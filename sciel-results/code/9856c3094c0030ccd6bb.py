import time
import bisect
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1), 1)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 30
    L = _lcm_range(n)
    weights = [L // (i + 1) for i in range(n)]
    
    n1 = 15
    n2 = 15
    idx1 = list(range(n1))
    idx2 = list(range(n1, n))
    w1 = [weights[i] for i in idx1]
    w2 = [weights[i] for i in idx2]
    
    # Generate sums for part 1
    sums1 = [0]
    idxs1 = [0]
    
    for k in range(n1):
        w = w1[k]
        new_sums = []
        new_idxs = []
        for i in range(len(sums1)):
            s = sums1[i]
            idx = idxs1[i]
            # 0
            new_sums.append(s)
            new_idxs.append(idx * 3)
            # 1
            new_sums.append(s + w)
            new_idxs.append(idx * 3 + 1)
            # -1
            new_sums.append(s - w)
            new_idxs.append(idx * 3 + 2)
        sums1 = new_sums
        idxs1 = new_idxs
        
    # Sort part 1 by sum
    # Create order of indices
    order1 = sorted(range(len(sums1)), key=sums1.__getitem__)
    sorted_sums1 = [sums1[i] for i in order1]
    sorted_idxs1 = [idxs1[i] for i in order1]
    
    # Free memory?
    # del sums1, idxs1, order1
    
    # Generate sums for part 2
    sums2 = [0]
    idxs2 = [0]
    
    for k in range(n2):
        w = w2[k]
        new_sums = []
        new_idxs = []
        for i in range(len(sums2)):
            s = sums2[i]
            idx = idxs2[i]
            new_sums.append(s)
            new_idxs.append(idx * 3)
            new_sums.append(s + w)
            new_idxs.append(idx * 3 + 1)
            new_sums.append(s - w)
            new_idxs.append(idx * 3 + 2)
        sums2 = new_sums
        idxs2 = new_idxs
        
    # Search
    best_val = None
    best_idx1 = None
    best_idx2 = None
    
    for i in range(len(sums2)):
        if time.time() > deadline - 5:
            break
        s2 = sums2[i]
        idx2 = idxs2[i]
        target = -s2
        pos = bisect.bisect_left(sorted_sums1, target)
        
        # Check neighbors
        for p in [pos - 1, pos, pos + 1]:
            if 0 <= p < len(sorted_sums1):
                s1 = sorted_sums1[p]
                total = s1 + s2
                if total == 0:
                    continue
                if best_val is None or abs(total) < abs(best_val):
                    best_val = total
                    best_idx1 = sorted_idxs1[p]
                    best_idx2 = idx2
                    
    if best_val is None:
        coeffs = [0] * 28 + [1, -1]
        return coeffs
        
    # Decode
    def decode(idx, n_digits):
        coeffs = [0] * n_digits
        for j in range(n_digits):
            d = idx % 3
            if d == 0: coeffs[j] = 0
            elif d == 1: coeffs[j] = 1
            else: coeffs[j] = -1
            idx //= 3
        return coeffs
        
    c1 = decode(best_idx1, n1)
    c2 = decode(best_idx2, n2)
    
    coeffs = [0] * n
    for i, c in zip(idx1, c1):
        coeffs[i] = c
    for i, c in zip(idx2, c2):
        coeffs[i] = c
        
    return coeffs
