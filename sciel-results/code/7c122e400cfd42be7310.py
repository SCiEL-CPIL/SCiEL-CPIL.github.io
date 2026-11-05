"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import bisect

def construct():
    deadline = time.time() + 85  # 90s limit, keep 5s margin
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    # weights[i] = L / (i+1), so sum(c_i / (i+1)) = sum(c_i * weights[i]) / L
    weights = [L // (i + 1) for i in range(30)]
    
    # Meet-in-the-middle: split into two groups of 15
    n1 = 15
    n2 = 15
    w1 = weights[:n1]
    w2 = weights[n1:n1+n2]
    
    # Generate (sum, code) for group 1
    pairs1 = []
    for code1 in range(3 ** n1):
        s1 = 0
        temp = code1
        for i in range(n1):
            digit = temp % 3
            temp //= 3
            if digit == 0:
                val = -1
            elif digit == 1:
                val = 0
            else:
                val = 1
            s1 += val * w1[i]
        pairs1.append((s1, code1))
    
    pairs1.sort(key=lambda x: x[0])
    sums1_only = [p[0] for p in pairs1]
    
    best_abs = None
    best_code1 = None
    best_code2 = None
    
    for code2 in range(3 ** n2):
        if time.time() > deadline:
            break
        s2 = 0
        temp = code2
        for i in range(n2):
            digit = temp % 3
            temp //= 3
            if digit == 0:
                val = -1
            elif digit == 1:
                val = 0
            else:
                val = 1
            s2 += val * w2[i]
        
        target = -s2
        idx = bisect.bisect_left(sums1_only, target)
        
        # Check neighbors
        for j in range(max(0, idx - 3), min(len(sums1_only), idx + 4)):
            s1 = sums1_only[j]
            total = s1 + s2
            if total == 0:
                continue
            if best_abs is None or abs(total) < best_abs:
                best_abs = abs(total)
                best_code1 = pairs1[j][1]
                best_code2 = code2
    
    # Reconstruct coefficients
    coeffs = [0] * 30
    
    if best_code1 is not None:
        # Decode code1
        temp = best_code1
        for i in range(n1):
            digit = temp % 3
            temp //= 3
            if digit == 0:
                coeffs[i] = -1
            elif digit == 1:
                coeffs[i] = 0
            else:
                coeffs[i] = 1
        
        # Decode code2
        temp = best_code2
        for i in range(n2):
            digit = temp % 3
            temp //= 3
            if digit == 0:
                coeffs[n1 + i] = -1
            elif digit == 1:
                coeffs[n1 + i] = 0
            else:
                coeffs[n1 + i] = 1
    else:
        # Fallback
        coeffs[0] = 1
    
    # Verify
    total = sum(coeffs[i] * weights[i] for i in range(30))
    if total == 0:
        coeffs[0] = 1
    
    return coeffs
# EVOLVE-BLOCK-END
