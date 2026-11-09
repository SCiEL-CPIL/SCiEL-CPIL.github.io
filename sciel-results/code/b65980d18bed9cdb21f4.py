import math
import bisect
from itertools import product

def construct():
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = math.lcm(L, i)
    
    weights = [L // i for i in range(1, N + 1)]
    
    half = N // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Encode coeffs into a single integer to save memory
    # c in {-1, 0, 1} -> 0, 1, 2
    # val = sum (c_i + 1) * 3^i
    
    def encode(coeffs):
        val = 0
        for i, c in enumerate(coeffs):
            val += (c + 1) * (3 ** i)
        return val

    def decode(val, length):
        coeffs = []
        for i in range(length):
            c = (val // (3 ** i)) % 3
            coeffs.append(c - 1)
        return tuple(coeffs)

    dict_A = {}
    for coeffs in product([-1, 0, 1], repeat=half):
        s = 0
        for c, w in zip(coeffs, w1):
            if c != 0:
                s += c * w
        enc = encode(coeffs)
        if s not in dict_A:
            dict_A[s] = enc
            
    keys_A = sorted(dict_A.keys())
    
    dict_B = {}
    for coeffs in product([-1, 0, 1], repeat=N - half):
        s = 0
        for c, w in zip(coeffs, w2):
            if c != 0:
                s += c * w
        enc = encode(coeffs)
        if s not in dict_B:
            dict_B[s] = enc
            
    keys_B = sorted(dict_B.keys())
    
    best_val = float('inf')
    best_enc_A = None
    best_enc_B = None
    
    for a in keys_A:
        target = -a
        idx = bisect.bisect_left(keys_B, target)
        
        candidates = []
        for k in range(max(0, idx - 3), min(len(keys_B), idx + 4)):
            candidates.append(keys_B[k])
            
        for b in candidates:
            total = a + b
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_val:
                best_val = abs_total
                best_enc_A = dict_A[a]
                best_enc_B = dict_B[b]
                
    if best_enc_A is None:
        return [0] * 28 + [1, -1]
        
    c_a = decode(best_enc_A, half)
    c_b = decode(best_enc_B, N - half)
    return list(c_a + c_b)
