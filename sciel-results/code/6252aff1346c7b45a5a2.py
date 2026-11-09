import math
from functools import reduce

def sigma(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            result *= (p ** (exp + 1) - 1) // (p - 1)
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (temp + 1)
    return result

def count_solutions(M):
    limit = int(math.isqrt(M))
    count = 0
    for k in range(1, limit + 1):
        if M % (k * sigma(k)) == 0:
            count += 1
    return count

def construct():
    candidates = []
    
    # Generate LCMs for initial segments
    # M_1: k=1..30
    M1 = 1
    for k in range(1, 31):
        M1 = math.lcm(M1, k * sigma(k))
    candidates.append(M1)
    
    # M_2: k=1..40
    M2 = 1
    for k in range(1, 41):
        M2 = math.lcm(M2, k * sigma(k))
    candidates.append(M2)
    
    # M_3: k=1..45
    M3 = 1
    for k in range(1, 46):
        M3 = math.lcm(M3, k * sigma(k))
    if M3 <= 10**18:
        candidates.append(M3)
        
    # M_4: k=1..50
    M4 = 1
    for k in range(1, 51):
        M4 = math.lcm(M4, k * sigma(k))
    if M4 <= 10**18:
        candidates.append(M4)
        
    best_M = 0
    best_k = []
    best_count = 0
    
    for M in candidates:
        if M > 10**18:
            continue
        limit = int(math.isqrt(M))
        ks = []
        for k in range(1, limit + 1):
            if M % (k * sigma(k)) == 0:
                ks.append(k)
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
            
    return {'M': best_M, 'k': best_k}
