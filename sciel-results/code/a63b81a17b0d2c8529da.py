import math
import time
import sys
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Strategy:
    # 1. Choose Q = S^2 where S is a product of small primes (primorial).
    # 2. Let N = Q * M.
    # 3. a = Q * x^2, b = Q * y^2. a, b powerful.
    # 4. x^2 + y^2 = M.
    # 5. Maximize number of pairs (x,y) with x <= y.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    best_N = 0
    best_pairs = []
    
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 991, 997]
    
    # Iterate k. Prioritize k=4,5,6 which are sweet spots.
    k_order = [4, 5, 6, 3, 7, 2, 8, 1]
    
    for k in k_order:
        if time.time() > deadline - 10:
            break
            
        S = 1
        for i in range(k):
            S *= primes[i]
        Q = S * S
        
        M_max = 10**18 // Q
        if M_max < 2:
            continue
            
        best_M = 1
        best_score = 0
        best_exponents = []
        
        search_primes = p_1mod4[:15]
        
        e_maxs = []
        for p in search_primes:
            e = 0
            val = 1
            while val * p <= M_max:
                val *= p
                e += 1
            e_maxs.append(e)
        
        def dfs(idx, current_M, current_score, exponents):
            nonlocal best_M, best_score, best_exponents
            
            if time.time() > deadline - 5:
                return
                
            if current_score > best_score:
                best_score = current_score
                best_M = current_M
                best_exponents = exponents[:]
                
            if idx == len(search_primes):
                return
            
            p = search_primes[idx]
            val = 1
            e = 0
            while current_M * val <= M_max:
                next_M = current_M * val
                next_score = current_score * (e + 1)
                
                exponents.append(e)
                dfs(idx + 1, next_M, next_score, exponents)
                exponents.pop()
                
                val *= p
                e += 1
                if e > e_maxs[idx]:
                    break

        dfs(0, 1, 1, [])
        
        M = best_M
        N = Q * M
        
        if N < 2 or N > 10**18:
            continue
            
        limit_x = int(math.isqrt(M // 2)) + 1
        
        if limit_x > 10**7:
            continue

        try:
            import numpy as np
            xs = np.arange(0, limit_x + 1, dtype=np.int64)
            y2 = M - xs * xs
            ys = np.isqrt(y2)
            mask = ys * ys == y2
            valid_x = xs[mask]
            valid_y = ys[mask]
            
            pairs = []
            for x, y in zip(valid_x, valid_y):
                if x <= y:
                    a = Q * x * x
                    b = Q * y * y
                    pairs.append([a, b])
                    
            if len(pairs) > len(best_pairs):
                best_pairs = pairs
                best_N = N
                
        except Exception:
            pairs = []
            for x in range(0, limit_x + 1):
                x2 = x * x
                y2 = M - x2
                y = math.isqrt(y2)
                if y * y == y2:
                    if x <= y:
                        a = Q * x2
                        b = Q * y2
                        pairs.append([a, b])
            
            if len(pairs) > len(best_pairs):
                best_pairs = pairs
                best_N = N

    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
