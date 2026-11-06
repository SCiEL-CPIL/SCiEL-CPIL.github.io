import math
import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    def is_powerful(n):
        if n == 1:
            return True
        m = n
        for p in range(2, 10**6):
            if p * p > m:
                break
            if m % p == 0:
                c = 0
                while m % p == 0:
                    m //= p
                    c += 1
                if c == 1:
                    return False
        if m > 1:
            return False
        return True

    def factorize(n):
        factors = {}
        m = n
        for p in range(2, 10**6):
            if p * p > m:
                break
            if m % p == 0:
                c = 0
                while m % p == 0:
                    m //= p
                    c += 1
                factors[p] = c
        if m > 1:
            factors[m] = 1
        return factors

    def find_pairs_for_M(M, Q):
        # Find all (x, y) with x <= y, x^2 + y^2 = M
        pairs = []
        limit_x = int(math.isqrt(M // 2)) + 1
        if limit_x <= 3 * 10**7:
            # Direct enumeration
            for x in range(0, limit_x + 1):
                x2 = x * x
                y2 = M - x2
                y = math.isqrt(y2)
                if y * y == y2:
                    if x <= y:
                        a = Q * x2
                        b = Q * y2
                        pairs.append([a, b])
        else:
            # Use Gaussian integer factorization / recursive method
            # For now, if M is too large for direct enumeration, we'll try to find
            # representations using the factorization of M into 1 mod 4 primes.
            # This is complex; we'll fall back to checking if M is small enough
            # or use a probabilistic check.
            # Actually, if limit_x > 3*10^7, M > 2*(3*10^7)^2 = 1.8*10^15.
            # Given Q >= 4, N = Q*M > 7.2*10^15, which is < 10^18.
            # So large M is possible.
            # We need a smarter method.
            # However, the number of representations is prod(e_i + 1) / 2.
            # If M has many small 1 mod 4 prime factors, the number of pairs is large,
            # but M might be small enough to enumerate.
            # If M is large, it likely has fewer distinct small prime factors,
            # so the number of pairs is smaller, and we can afford to search?
            # No, we need to find them.
            # Let's stick to enumeration for now, but with a tighter limit.
            # If limit_x is too big, we skip this M.
            return []
        return pairs

    def dfs_exponents(idx, current_M, current_score, exponents, primes_list, M_max):
        best_M = [1]
        best_score = [0]
        
        def _dfs(i, cM, cS, exps):
            if time_left() < 5:
                return
            if cS > best_score[0]:
                best_score[0] = cS
                best_M[0] = cM
                # Store exponents if needed
            if i == len(primes_list):
                return
            p = primes_list[i]
            val = 1
            e = 0
            while cM * val <= M_max:
                _dfs(i + 1, cM * val, cS * (e + 1), exps + [e])
                val *= p
                e += 1
                if val > M_max:
                    break
        
        _dfs(idx, current_M, current_score, exponents)
        return best_M[0], best_score[0]

    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257]
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    best_N = 0
    best_pairs = []

    # Iterate over possible Q values
    # Q = S^2, S = product of first k primes
    for k in range(1, 10):
        if time_left() < 10:
            break
        
        S = 1
        for i in range(k):
            S *= primes_all[i]
        Q = S * S
        
        M_max = 10**18 // Q
        if M_max < 2:
            continue
        
        # Find best M <= M_max
        # We want M to be composed of 1 mod 4 primes to maximize representations
        # Use DFS to find optimal exponent vector
        best_M, best_score = dfs_exponents(0, 1, 1, [], primes_1mod4[:15], M_max)
        
        if best_M < 1:
            continue
            
        M = best_M
        N = Q * M
        
        if N < 2 or N > 10**18:
            continue
            
        pairs = find_pairs_for_M(M, Q)
        
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N

    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]

    return {'N': best_N, 'pairs': best_pairs}
