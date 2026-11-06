import math
import time
import sys
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Strategy:
    # 1. Choose Q = S^2 where S is a product of small primes (primorial).
    #    Q is powerful.
    # 2. Let N = Q * M.
    # 3. We want N = a + b where a, b are powerful.
    #    If we set a = Q * x^2 and b = Q * y^2, then a, b are powerful 
    #    (since Q is powerful, x^2 is a square, product of powerfuls is powerful).
    #    Then Q * x^2 + Q * y^2 = Q * (x^2 + y^2) = N = Q * M.
    #    So we need x^2 + y^2 = M.
    # 4. The number of pairs (a,b) corresponds to the number of pairs (x,y) 
    #    with x <= y and x^2 + y^2 = M.
    #    Note: a <= b implies x <= y (since Q > 0).
    # 5. We want to maximize the number of representations of M as sum of two squares.
    #    This is governed by the number of prime factors of M congruent to 1 mod 4.
    #    If M = 2^k * prod p_i^e_i (p_i = 1 mod 4) * prod q_j^f_j (q_j = 3 mod 4),
    #    the number of representations r_2(M) is 4 * prod (e_i + 1) if all f_j are even, else 0.
    #    The number of unordered pairs {x,y} with x,y >= 0 is roughly r_2(M)/8 (adjusting for x=y and x=0).
    
    # Constraints:
    # N <= 10^18.
    # a, b <= N.
    # We need to find M such that Q*M <= 10^18.
    # So M <= 10^18 / Q.
    
    # Let's try different Q values.
    # Q = (product of first k primes)^2.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29...
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    best_N = 0
    best_pairs = []
    
    # We will iterate over the number of primes in the primorial S.
    # S = prod_{i=1}^k p_i. Q = S^2.
    # M_max = 10^18 // Q.
    
    # For a given M_max, we want to find M <= M_max with many 1 mod 4 prime factors.
    # The "densest" M in terms of 1 mod 4 factors will be of the form:
    # M = 2^a * (prod p_i^{e_i}) where p_i are 1 mod 4 primes.
    # To maximize prod(e_i+1) subject to M <= M_max, we should use small primes with high exponents.
    # This is a knapsack-like problem or greedy search.
    
    # Primes 1 mod 4: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, ...
    # Primes 3 mod 4: 3, 7, 11, 19, 23, 31, 43, 47, 59, 67, ...
    # We can also include 3 mod 4 primes if their exponents are even, but they don't increase the count of representations, only the size.
    # So it's optimal to use only 2 and 1 mod 4 primes for M.
    
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 991, 997]
    
    # We need to search for M.
    # Since M_max depends on Q, and Q grows fast, let's evaluate a few Q.
    
    for k in range(1, 8):
        if time.time() > deadline - 20:
            break
            
        S = 1
        for i in range(k):
            S *= primes[i]
        Q = S * S
        
        M_max = 10**18 // Q
        if M_max < 2:
            continue
            
        # Find best M <= M_max composed of 2 and 1 mod 4 primes.
        # We want to maximize score = prod(e_i + 1).
        # Greedy approach: Start with M=1, score=1.
        # Repeatedly multiply by the smallest prime p (including 2) such that M*p <= M_max.
        # This is not strictly optimal for maximizing product of (e+1), but usually close.
        # Actually, to maximize product of (e_i+1), we should balance exponents.
        # E.g., 5^2 * 13^1 vs 5^1 * 13^2. 5^2*13=325, 5*13^2=845.
        # Score: 5^2 -> (2+1)=3, 13^1 -> (1+1)=2. Total 6.
        # 5^1 -> 2, 13^2 -> 3. Total 6.
        # But 5^3 * 13^1 = 1625. Score 4*2=8.
        
        # Let's use a simple search over exponent vectors for the first few 1 mod 4 primes.
        # M = 2^a * 5^b * 13^c * 17^d * ...
        
        # To be safe and fast, let's just try a few structured M's.
        # Candidate M's:
        # 1. Powers of 2: M = 2^a. Score = 1 (only 2 is not 1 mod 4, wait. 2 is special.
        #    r_2(2^a) = 4 if a even? No.
        #    x^2+y^2=2. (1,1). 1 pair.
        #    x^2+y^2=8. (2,2). 1 pair.
        #    x^2+y^2=32. (4,4). 1 pair.
        #    Generally 2^a has 1 representation if a even? No.
        #    Formula: r_2(n) = 4 * (d_1(n) - d_3(n)).
        #    For n=2^a, divisors are 1, 2, ..., 2^a.
        #    1 mod 4 divisors: 1, 4, 16...
        #    3 mod 4 divisors: 2, 8, 32...
        #    If a is even, #1mod4 = a/2 + 1, #3mod4 = a/2. Diff = 1. r_2=4. 1 pair.
        #    If a is odd, #1mod4 = (a+1)/2, #3mod4 = (a+1)/2. Diff = 0. r_2=0.
        #    So powers of 2 are not good for many pairs.
        
        # We need primes 1 mod 4.
        
        # Let's try M = 5^b * 13^c * 17^d * ...
        # We'll do a limited DFS on exponents.
        
        best_M = 1
        best_score = 0
        best_exponents = []
        
        # Primes to consider: 5, 13, 17, 29, 37, 41, 53, 61
        search_primes = p_1mod4[:10]
        
        # Max exponent for 5: 5^e <= M_max.
        # M_max can be up to 10^18 (if Q=1, but Q>=4).
        
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
            # Try exponents 0, 1, 2, ...
            val = 1
            e = 0
            while current_M * val <= M_max:
                next_M = current_M * val
                next_score = current_score * (e + 1)
                # Pruning: even if we multiply by max possible score from remaining primes, can we beat best?
                # Max score from remaining is bounded.
                dfs(idx + 1, next_M, next_score, exponents + [e])
                val *= p
                e += 1
                if val > M_max:
                    break
        
        # Initial call
        dfs(0, 1, 1, [])
        
        # best_M is the optimal M found.
        M = best_M
        N = Q * M
        
        if N < 2 or N > 10**18:
            continue
            
        # Now find all pairs (x, y) such that x^2 + y^2 = M and x <= y.
        # x^2 <= M/2 => x <= sqrt(M/2).
        # M can be large (up to 10^18/4 = 2.5*10^17). sqrt is ~5*10^8.
        # Iterating x is too slow.
        
        # We need a faster way to list representations of M as sum of two squares.
        # Since M is composed of primes 1 mod 4, we can use Gaussian integer factorization.
        # But implementing that robustly is complex.
        
        # Alternative:
        # If M is small enough, iterate.
        # If M is large, we might have few pairs?
        # No, if M has many factors, number of pairs is prod(e_i+1)/2.
        # If best_score is high, M is likely small enough to iterate?
        # Not necessarily. 5^10 * 13^10 is huge.
        
        # Let's check the size of M.
        # If M <= 10^12, we can iterate x up to 10^6. Fast.
        # If M > 10^12, we need a smarter method or accept fewer pairs?
        # The score is the number of pairs.
        # If we can't find them fast, we can't certify them.
        
        # Let's try to generate pairs.
        pairs = []
        limit_x = int(math.isqrt(M // 2)) + 1
        
        # Optimization: If limit_x is too large, we might time out.
        # 10^8 iterations is ~5-10s in Python.
        # If limit_x > 5*10^7, we are in danger.
        
        # However, note that Q * M = N.
        # If we choose Q to be larger, M becomes smaller.
        # E.g., if Q = 10^10, M <= 10^8. sqrt(M) = 10^4. Very fast.
        # But Q = S^2. S = 2*3*5*7*11*13 = 30030. Q = 9*10^8.
        # M_max = 10^18 / 10^9 = 10^9. sqrt(M) = 31622. Fast.
        
        # So the strategy of large Q (many small primes) is good because it shrinks M.
        
        if limit_x > 5 * 10**7:
            # Too large to brute force x.
            # We could use the factorization method, but for now, let's skip or try a different Q.
            # Actually, if M is large, it has many factors, so many pairs.
            # But we can't find them.
            # Let's try to increase k (larger Q) to shrink M.
            continue
            
        # Iterate x
        M_set = M
        for x in range(0, limit_x + 1):
            x2 = x * x
            y2 = M_set - x2
            y = math.isqrt(y2)
            if y * y == y2:
                if x <= y:
                    a = Q * x2
                    b = Q * y2
                    # Check if a, b are powerful.
                    # a = Q * x^2. Q is powerful. x^2 is powerful. Product is powerful.
                    # Yes.
                    pairs.append([a, b])
                    
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N
            
    # Fallback if no pairs found
    if not best_pairs:
        # Simple powerful sum: 1 + 4 = 5. 1 is powerful, 4 is powerful.
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
