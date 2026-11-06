"""Rewritten program for maximizing pairs of powerful numbers summing to N."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import Counter
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin for 120s limit
    
    def gen_powerful(L):
        """Generate all powerful numbers <= L."""
        limit = isqrt(L)
        if limit < 2:
            return [1]
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(limit) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        primes = [i for i in range(2, limit + 1) if sieve[i]]
        
        powerful = set([1])
        
        def dfs(idx, current):
            if idx == len(primes):
                if current <= L:
                    powerful.add(current)
                return
            p = primes[idx]
            # Skip this prime
            dfs(idx + 1, current)
            # Include p with exponent 2, 3, 4, ...
            val = current
            p2 = p * p
            if p2 > L // current:
                return
            val *= p2
            while val <= L:
                dfs(idx + 1, val)
                if val > L // p:
                    break
                val *= p
        
        dfs(0, 1)
        return sorted(powerful)
    
    # Generate powerful numbers up to L
    L = 10**12
    P = gen_powerful(L)
    P_set = set(P)
    n = len(P)
    
    if time.time() > deadline - 50:
        L = 10**10
        P = gen_powerful(L)
        P_set = set(P)
        n = len(P)
    
    # Strategy: For each a in P, for each b in P with b >= a, if a+b <= 10^18,
    # count a+b. But this is O(n^2) which is too slow for large n.
    # 
    # Instead, let's use the fact that we want to maximize the number of pairs.
    # The most productive N will be one where many powerful numbers a have N-a also powerful.
    # 
    # Let's try a different approach: iterate over a in P, and for each a, 
    # find how many b in P satisfy b >= a and a+b <= 10^18 and N-a is powerful
    # for some fixed N. But we don't know N.
    #
    # Practical approach: Use the pairwise sum with early termination and 
    # limit the inner loop.
    
    MAX_N = 10**18
    counts = Counter()
    
    # For efficiency, only consider a up to MAX_N // 2
    # And b >= a, b <= MAX_N - a
    
    # Iterate over a in P
    for i, a in enumerate(P):
        if a > MAX_N // 2:
            break
        if time.time() > deadline - 15:
            break
        max_b = MAX_N - a
        # Find range of b in P: b >= a, b <= max_b
        left = bisect.bisect_left(P, a)
        right = bisect.bisect_right(P, max_b)
        # For each b in P[left:right], add count for a+b
        # This is still O(n) per a, so O(n^2) total
        # To speed up, let's limit the number of b's we check per a
        # Or use a different strategy
        
        # For now, just do it but with a cap on total operations
        num_b = right - left
        if num_b > 200000:
            # Too many, sample or skip
            # Just check a subset
            step = max(1, num_b // 10000)
            for j in range(left, right, step):
                b = P[j]
                s = a + b
                if s <= MAX_N:
                    counts[s] += 1
        else:
            for j in range(left, right):
                b = P[j]
                s = a + b
                if s <= MAX_N:
                    counts[s] += 1
    
    if counts:
        best_N = max(counts, key=counts.get)
        # Find all valid pairs for best_N
        best_pairs = []
        for a in P:
            if a > best_N:
                break
            b = best_N - a
            if b >= a and b in P_set:
                best_pairs.append([a, b])
    else:
        best_N = 2
        best_pairs = [[1, 1]]
    
    # Verify with exact factorization
    def is_powerful(n):
        if n == 1:
            return True
        m = n
        p = 2
        while p * p <= m:
            if m % p == 0:
                exp = 0
                while m % p == 0:
                    m //= p
                    exp += 1
                if exp == 1:
                    return False
            p += 1 if p == 2 else 2
        if m > 1:
            return False
        return True
    
    valid_pairs = []
    for a, b in best_pairs:
        if a + b == best_N and a <= b and is_powerful(a) and is_powerful(b):
            valid_pairs.append([a, b])
    
    if not valid_pairs:
        best_N = 2
        valid_pairs = [[1, 1]]
    
    return {'N': best_N, 'pairs': valid_pairs}
# EVOLVE-BLOCK-END
