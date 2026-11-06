import random
import math
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    random.seed(42)
    
    # Primes in [2, 256]
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [p for p in range(2, 257) if is_prime(p)]
    
    # Precompute: for each pair of primes (p, q) with p < q, 
    # check if there exist a,b in [1,128] with a+b=p and a'+b'=q 
    # such that we can form a valid 3x3 grid.
    
    best = None
    best_count = 0
    
    # Strategy: pick 9 primes > 128 (guaranteed pairwise coprime)
    # and try to find A, B such that A+B = exactly those 9 primes.
    # 
    # If A = {a1,a2,a3} and B = {b1,b2,b3}, then A+B has 9 values.
    # We need: for each i,j, a_i + b_j is one of the 9 target primes.
    # 
    # Key structure: fix a1=1. Then b_j = p_{1j} - 1 for some assignment.
    # Then a_i + b_j = a_i + p_{1j} - 1 must also be a target prime.
    
    # Let's try a more direct approach:
    # Pick 3 "row primes" for a1+B: r1, r2, r3 (these are a1+b1, a1+b2, a1+b3)
    # Then B = {r1-a1, r2-a1, r3-a1}
    # For a2: a2+bj = a2 + rj - a1 must be a prime in our target set for all j
    # For a3: a3+bj = a3 + rj - a1 must be a prime in our target set for all j
    
    # So we need: find 3 primes r1,r2,r3 and a1,a2,a3 in [1,128] such that:
    #   a2 + rj - a1 is prime (in target set) for j=1,2,3
    #   a3 + rj - a1 is prime (in target set) for j=1,2,3
    #   All 9 sums are distinct
    #   a1,a2,a3 distinct; b1,b2,b3 distinct; all in [1,128]
    
    # Simpler: just brute-force search over small A and B
    
    # Let's try: A = {1, x, y}, B = {p, q, r} where p,q,r are primes
    # Then A+B = {1+p, 1+q, 1+r, x+p, x+q, x+r, y+p, y+q, y+r}
    # We need all 9 to be pairwise coprime.
    
    # Even simpler: search over all A, B with small values
    
    # Precompute coprime check
    def pairwise_coprime(lst):
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    # Search: try A = {1, a2, a3}, B = {b1, b2, b3}
    # To maximize speed, try structured families.
    
    # Family 1: A = {1, 2, 3}, vary B
    # A+B = {1+b, 2+b, 3+b} for each b in B
    # For B = {b1,b2,b3}: sums = {1+b1,2+b1,3+b1, 1+b2,2+b2,3+b2, 1+b3,2+b3,3+b3}
    # Need all 9 pairwise coprime.
    
    # Let's do a broader search with time limit
    
    # Try many random (A,B) pairs
    found = None
    
    # First, try systematic search with A containing 1
    # A = {1, a2, a3}, B = {b1, b2, b3}
    # Sums: {1+b1, 1+b2, 1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3}
    
    # For speed, let's try A = {1, 2, 3} first
    for a2 in range(2, 50):
        for a3 in range(a2+1, 50):
            A = [1, a2, a3]
            for b1 in range(1, 129):
                for b2 in range(b1+1, 129):
                    for b3 in range(b2+1, 129):
                        if time.time() > deadline:
                            break
                        B = [b1, b2, b3]
                        sums = set()
                        valid = True
                        for a in A:
                            for b in B:
                                s = a + b
                                if s in sums:
                                    # duplicate, still ok but fewer distinct
                                    pass
                                sums.add(s)
                        distinct_sums = list(sums)
                        if len(distinct_sums) < 7:  # need at least 7 to beat seed
                            continue
                        if pairwise_coprime(distinct_sums):
                            if len(distinct_sums) > best_count:
                                best_count = len(distinct_sums)
                                best = (A, B)
                                if best_count == 9:
                                    return [best[0], best[1]]
                        if b3 > 100:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best and best_count > 6:
        return [best[0], best[1]]
    
    # Fallback: try more random search
    for _ in range(100000):
        if time.time() > deadline:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        distinct_sums = list(sums)
        if len(distinct_sums) >= 7 and pairwise_coprime(distinct_sums):
            if len(distinct_sums) > best_count:
                best_count = len(distinct_sums)
                best = (A, B)
                if best_count == 9:
                    return [best[0], best[1]]
    
    if best and best_count > 6:
        return [best[0], best[1]]
    
    # Return seed if nothing better
    return [[1, 7, 13], [4, 10, 16]]
