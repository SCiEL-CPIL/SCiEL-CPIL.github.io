"""Rewritten program: search for A,B with max distinct pairwise-coprime sums."""
import time
import math
import random
from itertools import combinations

_DEADLINE = 85.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(s):
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(s[i], s[j]) != 1:
                return False
    return True

def _is_valid(A, B):
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    for x in A:
        if x < 1 or x > 128:
            return False
    for x in B:
        if x < 1 or x > 128:
            return False
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) < 2:
        return False
    return _pairwise_coprime(sorted(sums))

def _score(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    return len(sums)

def _factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _solve():
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    best_score = 0
    
    # Structural approach: enumerate B, for each find A with disjoint prime factors
    # across all 9 sums
    
    # Generate B candidates: structured + random
    B_candidates = []
    for b1 in range(1, 40):
        for b2 in range(b1 + 1, 40):
            for b3 in range(b2 + 1, 40):
                B_candidates.append([b1, b2, b3])
    
    random.seed(42)
    for _ in range(500):
        B_candidates.append(random.sample(range(1, 129), 3))
    
    random.shuffle(B_candidates)
    
    for B in B_candidates:
        if _elapsed() > _DEADLINE:
            break
        if len(set(B)) != 3:
            continue
        if any(x < 1 or x > 128 for x in B):
            continue
        
        # For each a, compute sum-set and its prime factors
        valid_a = []
        for a in range(1, 129):
            s = [a + B[0], a + B[1], a + B[2]]
            # Check internal pairwise coprime
            ok = True
            for i in range(3):
                for j in range(i + 1, 3):
                    if _gcd(s[i], s[j]) != 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                pf = set()
                for val in s:
                    pf |= _factorize(val)
                valid_a.append((a, s, pf))
        
        if len(valid_a) < 3:
            continue
        
        # Find 3 a's with mutually disjoint prime factor sets
        n_va = len(valid_a)
        found = False
        for i in range(n_va):
            if found or _elapsed() > _DEADLINE:
                break
            for j in range(i + 1, n_va):
                if found or _elapsed() > _DEADLINE:
                    break
                if valid_a[i][2] & valid_a[j][2]:
                    continue
                for k in range(j + 1, n_va):
                    if found or _elapsed() > _DEADLINE:
                        break
                    if valid_a[i][2] & valid_a[k][2] or valid_a[j][2] & valid_a[k][2]:
                        continue
                    A = [valid_a[i][0], valid_a[j][0], valid_a[k][0]]
                    sc = _score(A, B)
                    if sc > best_score:
                        best_A = A[:]
                        best_B = B[:]
                        best_score = sc
                    found = True
                    break
                if found:
                    break
            if found:
                break
        
        if best_score >= 9:
            break
    
    if best_score >= 2 and _is_valid(best_A, best_B):
        return [best_A, best_B]
    
    # Fallback: brute force small search
    for a1 in range(1, 20):
        for a2 in range(a1 + 1, 20):
            for a3 in range(a2 + 1, 20):
                for b1 in range(1, 20):
                    for b2 in range(b1 + 1, 20):
                        for b3 in range(b2 + 1, 20):
                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            if _is_valid(A, B):
                                sc = _score(A, B)
                                if sc > best_score:
                                    best_A = A[:]
                                    best_B = B[:]
                                    best_score = sc
    if best_score >= 2:
        return [best_A, best_B]
    
    return [[1, 2, 3], [1, 2, 3]]

def construct():
    try:
        result = _solve()
        if _is_valid(result[0], result[1]):
            return result
    except Exception:
        pass
    return [[1, 2, 3], [1, 2, 3]]
