"""Construct pairwise-coprime sumsets with 3 distinct integers each in [1,128]."""
import time
from math import gcd
from itertools import combinations

def _is_prime(n):
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

def _pairwise_coprime(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    distinct = sorted(set(vals))
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return (num_distinct_coprime, is_valid)."""
    sums = [a + b for a in A for b in B]
    distinct = sorted(set(sums))
    if not _pairwise_coprime(distinct):
        return 0, False
    return len(distinct), True

def _search():
    deadline = time.time() + 80  # 80s to leave margin before 90s kill
    best = (0, [1, 7, 13], [4, 10, 16])  # baseline: 5 distinct coprime sums
    
    # Precompute primes up to 256 (max sum is 128+128=256)
    primes = [p for p in range(2, 257) if _is_prime(p)]
    prime_set = set(primes)
    
    # Strategy 1: Try to find 9 distinct primes as sums
    # If all 9 sums are distinct primes, they are automatically pairwise coprime
    # We need A = [a1,a2,a3], B = [b1,b2,b3] with a_i + b_j all distinct primes
    
    # Precompute valid triples for A: 3 distinct values in [1,128]
    # For efficiency, iterate over structured families first
    
    # Family: arithmetic progressions
    # A = [x, x+d, x+2d], B = [y, y+e, y+2e]
    # Sums: x+y, x+y+e, x+y+2e, x+d+y, x+d+y+e, x+d+y+2e, x+2d+y, x+2d+y+e, x+2d+y+2e
    # These are a 3x3 grid of values
    
    found_9 = False
    
    # Try small structured searches first
    for x in range(1, 127):
        if time.time() > deadline - 10:
            break
        for d in range(1, 60):
            if time.time() > deadline - 10:
                break
            a1, a2, a3 = x, x + d, x + 2 * d
            if a3 > 128:
                break
            if a1 == a2 or a2 == a3:
                continue
            for y in range(1, 127):
                if time.time() > deadline - 10:
                    break
                for e in range(1, 60):
                    if time.time() > deadline - 10:
                        break
                    b1, b2, b3 = y, y + e, y + 2 * e
                    if b3 > 128:
                        break
                    if b1 == b2 or b2 == b3:
                        continue
                    
                    A = [a1, a2, a3]
                    B = [b1, b2, b3]
                    sums = [a + b for a in A for b in B]
                    distinct = set(sums)
                    
                    if len(distinct) == 9:
                        # Check if all are primes (guarantees pairwise coprime)
                        if all(s in prime_set for s in distinct):
                            sc, valid = _score(A, B)
                            if valid and sc > best[0]:
                                best = (sc, A[:], B[:])
                                found_9 = True
                                return best  # Can't do better than 9
                        elif _pairwise_coprime(distinct):
                            sc, valid = _score(A, B)
                            if valid and sc > best[0]:
                                best = (sc, A[:], B[:])
    
    if found_9:
        return best
    
    # Strategy 2: Broader search - iterate over candidate A triples, then B triples
    # Use a smarter approach: for each A, find B that maximizes coprime distinct sums
    
    # Generate candidate A triples (biased toward values that produce prime sums)
    # Try all A with a_i in [1, 128], distinct, sorted
    # But 128 choose 3 is ~340k, too many. Use structured subsets.
    
    # Try A triples where elements are "spread out"
    candidate_As = []
    for a1 in range(1, 126):
        for a2 in range(a1 + 1, 127):
            for a3 in range(a2 + 1, 129):
                if a3 > 128:
                    break
                candidate_As.append([a1, a2, a3])
    
    # That's still ~340k. Let's be smarter.
    # Try A from a smaller structured set first
    small_As = []
    for a1 in range(1, 50):
        for a2 in range(a1 + 1, 80):
            for a3 in range(a2 + 1, 129):
                if a3 > 128:
                    break
                small_As.append([a1, a2, a3])
    
    for A in small_As:
        if time.time() > deadline - 5:
            break
        # For this A, try B triples
        for b1 in range(1, 126):
            if time.time() > deadline - 5:
                break
            for b2 in range(b1 + 1, 127):
                if time.time() > deadline - 5:
                    break
                for b3 in range(b2 + 1, 129):
                    if b3 > 128:
                        break
                    B = [b1, b2, b3]
                    sums = [a + b for a in A for b in B]
                    distinct = set(sums)
                    if len(distinct) <= best[0]:
                        continue
                    if len(distinct) == 9 and all(s in prime_set for s in distinct):
                        best = (9, A[:], B[:])
                        return best
                    if _pairwise_coprime(distinct):
                        sc, valid = _score(A, B)
                        if valid and sc > best[0]:
                            best = (sc, A[:], B[:])
                            if sc == 9:
                                return best
    
    return best

def construct():
    deadline = time.time() + 85
    best = _search()
    if time.time() > deadline:
        # Fallback to known good solution
        return [[1, 7, 13], [4, 10, 16]]
    return [best[1], best[2]]
