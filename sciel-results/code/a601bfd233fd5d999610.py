"""Improved program: structural change to precomputed lookup tables + exhaustive k-search.
Combines the precomputation idea from inspiration 1 (score 1.6682) with a structural
shift from per-target exhaustive search to table-driven lookup. The key insight is that
for n<=128, the optimal k is small (1-4), and we can precompute all valid antichain
representations once, then look them up. This is a different modeling family: instead of
searching per target, we enumerate all valid (k, terms) tuples and index by sum.
"""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_DEADLINE = 78.0  # safety margin before 90s kill
_START = time.time()

def _is_antichain_sorted(terms):
    """Check if sorted terms form a divisibility antichain."""
    n = len(terms)
    for i in range(n):
        a = terms[i]
        for j in range(i + 1, n):
            b = terms[j]
            if b % a == 0:
                return False
    return True

def _is_smooth(x):
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _is_antichain(terms):
    """Check if terms form a divisibility antichain."""
    n = len(terms)
    for i in range(n):
        a = terms[i]
        for j in range(i + 1, n):
            b = terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def construct():
    """
    Returns 128 rows for targets 1..128. Each row is a list of 1..30 distinct
    positive 2,3,5-smooth terms summing to the target, forming a divisibility antichain.
    Minimizes total number of summands.
    """
    # Precompute smooth numbers up to 128
    smooth = [x for x in range(1, 129) if _is_smooth(x)]
    smooth_sorted = sorted(smooth)
    
    # lookup: n -> (k, terms) for minimum k found so far
    lookup = {}
    
    # k=1: single smooth number
    for x in smooth:
        if x <= 128:
            lookup[x] = (1, (x,))
    
    # k=2: pairs
    for i in range(len(smooth_sorted)):
        a = smooth_sorted[i]
        for j in range(i + 1, len(smooth_sorted)):
            b = smooth_sorted[j]
            s = a + b
            if s > 128:
                break
            if s in lookup:  # already covered by k=1
                continue
            if b % a == 0:
                continue
            lookup[s] = (2, (a, b))
    
    # k=3: triples
    for i in range(len(smooth_sorted)):
        a = smooth_sorted[i]
        for j in range(i + 1, len(smooth_sorted)):
            b = smooth_sorted[j]
            s2 = a + b
            if s2 > 128:
                break
            if b % a == 0:
                continue
            for k in range(j + 1, len(smooth_sorted)):
                c = smooth_sorted[k]
                s3 = s2 + c
                if s3 > 128:
                    break
                if s3 in lookup:  # already covered by k<=2
                    continue
                if c % a == 0 or c % b == 0:
                    continue
                lookup[s3] = (3, (a, b, c))
    
    # Check which targets are still uncovered
    uncovered = [n for n in range(1, 129) if n not in lookup]
    
    # For uncovered targets, search for k>=4
    for n in uncovered:
        if time.time() - _START > _DEADLINE:
            break
        found = None
        for k in range(4, 11):
            if time.time() - _START > _DEADLINE:
                break
            candidates = [x for x in smooth_sorted if x < n]  # strict: need room for others
            # Actually, x can be <= n, but sum must equal n with k distinct terms
            candidates = [x for x in smooth_sorted if x <= n]
            for terms in combinations(candidates, k):
                if sum(terms) == n and _is_antichain_sorted(terms):
                    found = list(terms)
                    break
            if found:
                break
        if found:
            lookup[n] = (len(found), tuple(found))
    
    # Build answers
    answers = []
    for n in range(1, 129):
        if n in lookup:
            answers.append(list(lookup[n][1]))
        else:
            # Ultimate fallback
            if _is_smooth(n):
                answers.append([n])
            elif n > 1 and _is_smooth(n - 1) and (n - 1) != 1:
                answers.append([1, n - 1])
            else:
                answers.append([1, n - 1])
    
    return answers
# EVOLVE-BLOCK-END
