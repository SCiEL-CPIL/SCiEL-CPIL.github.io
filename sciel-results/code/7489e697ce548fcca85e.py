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
    smooth_set = set(smooth)
    smooth_sorted = sorted(smooth)  # already sorted
    
    # For each target n, find the minimum-k antichain representation
    # Strategy: precompute all valid antichains of size k=1,2,3,4 and index by sum
    # Then for each target, pick the representation with smallest k
    
    # k=1: single smooth number
    # k=2: pairs of smooth numbers that are antichain
    # k=3: triples
    # k=4: quadruples (unlikely needed for n<=128, but include for completeness)
    
    # Build lookup: sum -> list of (k, terms) sorted by k
    lookup = {}  # n -> list of (k, tuple(terms))
    
    # k=1
    for x in smooth:
        if x <= 128:
            if x not in lookup:
                lookup[x] = []
            lookup[x].append((1, (x,)))
    
    # k=2
    for i in range(len(smooth_sorted)):
        a = smooth_sorted[i]
        for j in range(i + 1, len(smooth_sorted)):
            b = smooth_sorted[j]
            s = a + b
            if s > 128:
                break
            # Check antichain: b % a != 0 (since a < b)
            if b % a == 0:
                continue
            if s not in lookup:
                lookup[s] = []
            lookup[s].append((2, (a, b)))
    
    # k=3
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
                # Check antichain: c%a!=0, c%b!=0 (b%a already checked)
                if c % a == 0 or c % b == 0:
                    continue
                if s3 not in lookup:
                    lookup[s3] = []
                lookup[s3].append((3, (a, b, c)))
    
    # k=4 (only if needed)
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
                if c % a == 0 or c % b == 0:
                    continue
                for l in range(k + 1, len(smooth_sorted)):
                    d = smooth_sorted[l]
                    s4 = s3 + d
                    if s4 > 128:
                        break
                    # Check antichain: d%a, d%b, d%c
                    if d % a == 0 or d % b == 0 or d % c == 0:
                        continue
                    if s4 not in lookup:
                        lookup[s4] = []
                    lookup[s4].append((4, (a, b, c, d)))
    
    # Now construct answers
    answers = []
    for n in range(1, 129):
        if time.time() - _START > _DEADLINE:
            # Fallback: use whatever we have
            if n in lookup:
                best = min(lookup[n], key=lambda x: x[0])
                answers.append(list(best[1]))
            else:
                # Last resort: try a simple greedy
                answers.append([n] if _is_smooth(n) else [1, n-1] if _is_smooth(n-1) else [1, 1, n-2])
            continue
        
        if n in lookup:
            # Pick the representation with smallest k
            best = min(lookup[n], key=lambda x: x[0])
            answers.append(list(best[1]))
        else:
            # Should not happen for n<=128, but fallback
            # Try a simple search
            found = None
            for k in range(1, 8):
                if time.time() - _START > _DEADLINE:
                    break
                for terms in combinations([x for x in smooth if x <= n], k):
                    if sum(terms) == n and _is_antichain(terms):
                        found = list(terms)
                        break
                if found:
                    break
            if found:
                answers.append(found)
            else:
                # Ultimate fallback
                answers.append([n] if _is_smooth(n) else [1, n-1])
    
    return answers
# EVOLVE-BLOCK-END
