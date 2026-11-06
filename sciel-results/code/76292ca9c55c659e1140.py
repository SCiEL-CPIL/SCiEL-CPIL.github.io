"""Search for ambient 12-set minimizing largest Sidon subset size."""
import random
import time
from itertools import combinations

def _is_sidon(S):
    n = len(S)
    S2 = {x * x for x in S}
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in S2:
                return False
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(A, timeout=2.0):
    n = len(A)
    idx = list(range(n))
    # check full set
    if _is_sidon(A):
        return n
    # try size 11
    for skip in idx:
        if time.time() > timeout:
            return None
        cand = [A[i] for i in idx if i != skip]
        if _is_sidon(cand):
            return 11
    # try size 10
    for skip1 in idx:
        for skip2 in idx:
            if skip2 <= skip1:
                continue
            if time.time() > timeout:
                return None
            cand = [A[i] for i in idx if i != skip1 and i != skip2]
            if _is_sidon(cand):
                return 10
    # try size 9
    for skip1 in idx:
        for skip2 in idx:
            if skip2 <= skip1:
                continue
            for skip3 in idx:
                if skip3 <= skip2:
                    continue
                if time.time() > timeout:
                    return None
                cand = [A[i] for i in idx if i != skip1 and i != skip2 and i != skip3]
                if _is_sidon(cand):
                    return 9
    return 8

def construct():
    deadline = time.time() + 80.0

    def _validate(A):
        if len(A) != 12:
            return False
        if len(set(A)) != 12:
            return False
        if any(x < 0 or x > 128 for x in A):
            return False
        return True

    candidates = []

    # Strategy 1: Clusters of 3 close elements
    # Within a cluster {a, a+1, a+2}: sums a+a=2a, a+(a+1)=2a+1, a+(a+2)=2a+2, (a+1)+(a+1)=2a+2
    # Collision: a+(a+2) = (a+1)+(a+1) = 2a+2, so at most 2 from each cluster
    # 4 clusters of 3 => max Sidon <= 8
    
    # Try different cluster spacings
    for gap in [10, 15, 20, 25, 30, 40, 50]:
        for offset in range(0, 129 - 3*gap, 2):
            A = []
            for k in range(4):
                base = offset + k * gap
                if base + 2 > 128:
                    break
                A.extend([base, base+1, base+2])
            if len(A) == 12 and len(set(A)) == 12:
                ms = _max_sidon_size(A, timeout=0.5)
                if ms is not None and ms < 12:
                    candidates.append((A, ms))
    
    # Strategy 2: Pairs of close elements (6 pairs)
    # Within pair {a, a+1}: sums 2a, 2a+1, 2a+2 - all unique within pair
    # But across pairs, collisions happen
    for gap in [5, 8, 10, 12, 15, 20]:
        for offset in range(0, 129 - 6*gap, 2):
            A = []
            for k in range(6):
                base = offset + k * gap
                if base + 1 > 128:
                    break
                A.extend([base, base+1])
            if len(A) == 12 and len(set(A)) == 12:
                ms = _max_sidon_size(A, timeout=0.5)
                if ms is not None and ms < 12:
                    candidates.append((A, ms))
    
    # Strategy 3: Random with local search
    rng = random.Random(42)
    for _ in range(30000):
        if time.time() > deadline - 5.0:
            break
        A = sorted(rng.sample(range(0, 129), 12))
        ms = _max_sidon_size(A, timeout=0.3)
        if ms is not None and ms < 12:
            candidates.append((A, ms))
    
    if not candidates:
        # Fallback: try a known bad set
        # {0,1,2, 10,11,12, 20,21,22, 30,31,32}
        A = [0,1,2, 10,11,12, 20,21,22, 30,31,32]
        if _validate(A):
            return A
        return [3*i for i in range(12)]
    
    candidates.sort(key=lambda t: t[1])
    best = candidates[0][0]
    
    if not _validate(best):
        return [3*i for i in range(12)]
    
    return best
