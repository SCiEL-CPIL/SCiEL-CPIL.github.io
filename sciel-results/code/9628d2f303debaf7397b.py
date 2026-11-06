"""Search for ambient 12-set minimizing largest Sidon subset size."""
import random
import time
from itertools import combinations

def _is_sidon(S):
    """Check if S is a Sidon set (all pair sums including diagonals are unique)."""
    n = len(S)
    seen = set()
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(A):
    """Find the maximum size of a Sidon subset of A. Exact for small A."""
    n = len(A)
    # Check sizes from n down to 1
    for k in range(n, 0, -1):
        for subset in combinations(A, k):
            if _is_sidon(subset):
                return k
    return 0

def _quick_max_sidon_size(A, best_so_far=12):
    """Quick check: can we find a Sidon subset of size >= best_so_far?
    Returns the max Sidon size found, or None if timeout."""
    n = len(A)
    # If we only need to check if max >= best_so_far, we can stop early
    for k in range(n, best_so_far - 1, -1):
        for subset in combinations(A, k):
            if _is_sidon(subset):
                return k
    # Check if best_so_far - 1 is achievable
    for subset in combinations(A, best_so_far - 1):
        if _is_sidon(subset):
            return best_so_far - 1
    return best_so_far - 2  # Conservative: at least this small

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def _validate(A):
        if len(A) != 12:
            return False
        if len(set(A)) != 12:
            return False
        if any(x < 0 or x > 128 for x in A):
            return False
        return True

    best_set = None
    best_size = 12  # We want to minimize this

    def _try_set(A):
        nonlocal best_set, best_size
        if not _validate(A):
            return
        ms = _max_sidon_size(A)
        if ms < best_size:
            best_size = ms
            best_set = A
            # If we found size 5 or less, that's likely optimal, stop early
            if best_size <= 5:
                return True
        return False

    # Strategy 1: Exhaustive search over small ranges
    # For range [0, R], number of 12-subsets is C(R+1, 12)
    # R=12: C(13,12)=13
    # R=15: C(16,12)=1820
    # R=18: C(19,12)=50388
    # R=20: C(21,12)=293930
    # R=22: C(23,12)=1352078
    
    for max_val in [12, 13, 14, 15, 16, 17, 18, 19, 20]:
        if time.time() > deadline - 15.0:
            break
        count = 0
        for subset in combinations(range(max_val + 1), 12):
            A = list(subset)
            _try_set(A)
            count += 1
            if count % 10000 == 0 and time.time() > deadline - 15.0:
                break
        if best_size <= 5:
            break

    # Strategy 2: Structured sets - triples at regular intervals
    for gap in range(6, 30):
        for offset in range(0, 129 - 3*gap - 2):
            if time.time() > deadline - 15.0:
                break
            A = []
            for k in range(4):
                base = offset + k * gap
                if base + 2 > 128:
                    break
                A.extend([base, base+1, base+2])
            if len(A) == 12 and len(set(A)) == 12:
                _try_set(A)
        if best_size <= 5:
            break

    # Strategy 3: Pairs at regular intervals
    for gap in range(4, 25):
        for offset in range(0, 129 - 6*gap - 1):
            if time.time() > deadline - 15.0:
                break
            A = []
            for k in range(6):
                base = offset + k * gap
                if base + 1 > 128:
                    break
                A.extend([base, base+1])
            if len(A) == 12 and len(set(A)) == 12:
                _try_set(A)
        if best_size <= 5:
            break

    # Strategy 4: Quadruples at regular intervals
    for gap in range(5, 20):
        for offset in range(0, 129 - 4*gap - 3):
            if time.time() > deadline - 15.0:
                break
            A = []
            for k in range(3):
                base = offset + k * gap
                if base + 3 > 128:
                    break
                A.extend([base, base+1, base+2, base+3])
            if len(A) == 12 and len(set(A)) == 12:
                _try_set(A)
        if best_size <= 5:
            break

    # Strategy 5: Mixed structures - some pairs, some triples
    for gap1 in range(6, 20):
        for gap2 in range(8, 25):
            for offset in range(0, 60):
                if time.time() > deadline - 15.0:
                    break
                A = []
                # 2 triples + 1 pair = 8 elements, need 4 more
                # Let's do: 4 pairs
                for k in range(4):
                    base = offset + k * gap1
                    if base + 1 > 128:
                        break
                    A.extend([base, base+1])
                # Add 4 more elements spread out
                for k in range(4):
                    base = offset + gap2 + k * gap2
                    if base > 128:
                        break
                    if base not in A:
                        A.append(base)
                if len(A) == 12 and len(set(A)) == 12:
                    _try_set(A)
            if best_size <= 5:
                break
        if best_size <= 5:
            break

    # Strategy 6: Random search with local improvement
    rng = random.Random(42)
    for _ in range(50000):
        if time.time() > deadline - 5.0:
            break
        A = sorted(rng.sample(range(0, 129), 12))
        _try_set(A)
        if best_size <= 5:
            break

    # Strategy 7: Local search - perturb best found set
    if best_set is not None:
        for _ in range(20000):
            if time.time() > deadline - 3.0:
                break
            A = list(best_set)
            # Randomly swap one element
            idx = rng.randint(0, 11)
            old_val = A[idx]
            new_val = rng.randint(0, 128)
            if new_val != old_val and new_val not in A:
                A[idx] = new_val
                A.sort()
                _try_set(A)
            if best_size <= 5:
                break

    # Fallback
    if best_set is None:
        best_set = [0,1,2, 10,11,12, 20,21,22, 30,31,32]
    
    if not _validate(best_set):
        best_set = [3*i for i in range(12)]
    
    return best_set
