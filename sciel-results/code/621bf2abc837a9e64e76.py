"""Search for ambient 12-set minimizing largest Sidon subset size."""
import random
import time
from itertools import combinations

_END = None  # set in construct

def _is_sidon(S):
    """Check if S is a Sidon set (all unordered pair sums including diagonals are unique)."""
    n = len(S)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(A, timeout=2.0):
    """Find the maximum size of a Sidon subset of A."""
    n = len(A)
    if n <= 1:
        return n
    if _is_sidon(A):
        return n
    # Try decreasing sizes from n-1 down
    for size in range(n - 1, 0, -1):
        if time.time() > timeout:
            return None
        # Use combinations to check all subsets of given size
        for combo in combinations(range(n), size):
            if time.time() > timeout:
                return None
            cand = [A[i] for i in combo]
            if _is_sidon(cand):
                return size
    return 1

def construct():
    global _END
    _END = time.time() + 85.0

    def _validate(A):
        if len(A) != 12:
            return False
        if len(set(A)) != 12:
            return False
        if any(x < 0 or x > 128 for x in A):
            return False
        return True

    # Strategy: generate sets with many sum collisions
    # Key insight: consecutive integers create collisions (e.g., 0+3 = 1+2)
    # Use structured sets: pairs of close numbers, arithmetic progressions, etc.
    
    best_set = None
    best_size = 12  # worst case
    
    rng = random.Random(42)
    
    # Generate candidate sets with various structures
    structures = []
    
    # Structure 1: Consecutive blocks
    for start in range(0, 120):
        for block_len in range(4, 13):
            if start + block_len <= 129:
                # Take consecutive numbers
                A = list(range(start, start + 12))
                if _validate(A):
                    structures.append(A)
    
    # Structure 2: Pairs of close numbers
    for seed in range(200):
        rng2 = random.Random(seed)
        A = []
        while len(A) < 12:
            base = rng2.randint(0, 120)
            for offset in range(3):
                if base + offset <= 128 and base + offset not in A:
                    A.append(base + offset)
                    if len(A) == 12:
                        break
        A = sorted(set(A))
        if len(A) == 12 and _validate(A):
            structures.append(A)
    
    # Structure 3: Random with bias toward clusters
    for seed in range(500):
        rng3 = random.Random(seed * 7 + 13)
        A = set()
        while len(A) < 12:
            # Pick a cluster center
            center = rng3.randint(0, 120)
            # Add 1-3 numbers near center
            for _ in range(rng3.randint(1, 3)):
                val = center + rng3.randint(-2, 2)
                if 0 <= val <= 128:
                    A.add(val)
        A = sorted(A)
        if len(A) == 12 and _validate(A):
            structures.append(A)
    
    # Structure 4: All even/odd in narrow range
    for start in range(0, 116, 2):
        A = list(range(start, start + 24, 2))[:12]
        if _validate(A):
            structures.append(A)
        A = list(range(start + 1, start + 24, 2))[:12]
        if _validate(A):
            structures.append(A)
    
    # Structure 5: Mix of small and large
    for seed in range(300):
        rng4 = random.Random(seed * 11 + 5)
        small = rng4.sample(range(0, 40), 6)
        large = rng4.sample(range(80, 129), 6)
        A = sorted(small + large)
        if _validate(A):
            structures.append(A)
    
    # Evaluate all structures
    for A in structures:
        if time.time() > _END - 3.0:
            break
        ms = _max_sidon_size(A, timeout=1.0)
        if ms is not None and ms < best_size:
            best_size = ms
            best_set = A
            if best_size <= 5:  # Good enough, stop early
                break
    
    # Random search if needed
    if best_set is None or best_size > 6:
        tries = 0
        while time.time() < _END - 2.0 and tries < 50000:
            tries += 1
            A = sorted(rng.sample(range(0, 129), 12))
            # Quick check: count sum collisions
            sums = {}
            for i in range(12):
                for j in range(i, 12):
                    s = A[i] + A[j]
                    sums[s] = sums.get(s, 0) + 1
            collisions = sum(1 for v in sums.values() if v > 1)
            if collisions >= 3:  # Many collisions likely means small Sidon subset
                ms = _max_sidon_size(A, timeout=0.5)
                if ms is not None and ms < best_size:
                    best_size = ms
                    best_set = A
                    if best_size <= 5:
                        break
    
    if best_set is None:
        return list(range(12))
    if not _validate(best_set):
        return list(range(12))
    return best_set

# EVOLVE-BLOCK-END
