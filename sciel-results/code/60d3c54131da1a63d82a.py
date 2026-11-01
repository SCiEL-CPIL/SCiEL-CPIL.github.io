import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 5.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_small(U):
    """Exact O(n^2) score for small sets."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom

def _is_golomb(U):
    """Check if U is a Golomb ruler (all pairwise differences are unique)."""
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _gen_golomb_optimal(n):
    """Return known optimal Golomb rulers for small n."""
    optimal = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 55],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 67],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 80],
    }
    if n in optimal:
        return list(optimal[n])
    return None

def _greedy_golomb(n, seed=42):
    """Greedy construction of a Golomb ruler with n marks."""
    rng = random.Random(seed)
    U = [0]
    diffs = set()
    for _ in range(n - 1):
        # Try to place the next mark
        best_pos = None
        # Start from just after the last mark
        start = U[-1] + 1 if U else 1
        # Try a range of positions
        for pos in range(start, start + 500):
            valid = True
            for existing in U:
                d = pos - existing
                if d in diffs:
                    valid = False
                    break
            if valid:
                best_pos = pos
                break
        if best_pos is None:
            # If we can't find one in the range, try harder
            for pos in range(start, start + 5000):
                valid = True
                for existing in U:
                    d = pos - existing
                    if d in diffs:
                        valid = False
                        break
                if valid:
                    best_pos = pos
                    break
        if best_pos is None:
            return None
        U.append(best_pos)
        for existing in U[:-1]:
            diffs.add(best_pos - existing)
    return U

def _bose_chowla(n, p=None):
    """Bose-Chowla construction using a prime p > n-1."""
    if p is None:
        # Find a prime p > n-1
        p = n
        while not _is_prime(p):
            p += 1
    # Use the multiplicative group mod p
    # Quadratic residues approach
    U = [0]
    used_diffs = set()
    for a in range(1, n):
        found = False
        for b in range(1, 1000):
            pos = (a * b) % p
            if pos == 0:
                continue
            # Check if adding pos maintains Golomb property
            valid = True
            for existing in U:
                d = abs(pos - existing)
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                U.append(pos)
                for existing in U[:-1]:
                    used_diffs.add(abs(pos - existing))
                found = True
                break
        if not found:
            return None
    return sorted(U)

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

def _local_search(U, time_budget=10.0):
    """
    Fast local search: try moving each element by various integers,
    plus structural moves (insert/delete).
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    best_U = list(U)
    best_s = _score_small(best_U)
    start = time.time()
    rng = random.Random(42)
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        improved = False
        n = len(best_U)
        
        # Move 1: shift existing elements
        indices = list(range(1, n))
        rng.shuffle(indices)
        for i in indices:
            if _remaining() < 2.0:
                break
            orig = best_U[i]
            # Try a range of shifts, weighted toward smaller ones
            deltas = list(range(-15, 16))
            rng.shuffle(deltas)
            for delta in deltas:
                if delta == 0:
                    continue
                new_val = orig + delta
                if new_val < 0 or new_val in best_U:
                    continue
                new_U = list(best_U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                if _is_golomb(new_U):
                    s = _score_small(new_U)
                    if s > best_s + 1e-10:
                        best_s = s
                        best_U = new_U
                        improved = True
                        break
            if improved:
                break
        
        # Move 2: insert a new element
        if not improved and _remaining() > 3.0:
            if rng.random() < 0.6:
                # Try inserting near existing elements or in gaps
                candidates_pos = []
                for j in range(1, len(best_U)):
                    gap = best_U[j] - best_U[j-1]
                    if gap > 2:
                        # Try positions in the gap
                        for k in range(1, min(gap-1, 5)):
                            candidates_pos.append(best_U[j-1] + k)
                # Also try near the end
                if len(best_U) > 1:
                    last = best_U[-1]
                    for k in range(1, 8):
                        candidates_pos.append(last + k)
                # And near the beginning
                if best_U[0] == 0 and len(best_U) > 1:
                    for k in range(1, 5):
                        candidates_pos.append(k)
                
                rng.shuffle(candidates_pos)
                for pos in candidates_pos[:30]:
                    if pos < 0 or pos in best_U:
                        continue
                    new_U = list(best_U)
                    new_U.append(pos)
                    new_U = sorted(new_U)
                    if _is_golomb(new_U):
                        s = _score_small(new_U)
                        if s > best_s + 1e-10:
                            best_s = s
                            best_U = new_U
                            improved = True
                            break
        
        # Move 3: delete an element (if n > 3)
        if not improved and n > 3 and _remaining() > 3.0:
            if rng.random() < 0.4:
                idx = rng.randint(1, n-1)
                new_U = [x for j, x in enumerate(best_U) if j != idx]
                if _is_golomb(new_U):
                    s = _score_small(new_U)
                    if s > best_s + 1e-10:
                        best_s = s
                        best_U = new_U
                        improved = True
        
        # Move 4: swap two elements
        if not improved and n > 3 and _remaining() > 3.0:
            if rng.random() < 0.4:
                i, j = rng.sample(range(1, n), 2)
                new_U = list(best_U)
                new_U[i], new_U[j] = new_U[j], new_U[i]
                new_U = sorted(new_U)
                if _is_golomb(new_U):
                    s = _score_small(new_U)
                    if s > best_s + 1e-10:
                        best_s = s
                        best_U = new_U
                        improved = True

    return best_U, best_s

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    candidates = []
    
    # 1. Known optimal Golomb rulers for small n
    for n in range(3, 13):
        g = _gen_golomb_optimal(n)
        if g:
            candidates.append(g)
            
    # 2. Greedy Golomb rulers for larger n
    for n in range(13, 20):
        for seed in range(5):
            g = _greedy_golomb(n, seed=seed)
            if g:
                candidates.append(g)
    
    # 3. Add random variations of small Golomb rulers
    for n in range(3, 12):
        base = _gen_golomb_optimal(n)
        if base:
            for _ in range(10):
                perturbed = list(base)
                if len(perturbed) > 2:
                    # Random shift of one element
                    idx = random.randint(1, len(perturbed) - 1)
                    delta = random.randint(-4, 4)
                    if delta != 0:
                        new_val = perturbed[idx] + delta
                        if new_val > 0 and new_val not in perturbed:
                            perturbed[idx] = new_val
                            perturbed = sorted(perturbed)
                            if _is_golomb(perturbed):
                                candidates.append(perturbed)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(c))

    # Score all initial candidates
    scored = [(_score_small(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    if not scored:
        return [0, 1]

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Local search on top candidates
    top_candidates = [s for _, s in scored[:10]]
    
    # Allocate time: 10 candidates * 8s each = 80s, plus 30s polish = 110s total
    per_set_time = 8.0
    
    for s in top_candidates:
        if _remaining() < 15.0:
            break
        result, rs = _local_search(s, time_budget=per_set_time)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish on the best found with larger budget
    if _remaining() > 20.0:
        polished, ps = _local_search(best_U, time_budget=min(35.0, _remaining() - 5.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
