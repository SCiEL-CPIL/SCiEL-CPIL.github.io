import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

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
            deltas = [-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10]
            rng.shuffle(deltas)
            for delta in deltas:
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
            if rng.random() < 0.5:
                # Try inserting near existing elements or in gaps
                candidates_pos = []
                for j in range(1, len(best_U)):
                    mid = (best_U[j-1] + best_U[j]) // 2
                    if mid > best_U[j-1] and mid < best_U[j]:
                        candidates_pos.extend([mid - 1, mid, mid + 1])
                # Also try near the end
                if len(best_U) > 1:
                    last = best_U[-1]
                    candidates_pos.extend([last + 1, last + 2, last + 3])
                # And near the beginning
                if best_U[0] == 0 and len(best_U) > 1:
                    candidates_pos.extend([1, 2, 3])
                
                rng.shuffle(candidates_pos)
                for pos in candidates_pos:
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
            if rng.random() < 0.3:
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
            if rng.random() < 0.3:
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
            
    # 2. Add random variations of small Golomb rulers
    for n in range(3, 12):
        base = _gen_golomb_optimal(n)
        if base:
            for _ in range(8):
                perturbed = list(base)
                if len(perturbed) > 2:
                    # Random shift of one element
                    idx = random.randint(1, len(perturbed) - 1)
                    delta = random.randint(-3, 3)
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
    top_candidates = [s for _, s in scored[:8]]
    
    # Allocate time: 8 candidates * 8s each = 64s, plus 20s polish = 84s total
    per_set_time = 8.0
    
    for s in top_candidates:
        if _remaining() < 10.0:
            break
        result, rs = _local_search(s, time_budget=per_set_time)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish on the best found with larger budget
    if _remaining() > 15.0:
        polished, ps = _local_search(best_U, time_budget=min(25.0, _remaining() - 3.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
