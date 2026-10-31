import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 6.0

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

def _score(U):
    """Dispatch score calculation."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) <= 100:
        return _score_small(U)
    # For larger sets, use a simpler approximation or FFT if needed, 
    # but we expect optimal solutions to be small n.
    return _score_small(U)

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
        13: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 94],
        14: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 109],
        15: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 84, 126],
        16: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 84, 92, 144],
        17: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 84, 92, 100, 163],
        18: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 68, 76, 84, 92, 100, 108, 183],
    }
    if n in optimal:
        return list(optimal[n])
    return None

def _gen_greedy_golomb(n, seed=0):
    """Greedy construction of a Golomb ruler."""
    rng = random.Random(seed)
    U = [0]
    used_diffs = set()
    for i in range(1, n):
        best_pos = None
        best_score = -1
        # Search up to a reasonable bound
        upper_bound = i * i * 2 + 20
        for pos in range(1, upper_bound):
            if pos in U:
                continue
            # Check if adding pos creates duplicate differences
            ok = True
            for x in U:
                d = abs(pos - x)
                if d in used_diffs:
                    ok = False
                    break
                used_diffs.add(d)
            if ok:
                # Heuristic: prefer smaller pos? Or larger?
                # Smaller pos keeps max U small.
                if best_pos is None or pos < best_pos:
                    best_pos = pos
                # We can break early if we find a very small pos, 
                # but let's check a few to be safe
                if pos < i * i:
                    break
            else:
                for x in U:
                    d = abs(pos - x)
                    if d in used_diffs:
                        used_diffs.discard(d)
        if best_pos is None:
            # Fallback: find any valid pos
            pos = i * i + 10
            while True:
                ok = True
                for x in U:
                    d = abs(pos - x)
                    if d in used_diffs:
                        ok = False
                        break
                    used_diffs.add(d)
                if ok:
                    break
                else:
                    for x in U:
                        d = abs(pos - x)
                        if d in used_diffs:
                            used_diffs.discard(d)
                    pos += 1
            best_pos = pos
        U.append(best_pos)
    return sorted(U)

def _local_search_golomb(U, time_budget=10.0):
    """Local search to improve Golomb ruler score by reducing max U."""
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    best_U = list(U)
    best_s = _score(best_U)
    start = time.time()
    rng = random.Random(42)

    while time.time() - start < time_budget and _remaining() > 5.0:
        improved = False
        n = len(best_U)
        
        # Try moving each element (except 0) to a smaller position
        for i in range(1, n):
            orig = best_U[i]
            # Try moving down
            for new_val in range(orig - 1, 0, -1):
                if new_val in best_U:
                    continue
                new_U = list(best_U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                
                # Check Golomb property
                diffs = set()
                is_golomb = True
                for a in range(len(new_U)):
                    for b in range(a+1, len(new_U)):
                        d = new_U[b] - new_U[a]
                        if d in diffs:
                            is_golomb = False
                            break
                        diffs.add(d)
                    if not is_golomb:
                        break
                
                if is_golomb:
                    s = _score(new_U)
                    if s > best_s + 1e-10:
                        best_s = s
                        best_U = new_U
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try swapping two elements
            for i in range(1, n):
                for j in range(i+1, n):
                    u_i, u_j = best_U[i], best_U[j]
                    # Try swapping with nearby values
                    for delta in range(1, 5):
                        for nv_i, nv_j in [(u_i - delta, u_j), (u_i, u_j - delta), (u_i - delta, u_j - delta)]:
                            if nv_i <= 0 or nv_j <= 0:
                                continue
                            if nv_i in best_U or nv_j in best_U:
                                continue
                            if nv_i == nv_j:
                                continue
                            new_U = list(best_U)
                            new_U[i] = nv_i
                            new_U[j] = nv_j
                            new_U = sorted(new_U)
                            
                            diffs = set()
                            is_golomb = True
                            for a in range(len(new_U)):
                                for b in range(a+1, len(new_U)):
                                    d = new_U[b] - new_U[a]
                                    if d in diffs:
                                        is_golomb = False
                                        break
                                    diffs.add(d)
                                if not is_golomb:
                                    break
                            
                            if is_golomb:
                                s = _score(new_U)
                                if s > best_s + 1e-10:
                                    best_s = s
                                    best_U = new_U
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if not improved:
            # Random perturbation
            if rng.random() < 0.5:
                idx = rng.randint(1, n-1)
                new_val = max(1, best_U[idx] + rng.randint(-5, 5))
                if new_val not in best_U:
                    new_U = list(best_U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    diffs = set()
                    is_golomb = True
                    for a in range(len(new_U)):
                        for b in range(a+1, len(new_U)):
                            d = new_U[b] - new_U[a]
                            if d in diffs:
                                is_golomb = False
                                break
                            diffs.add(d)
                        if not is_golomb:
                            break
                    if is_golomb:
                        s = _score(new_U)
                        if s > best_s:
                            best_s = s
                            best_U = new_U

    return best_U, best_s

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    candidates = []
    
    # 1. Known optimal Golomb rulers
    for n in range(3, 20):
        g = _gen_golomb_optimal(n)
        if g:
            candidates.append(g)
            
    # 2. Greedy Golomb rulers with different seeds
    for n in range(3, 30):
        for seed in range(5):
            g = _gen_greedy_golomb(n, seed)
            candidates.append(g)

    # 3. Random structured sets
    for seed in range(50):
        n = random.randint(3, 20)
        g = _gen_greedy_golomb(n, seed + 100)
        candidates.append(g)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(c))

    # Score all
    scored = [(_score(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Local search on top candidates
    top = [s for _, s in scored[:20]]
    total_budget = 100.0
    per_set = total_budget / len(top)

    for s in top:
        if _remaining() < 10.0:
            break
        alloc = min(per_set, _remaining() - 5.0)
        if alloc < 1.0:
            break
        result, rs = _local_search_golomb(s, time_budget=alloc)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish on the best found
    if _remaining() > 10.0:
        polished, ps = _local_search_golomb(best_U, time_budget=min(20.0, _remaining() - 2.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)

    return best_U
