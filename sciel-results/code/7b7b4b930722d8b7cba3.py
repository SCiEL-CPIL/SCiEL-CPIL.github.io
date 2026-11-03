"""Construct a finite set U of nonnegative integers (containing 0) to maximize
   1 + log(|U-U|/|U+U|) / log(2*max(U)+1).
"""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

_DEADLINE = 110.0  # safety margin before 120s kill


def _sumset_size(U):
    """Return number of distinct sums a+b for a,b in U."""
    U = list(U)
    n = len(U)
    if n <= 1:
        return n
    if n > 100:
        arr = np.array(U, dtype=np.int64)
        sums = np.add.outer(arr, arr)
        return len(np.unique(sums))
    sums = set()
    for i in range(n):
        ai = U[i]
        for j in range(i, n):
            sums.add(ai + U[j])
    return len(sums)


def _diffset_size(U):
    """Return number of distinct differences a-b for a,b in U."""
    U = list(U)
    n = len(U)
    if n <= 1:
        return n
    if n > 100:
        arr = np.array(U, dtype=np.int64)
        diffs = np.subtract.outer(arr, arr)
        return len(np.unique(diffs))
    diffs = set()
    for i in range(n):
        ai = U[i]
        for j in range(n):
            diffs.add(ai - U[j])
    return len(diffs)


def _score(U):
    """Compute the task score for a set U (list of nonnegative ints, 0 in U)."""
    if not U or 0 not in U:
        return -1.0
    U = sorted(set(U))
    mx = U[-1]
    if mx == 0:
        return 1.0
    ss = _sumset_size(U)
    ds = _diffset_size(U)
    if ss == 0:
        return 1.0
    ratio = ds / ss
    if ratio <= 0:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)


def _geometric(base, k):
    """U = {0} U {base^1, base^2, ..., base^k}."""
    U = [0]
    v = 1
    for _ in range(k):
        v *= base
        U.append(v)
    return U


def _hybrid(ap_len, base, k):
    """U = AP block {0..ap_len-1} scaled, plus geometric tail."""
    U = list(range(ap_len))
    v = U[-1] if U else 0
    # tail starts above the AP block
    start = U[-1] * base if U else base
    for _ in range(k):
        start *= base
        U.append(start)
    return U


def _fibonacci_like(ratio, k):
    """U = {0, 1, ratio, ratio^2, ...} with ratio slightly > 1."""
    U = [0, 1]
    v = 1
    for _ in range(k):
        v = int(v * ratio + 0.5)
        if v == U[-1]:
            v += 1
        U.append(v)
    return U


def _mixed_bases(bases, k_each):
    """U = union of geometric sequences with different bases."""
    U = {0}
    for base in bases:
        v = 1
        for _ in range(k_each):
            v *= base
            U.add(v)
    return sorted(U)


def _greedy_random(n_target, seed, max_val=200000):
    """Greedily add elements to maximize score heuristic; randomized."""
    rng = random.Random(seed)
    U = [0]
    # start with a small random base element
    U.append(rng.randint(2, 30))
    U.sort()
    best = _score(U)
    while len(U) < n_target and time.time() < _DEADLINE - 5:
        # propose a new element: geometric-ish growth or random
        mx = U[-1]
        candidates = []
        # geometric candidate
        g = int(mx * rng.uniform(1.3, 2.5))
        candidates.append(g)
        # random candidate
        candidates.append(rng.randint(mx // 2 + 1, mx * 3 + 10))
        candidates.append(mx + rng.randint(1, max(2, mx // 10)))
        best_gain = None
        best_elem = None
        for c in candidates:
            if c <= mx:
                continue
            newU = U + [c]
            s = _score(newU)
            if best_gain is None or s > best_gain:
                best_gain = s
                best_elem = c
        if best_elem is None:
            break
        U.append(best_elem)
        U.sort()
        best = best_gain
    return U


def _local_search(U, max_iter=50):
    """Try removing or replacing elements to improve score."""
    U = sorted(set(U))
    if not U or 0 not in U:
        return U
    best_U = list(U)
    best_score = _score(U)
    rng = random.Random(42)
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 3:
            break
        improved = False
        
        # Try removing a non-zero element
        for i in range(1, len(U)):
            if time.time() > _DEADLINE - 3:
                break
            newU = U[:i] + U[i+1:]
            if not newU or 0 not in newU:
                continue
            s = _score(newU)
            if s > best_score + 1e-12:
                best_score = s
                best_U = newU
                U = newU
                improved = True
                break
        
        # Try replacing a non-zero element with a nearby value
        if not improved:
            for i in range(1, len(U)):
                if time.time() > _DEADLINE - 3:
                    break
                orig = U[i]
                # Try a few nearby values
                for delta in [1, 2, 3, -1, -2, -3]:
                    new_val = orig + delta
                    if new_val <= 0 or new_val in U:
                        continue
                    newU = U[:i] + [new_val] + U[i+1:]
                    newU = sorted(newU)
                    s = _score(newU)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = newU
                        U = newU
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_U


def construct():
    t0 = time.time()
    best_U = [0, 1, 3]
    best_score = _score(best_U)

    def consider(U):
        nonlocal best_U, best_score
        if time.time() > _DEADLINE - 3:
            return
        try:
            s = _score(U)
        except Exception:
            return
        if s > best_score:
            best_score = s
            best_U = list(U)

    # 1) Geometric sweep
    for base in range(2, 15):
        for k in range(2, 45):
            if time.time() > _DEADLINE - 3:
                break
            U = _geometric(base, k)
            consider(U)
        if time.time() > _DEADLINE - 3:
            break

    # 2) Hybrid sweep
    for ap_len in [1, 2, 3, 5, 8, 13]:
        for base in range(2, 10):
            for k in range(1, 30):
                if time.time() > _DEADLINE - 3:
                    break
                U = _hybrid(ap_len, base, k)
                consider(U)
            if time.time() > _DEADLINE - 3:
                break
        if time.time() > _DEADLINE - 3:
            break

    # 3) Fibonacci-like sequences
    for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
        for k in range(5, 30):
            if time.time() > _DEADLINE - 3:
                break
            U = _fibonacci_like(ratio, k)
            consider(U)
        if time.time() > _DEADLINE - 3:
            break

    # 4) Mixed bases
    base_sets = [
        [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [2, 3, 5], [2, 4, 7], [3, 5, 7]
    ]
    for bases in base_sets:
        for k_each in range(2, 20):
            if time.time() > _DEADLINE - 3:
                break
            U = _mixed_bases(bases, k_each)
            consider(U)
        if time.time() > _DEADLINE - 3:
            break

    # 5) Randomized greedy constructions
    seed = 0
    while time.time() < _DEADLINE - 8:
        for n_target in [8, 12, 15, 20, 25]:
            if time.time() > _DEADLINE - 8:
                break
            U = _greedy_random(n_target, seed)
            consider(U)
            seed += 1

    # 6) Local search on best found
    if time.time() < _DEADLINE - 3:
        refined = _local_search(best_U, max_iter=30)
        consider(refined)

    return best_U
# EVOLVE-BLOCK-END
