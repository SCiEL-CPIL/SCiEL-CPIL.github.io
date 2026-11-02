"""Sums and differences of finite sets — maximize sumset/difference-set exponent lower bound."""
import time
import math
import itertools
import random
from functools import lru_cache

# EVOLVE-BLOCK-START

def _score_of(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    # Compute U+U and U-U
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * maxU + 1, 2)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio, 2) / denom


def _exhaustive_small(n, max_val, deadline):
    """Exhaustive search over subsets of {0,...,max_val} of size n containing 0."""
    best = [0, 1, 3]
    best_score = _score_of(best)
    candidates = list(range(1, max_val + 1))
    count = 0
    for combo in itertools.combinations(candidates, n - 1):
        if time.time() > deadline:
            break
        U = sorted([0] + list(combo))
        s = _score_of(U)
        if s > best_score:
            best_score = s
            best = U
        count += 1
    return best, best_score, count


def _greedy_build(deadline):
    """Greedy construction: start with {0}, add elements that maximize score."""
    U = [0]
    best_score = 0.0
    # Try adding small elements first
    for trial in range(1, 50):
        if time.time() > deadline:
            break
        best_elem = None
        best_s = best_score
        for x in range(1, 60):
            if x in U:
                continue
            U2 = sorted(U + [x])
            s = _score_of(U2)
            if s > best_s:
                best_s = s
                best_elem = x
        if best_elem is None:
            break
        U.append(best_elem)
        U.sort()
        best_score = best_s
    return U, best_score


def _local_search(U, deadline, max_iter=200):
    """Local search: try swapping or adding/removing elements."""
    U = sorted(set(U))
    best_score = _score_of(U)
    best_U = U[:]
    rng = random.Random(42)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding an element
        if len(U) < 30:
            for x in range(1, 200):
                if x in U:
                    continue
                U2 = sorted(U + [x])
                s = _score_of(U2)
                if s > best_score + 1e-12:
                    U = U2
                    best_score = s
                    best_U = U2[:]
                    improved = True
                    break
        
        # Try removing an element (if set is large enough)
        if len(U) > 3:
            for i, x in enumerate(U):
                if x == 0:
                    continue
                U2 = U[:i] + U[i+1:]
                s = _score_of(U2)
                if s > best_score + 1e-12:
                    U = U2
                    best_score = s
                    best_U = U2[:]
                    improved = True
                    break
        
        # Try swapping an element
        if len(U) >= 3:
            for i, x in enumerate(U):
                if x == 0:
                    continue
                for y in range(1, 300):
                    if y in U:
                        continue
                    U2 = U[:i] + [y] + U[i+1:]
                    U2 = sorted(set(U2))
                    if len(U2) != len(U):
                        continue
                    s = _score_of(U2)
                    if s > best_score + 1e-12:
                        U = U2
                        best_score = s
                        best_U = U2[:]
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Random perturbation
            if len(U) > 2:
                idx = rng.randint(1, len(U) - 1)
                new_val = rng.randint(1, 400)
                if new_val not in U:
                    U2 = U[:idx] + [new_val] + U[idx+1:]
                    U2 = sorted(set(U2))
                    if len(U2) == len(U):
                        s = _score_of(U2)
                        if s > best_score - 0.01:  # allow small regressions
                            U = U2
                            best_score = max(best_score, s)
                            best_U = U2[:]
    
    return best_U, best_score


def _symmetric_core_search(deadline):
    """Search over symmetric core structure U = {0} ∪ {x, M-x | x in S}."""
    best_U = [0, 1, 3]
    best_score = _score_of(best_U)
    
    for M in range(4, 80):
        if time.time() > deadline:
            break
        # S is a subset of {1, ..., M//2}
        half = M // 2
        for size in range(1, min(6, half + 1)):
            if time.time() > deadline:
                break
            for S in itertools.combinations(range(1, half + 1), size):
                U = {0}
                for x in S:
                    U.add(x)
                    if M - x != 0:
                        U.add(M - x)
                U = sorted(U)
                if len(U) < 2:
                    continue
                s = _score_of(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    return best_U, best_score


def construct():
    start = time.time()
    # Safety margin: aim to finish by 140s (10s buffer before 150s kill)
    deadline = start + 140.0
    
    best_U = [0, 1, 3]
    best_score = _score_of(best_U)
    
    # Phase 1: Exhaustive search for small sets (n=3,4,5)
    for n in [3, 4, 5, 6]:
        max_val = 30 if n <= 4 else 20
        U, s, _ = _exhaustive_small(n, max_val, deadline)
        if s > best_score:
            best_score = s
            best_U = U[:]
        if time.time() > deadline - 30:
            break
    
    # Phase 2: Symmetric core search
    U, s = _symmetric_core_search(deadline - 20)
    if s > best_score:
        best_score = s
        best_U = U[:]
    
    # Phase 3: Local search from best found so far
    U, s = _local_search(best_U, deadline - 10, max_iter=500)
    if s > best_score:
        best_score = s
        best_U = U[:]
    
    # Phase 4: Greedy build from scratch
    U, s = _greedy_build(deadline - 5)
    if s > best_score:
        best_score = s
        best_U = U[:]
    
    # Final: local search from greedy result too
    U, s = _local_search(U, deadline - 1, max_iter=200)
    if s > best_score:
        best_score = s
        best_U = U[:]
    
    # Ensure 0 is in the set and all are nonnegative
    best_U = sorted(set(x for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U

# EVOLVE-BLOCK-END
