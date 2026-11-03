import time
import itertools
from math import log

def _score(U):
    if not U or max(U) == 0:
        return 0.0
    S = set()
    D = set()
    for a in U:
        for b in U:
            S.add(a + b)
            D.add(a - b)
    nS = len(S)
    nD = len(D)
    denom = log(2 * max(U) + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + log(nD / nS) / denom

def _local_search(U, max_val, iterations=100, seed=42):
    import random
    rng = random.Random(seed)
    U = list(U)
    best_score = _score(U)
    for _ in range(iterations):
        improved = False
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            # Try a few random replacements
            for _ in range(10):
                new_val = rng.randint(1, max_val)
                if new_val in U:
                    continue
                U[i] = new_val
                s = _score(U)
                if s > best_score:
                    best_score = s
                    improved = True
                else:
                    U[i] = old_val
                    break
            # Also try swapping with another element
            for j in range(i + 1, len(U)):
                if U[j] == 0:
                    continue
                U[i], U[j] = U[j], U[i]
                s = _score(U)
                if s > best_score:
                    best_score = s
                    improved = True
                else:
                    U[i], U[j] = U[j], U[i]
        if not improved:
            break
    return U, best_score

def construct():
    deadline = time.time() + 140
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Exhaustive search for small n and M
    for n in range(3, 7):
        for M in range(1, 25):
            if time.time() > deadline:
                break
            # Generate all combinations of n-1 elements from 1..M
            for comb in itertools.combinations(range(1, M + 1), n - 1):
                U = [0] + list(comb)
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_set = U
        if time.time() > deadline:
            break
            
    # Local search on the best set found
    if time.time() < deadline - 10:
        refined, rs = _local_search(best_set, max(best_set) * 2, iterations=200, seed=42)
        if rs > best_score:
            best_score = rs
            best_set = refined
            
    return best_set
