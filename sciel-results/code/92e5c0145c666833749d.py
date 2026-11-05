# EVOLVE-BLOCK-START
import time
import random

N = 192
K = 3
MAX_CARD = 48

def _check_add(c, S):
    """Fast check: does adding c to list S preserve the no-divides-3-sum property?"""
    n = len(S)
    # c must not divide sum of any 3 distinct elements of S
    for i in range(n):
        si = S[i]
        for j in range(i+1, n):
            s_ij = si + S[j]
            for k in range(j+1, n):
                if (s_ij + S[k]) % c == 0:
                    return False
    # for each a in S, no new triple (c,x,y) with x,y in S\{a} sums to a multiple of a
    for idx, a in enumerate(S):
        for i in range(n):
            if i == idx:
                continue
            xi = S[i]
            for j in range(n):
                if j == idx or j == i:
                    continue
                if (c + xi + S[j]) % a == 0:
                    return False
    return True

def _is_valid_set(S):
    n = len(S)
    for idx, a in enumerate(S):
        for i in range(n):
            if i == idx:
                continue
            si = S[i]
            for j in range(n):
                if j == idx or j == i:
                    continue
                sj = S[j]
                for k in range(n):
                    if k == idx or k == i or k == j:
                        continue
                    if (si + sj + S[k]) % a == 0:
                        return False
    return True

def construct():
    deadline = time.time() + 78.0
    rng = random.Random(7)

    # Phase 1: greedy seed, large to small (large numbers are safe divisors)
    S = []
    for x in range(N, 0, -1):
        if _check_add(x, S):
            S.append(x)

    best = list(S)
    best_size = len(S)
    if not _is_valid_set(best):
        best = []
        best_size = 0

    def excluded_set():
        s = set(S)
        return [x for x in range(1, N+1) if x not in s]

    no_progress = 0
    swap_count = 0
    while time.time() < deadline and no_progress < 600:
        excluded = excluded_set()
        progress = False
        for c in excluded:
            if _check_add(c, S):
                S.append(c)
                if len(S) > best_size:
                    best_size = len(S)
                    best = list(S)
                progress = True
                no_progress = 0
                break
        if progress:
            continue
        # Try a size-preserving swap to unlock further additions
        if swap_count < 8000:
            done = False
            for out in S:
                for c in excluded:
                    S2 = [v for v in S if v != out]
                    S2.append(c)
                    if _is_valid_set(S2):
                        S = S2
                        swap_count += 1
                        done = True
                        break
                if done:
                    break
            if done:
                no_progress = 0
            else:
                no_progress += 1
        else:
            no_progress += 1

    if _is_valid_set(S) and len(S) > best_size:
        best = list(S)

    if not _is_valid_set(best) or not best:
        # Fallback: rebuild greedy
        S2 = []
        for x in range(N, 0, -1):
            if _check_add(x, S2):
                S2.append(x)
        best = S2

    return best
# EVOLVE-BLOCK-END
