"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from functools import lru_cache

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0  # safety margin in seconds

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _sumset_diffset(U):
    """Return (|U+U|, |U-U|, maxU) for a set U of nonnegative ints."""
    U = list(U)
    n = len(U)
    if n == 0:
        return 0, 0, 0
    maxU = max(U)
    # difference set
    diffs = set()
    sums = set()
    for a in U:
        for b in U:
            diffs.add(a - b)
            sums.add(a + b)
    return len(sums), len(diffs), maxU

def _score(U):
    """Compute the lower bound score for a set U (must contain 0)."""
    if 0 not in U:
        U = set(U) | {0}
    sums, diffs, maxU = _sumset_diffset(U)
    if maxU <= 0:
        return 0.0
    if sums == 0:
        return 0.0
    import math
    if diffs <= sums:
        # log ratio <= 0, score <= 1
        return 1.0 + math.log(diffs / sums) / math.log(2 * maxU + 1)
    return 1.0 + math.log(diffs / sums) / math.log(2 * maxU + 1)

def _geometric_set(base, k):
    """{base^0, base^1, ..., base^(k-1)} union {0}."""
    s = {0}
    v = 1
    for _ in range(k):
        s.add(v)
        v *= base
    return s

def _arithmetic_set(a, d, k):
    """{a + i*d for i in range(k)} union {0}."""
    s = {0}
    for i in range(k):
        s.add(a + i * d)
    return s

def _symmetric_core(M, S):
    """U = {0} U {x, M-x for x in S}."""
    s = {0}
    for x in S:
        s.add(x)
        s.add(M - x)
    return s

def _local_search(U, max_iter=2000, max_add=40):
    """Greedily add/remove elements to improve score."""
    U = set(U)
    if 0 not in U:
        U.add(0)
    best = _score(U)
    best_set = set(U)
    for it in range(max_iter):
        if _remaining() < 0:
            break
        improved = False
        # Try adding a candidate element
        maxU = max(U) if U else 0
        # candidate pool: small numbers, geometric extensions, near existing
        cands = set()
        cands.add(maxU + 1)
        cands.add(maxU * 2 + 1)
        cands.add(maxU * 3 + 1)
        cands.add(maxU * 2)
        cands.add(maxU + maxU // 2)
        for x in U:
            cands.add(x + 1)
            cands.add(x * 2 + 1)
            cands.add(x + maxU // 3)
        # small numbers
        for v in range(1, min(maxU + 1, 50)):
            cands.add(v)
        # cap candidate set
        cands = list(cands)
        random.shuffle(cands)
        cands = cands[:max_add]
        best_cand = None
        best_cand_score = best
        for c in cands:
            if c in U:
                continue
            newU = U | {c}
            s = _score(newU)
            if s > best_cand_score + 1e-12:
                best_cand_score = s
                best_cand = c
        if best_cand is not None:
            U = U | {best_cand}
            best = best_cand_score
            best_set = set(U)
            improved = True
        # Try removing an element (not 0)
        if not improved:
            removable = [x for x in U if x != 0]
            random.shuffle(removable)
            for r in removable[:max_add]:
                newU = U - {r}
                s = _score(newU)
                if s > best + 1e-12:
                    U = newU
                    best = s
                    best_set = set(U)
                    improved = True
                    break
        if not improved:
            # small perturbation: swap
            removable = [x for x in U if x != 0]
            if removable and _remaining() > 0:
                r = random.choice(removable)
                c = random.choice(list(cands)) if cands else maxU + 1
                if c not in U:
                    newU = (U - {r}) | {c}
                    s = _score(newU)
                    if s > best + 1e-12:
                        U = newU
                        best = s
                        best_set = set(U)
                        improved = True
            if not improved:
                break
    return best_set, best

def _try_families():
    """Generate a variety of initial sets from different families."""
    inits = []
    # geometric
    for base in [2, 3, 4, 5, 7, 10, 16]:
        for k in range(2, 9):
            inits.append(_geometric_set(base, k))
    # arithmetic
    for a in [1, 2, 3]:
        for d in [1, 2, 3, 5, 8]:
            for k in range(2, 12):
                inits.append(_arithmetic_set(a, d, k))
    # symmetric core
    for M in [10, 20, 30, 50, 100]:
        for r in range(1, 6):
            S = set(random.sample(range(1, M), min(r, M - 1)))
            inits.append(_symmetric_core(M, S))
    # mixed: geometric with small extra
    s = _geometric_set(3, 6)
    inits.append(s)
    s = _geometric_set(4, 5)
    inits.append(s)
    return inits

def construct():
    rng = random.Random(12345)
    # seed initial best
    best_set = {0, 1, 3}
    best_score = _score(best_set)

    # Try a few deterministic strong starts
    starts = [
        {0, 1, 2, 4, 8, 16},
        {0, 1, 3, 9, 27, 81},
        {0, 1, 2, 4, 8, 16, 32, 64},
        {0, 1, 3, 9, 27},
        {0, 1, 2, 5, 10, 20},
    ]
    for s in starts:
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_set = set(s)

    # Local search from several initial points
    inits = _try_families()
    # score them quickly, keep top few
    scored = []
    for s in inits:
        if _remaining() < 0:
            break
        sc = _score(s)
        scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])

    # local search from top initial points and from current best
    to_search = [best_set] + [s for _, s in scored[:6]]

    for init in to_search:
        if _remaining() < 1.0:
            break
        res_set, res_score = _local_search(init, max_iter=1500, max_add=30)
        if res_score > best_score:
            best_score = res_score
            best_set = res_set

    # Final polish: a couple more local-search passes on the best
    for _ in range(3):
        if _remaining() < 1.0:
            break
        res_set, res_score = _local_search(best_set, max_iter=800, max_add=20)
        if res_score > best_score:
            best_score = res_score
            best_set = res_set

    # Return sorted list containing 0
    result = sorted(best_set)
    if 0 not in result:
        result = [0] + result
    return result

# EVOLVE-BLOCK-END
