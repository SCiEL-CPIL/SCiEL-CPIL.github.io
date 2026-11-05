"""Iterated Local Search for low-discrepancy signs on all arithmetic progressions."""
import time
import random

N = 256
DEADLINE = 75.0  # safety margin under 90s kill

def max_discrepancy(signs):
    """Compute max absolute signed sum over any AP contained in [0, N).
    An AP is (start, step, length>=1) with all indices in range.
    For each step d, for each residue class r in [0, d), the APs are prefixes
    of the sequence signs[r], signs[r+d], signs[r+2d], ...
    So we just need max |prefix sum| for each such subsequence.
    """
    best = 0
    for d in range(1, N):
        for r in range(d):
            s = 0
            idx = r
            while idx < N:
                s += signs[idx]
                a = -s if s < 0 else s
                if a > best:
                    best = a
                idx += d
    return best

def _eval(signs):
    return max_discrepancy(signs)

def _flip(signs, i):
    signs[i] = -signs[i]

def _local_search(signs, start_time):
    """Hill-climb: repeatedly flip the single index that most reduces max discrepancy.
    Use fast incremental update: flipping index i only affects APs that contain i,
    which are exactly the (d, r) with r = i mod d for each d, and i is in that class.
    But simpler: just re-evaluate; N=256, eval is cheap enough.
    """
    cur = _eval(signs)
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 5:
        improved = False
        best_gain = 0
        best_i = -1
        # Try each single flip; keep the one that reduces cur the most.
        for i in range(N):
            _flip(signs, i)
            new = _eval(signs)
            _flip(signs, i)
            gain = cur - new
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i >= 0 and best_gain > 0:
            _flip(signs, best_i)
            cur -= best_gain
            improved = True
    return cur

def _perturb(signs, rng, k):
    """Flip k random distinct positions."""
    idxs = rng.sample(range(N), k)
    for i in idxs:
        _flip(signs, i)

def construct():
    start_time = time.time()
    rng = random.Random(2024)

    # Initial random solution
    signs = [rng.choice((-1, 1)) for _ in range(N)]
    best_signs = signs[:]
    best_val = _eval(best_signs)

    # Initial local search
    _local_search(signs, start_time)
    v = _eval(signs)
    if v < best_val:
        best_val = v
        best_signs = signs[:]

    # ILS loop
    while (time.time() - start_time) < DEADLINE - 8:
        # Perturb the current best
        signs = best_signs[:]
        k = rng.randint(3, 12)
        _perturb(signs, rng, k)
        _local_search(signs, start_time)
        v = _eval(signs)
        if v < best_val:
            best_val = v
            best_signs = signs[:]
            # Occasionally also try a larger perturbation from best
            if rng.random() < 0.3:
                signs2 = best_signs[:]
                _perturb(signs2, rng, rng.randint(15, 30))
                _local_search(signs2, start_time)
                v2 = _eval(signs2)
                if v2 < best_val:
                    best_val = v2
                    best_signs = signs2[:]

    return best_signs
