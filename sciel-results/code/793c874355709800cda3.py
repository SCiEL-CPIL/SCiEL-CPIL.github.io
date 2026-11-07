"""Large sets without a triple and all its pair sums (1..128)."""
import time
import random
from itertools import combinations

_N = 128
_GAP = tuple(range(54, 74))          # 54..73 inclusive (20 elements)
_GAP_SET = set(_GAP)
_SMALLS = list(range(17, 54))        # 17..53 (37 candidates)
_BIGS = list(range(74, 129))         # 74..128 (55 candidates)
_DEADLINE = 52.0                      # safety margin under the 90 s kill


def _has_bad_triple(s):
    """True iff some three distinct members have all three pair-sums in s."""
    m = s
    for a, b, c in combinations(m, 3):
        if (a + b) in m and (a + c) in m and (b + c) in m:
            return True
    return False


def _greedy_add(s, cands, t0):
    """Add any candidate that keeps the set valid, repeatedly until none fit."""
    changed = True
    while changed:
        changed = False
        for x in cands:
            if time.monotonic() - t0 > _DEADLINE:
                return s
            if x in s:
                continue
            trial = s | {x}
            if not _has_bad_triple(trial):
                s = trial
                changed = True
        if not changed:
            break
    return s


def _local_search(s, cands, t0, iters=3000):
    """Swap-based local search: replace one selected element with an unselected
    candidate if the new set is valid. Improves size when swaps free up room."""
    improved = True
    n_iter = 0
    while improved and n_iter < iters:
        improved = False
        n_iter += 1
        if time.monotonic() - t0 > _DEADLINE:
            break
        sel = sorted(s)
        for i in range(len(sel)):
            for x in cands:
                if x in s:
                    continue
                trial = (s - {sel[i]}) | {x}
                if not _has_bad_triple(trial):
                    s = trial
                    improved = True
                    break
            if improved:
                break
    return s


def construct():
    t0 = time.monotonic()
    best = set()

    # Strong structural seed: all bigs + a few safe smalls.
    seed_small = [17, 18, 19, 20, 21, 22]
    base = set(_BIGS) | set(seed_small)
    if not _has_bad_triple(base):
        best = set(base)

    # Greedy add over all non-gap candidates.
    cands = [x for x in range(1, _N + 1) if x not in _GAP_SET]
    cur = _greedy_add(set(base), cands, t0)
    if len(cur) > len(best):
        best = cur

    # Local search (swap) to refine.
    cur = _local_search(set(cur), cands, t0, iters=4000)
    if len(cur) > len(best):
        best = cur

    # A few randomized restarts to escape local optima, within the deadline.
    rnd = random.Random(20240517)
    for _ in range(40):
        if time.monotonic() - t0 > _DEADLINE:
            break
        rs = set(_BIGS)
        # Random subset of smalls.
        for x in _SMALLS:
            if rnd.random() < 0.35:
                rs.add(x)
        if _has_bad_triple(rs):
            continue
        rs = _greedy_add(rs, cands, t0)
        rs = _local_search(rs, cands, t0, iters=1500)
        if len(rs) > len(best):
            best = rs

    if time.monotonic() - t0 > _DEADLINE:
        pass  # already stopped inside loops

    result = [int(x) for x in sorted(best)]
    # Final safety: ensure validity; if somehow broken, fall back to known-good.
    if _has_bad_triple(set(result)):
        result = [int(x) for x in sorted(set(_BIGS) | set(seed_small))]
    return result
