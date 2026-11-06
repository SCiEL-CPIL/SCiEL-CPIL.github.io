"""Rewritten: max-coverage set packing via reverse greedy + local search."""
import time
import numpy as np

_BASE = 51975
_DEADLINE = 120.0  # safety margin under the 150s kill
# 48 odd divisors of 51975 > 1, ascending
_DIVS = [3, 5, 7, 9, 11, 15, 21, 25, 27, 33, 35, 45, 55, 63, 75, 77, 99, 105,
         135, 165, 175, 189, 225, 231, 275, 315, 385, 405, 495, 525, 567, 675,
         693, 945, 1155, 1375, 1575, 1845, 2205, 2625, 2970, 3465, 4725, 5445,
         7775, 8925, 10890, 15750, 26250, 51975]
# 48 residue classes per modulus (index i -> class i)
_CLASSES = [np.arange(m, dtype=np.int64) for m in _DIVS]


def _score(cover):
    """Number of residues covered by at least one class (higher is better)."""
    return int(np.count_nonzero(cover))


def _greedy_baseline():
    """Reverse greedy: large moduli first (sparse), then small to fill gaps."""
    cover = np.zeros(_BASE, dtype=np.int8)
    chosen = [None] * len(_DIVS)
    for i in range(len(_DIVS) - 1, -1, -1):
        m = _DIVS[i]
        classes = _CLASSES[i]
        # marginal gain of each residue class = number of currently-uncovered hits
        gain = np.zeros(m, dtype=np.int64)
        for r in range(m):
            gain[r] = np.count_nonzero(cover[r::m] == 0)
        best_r = int(np.argmax(gain))
        chosen[i] = best_r
        cover[best_r::m] = 1
    return cover, chosen


def _local_search(cover, chosen, budget):
    """Single-class reassignment (max marginal gain), then 2-swap polish."""
    n = len(_DIVS)
    cur = _score(cover)
    while time.time() < budget:
        improved = False
        # Pass 1: reassign each class to its best residue (single move)
        for i in range(n):
            m = _DIVS[i]
            r_old = chosen[i]
            classes = _CLASSES[i]
            gain = np.zeros(m, dtype=np.int64)
            for r in range(m):
                gain[r] = np.count_nonzero(cover[r::m] == 0)
            best_r = int(np.argmax(gain))
            if best_r != r_old:
                cover[r_old::m] = 0
                cover[best_r::m] = 1
                chosen[i] = best_r
                if _score(cover) > cur:
                    cur = _score(cover)
                    improved = True
        # Pass 2: 2-swap polish (reassign pair i,j to jointly best residues)
        for i in range(n):
            if time.time() >= budget:
                break
            for j in range(i + 1, n):
                if time.time() >= budget:
                    break
                mi, mj = _DIVS[i], _DIVS[j]
                ri, rj = chosen[i], chosen[j]
                # best single residue for i (given j at rj)
                gi = np.zeros(mi, dtype=np.int64)
                for r in range(mi):
                    gi[r] = np.count_nonzero(cover[r::mi] == 0)
                bi = int(np.argmax(gi))
                # best single residue for j (given i at bi)
                gj = np.zeros(mj, dtype=np.int64)
                for r in range(mj):
                    gj[r] = np.count_nonzero(cover[r::mj] == 0)
                bj = int(np.argmax(gj))
                if bi != ri or bj != rj:
                    cover[ri::mi] = 0
                    cover[rj::mj] = 0
                    cover[bi::mi] = 1
                    cover[bj::mj] = 1
                    chosen[i], chosen[j] = bi, bj
                    if _score(cover) > cur:
                        cur = _score(cover)
                        improved = True
        if not improved:
            break
    return cur


def construct():
    t0 = time.time()
    budget = t0 + _DEADLINE - 10.0  # 10s safety margin
    cover, chosen = _greedy_baseline()
    _local_search(cover, chosen, budget)
    return [[_DIVS[i], chosen[i]] for i in range(len(_DIVS))]
