import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin vs 150s kill

def _score_set(U):
    U = np.asarray(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0:
        return 0.0
    if U[0] != 0:
        U = np.concatenate(([0], U))
        n = len(U)
    M = int(U[-1])
    if M < 1:
        return 0.0
    # indicator vector of U
    vec = np.zeros(M + 1, dtype=np.float64)
    vec[U] = 1.0
    f = np.fft.rfft(vec, n=2 * (M + 1))
    # sumset: |U+U|
    s = np.fft.irfft(f * f, n=2 * (M + 1))
    sumset = int(np.sum(s >= 0.5))
    # difference set: |U-U|
    g = np.fft.irfft(f * np.conj(f), n=2 * (M + 1))
    diffset = int(np.sum(g >= 0.5))
    ratio = diffset / sumset if sumset > 0 else 0.0
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def _greedy_build():
    U = {0}
    Uarr = np.array([0], dtype=np.int64)
    for _ in range(400):
        if time.time() > _DEADLINE - 5:
            break
        best_val = _score_set(Uarr)
        best_add = None
        # candidate elements to add
        cur_max = Uarr[-1]
        cands = set()
        for _ in range(8):
            cands.add(random.randint(1, max(2, cur_max * 2 + 1)))
        for x in range(1, max(2, cur_max + 2)):
            cands.add(x)
        cands.add(cur_max * 2 + 1)
        cands.add(cur_max + random.randint(1, 5))
        for c in cands:
            if c in U:
                continue
            trial = np.sort(np.concatenate([Uarr, [c]]))
            v = _score_set(trial)
            if v > best_val + 1e-12:
                best_val = v
                best_add = c
                best_arr = trial
        if best_add is not None:
            U.add(best_add)
            Uarr = best_arr
    return list(Uarr)

def _local_search(U, budget):
    Uarr = np.asarray(sorted(U), dtype=np.int64)
    best = _score_set(Uarr)
    cur = Uarr
    while time.time() < budget:
        improved = False
        cur_max = int(cur[-1])
        # removal candidates
        for idx in range(1, len(cur)):
            if time.time() > budget:
                break
            trial = np.delete(cur, idx)
            v = _score_set(trial)
            if v > best + 1e-12:
                best = v
                cur = trial
                improved = True
                break
        if improved:
            continue
        # addition candidates
        for _ in range(6):
            if time.time() > budget:
                break
            c = random.randint(1, max(2, cur_max * 2 + 1))
            if c in set(cur.tolist()):
                continue
            trial = np.sort(np.concatenate([cur, [c]]))
            v = _score_set(trial)
            if v > best + 1e-12:
                best = v
                cur = trial
                improved = True
        if improved:
            continue
        # swap candidates
        for _ in range(6):
            if time.time() > budget:
                break
            if len(cur) < 2:
                break
            idx = random.randint(1, len(cur) - 1)
            c = random.randint(1, max(2, cur_max * 2 + 1))
            if c in set(cur.tolist()):
                continue
            trial = np.delete(cur, idx)
            trial = np.sort(np.concatenate([trial, [c]]))
            v = _score_set(trial)
            if v > best + 1e-12:
                best = v
                cur = trial
                improved = True
        if not improved:
            # perturb: drop a random element to escape plateau
            if len(cur) > 2:
                idx = random.randint(1, len(cur) - 1)
                cur = np.delete(cur, idx)
                best = _score_set(cur)
            else:
                break
    return list(cur), best

def construct():
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + 140.0

    best_U = [0, 1, 3]
    best_score = _score_set(np.array(best_U, dtype=np.int64))

    # phase 1: greedy build
    if time.time() < _DEADLINE - 10:
        gU = _greedy_build()
        gs = _score_set(np.array(gU, dtype=np.int64))
        if gs > best_score:
            best_U, best_score = gU, gs

    # phase 2: local search on best
    if time.time() < _DEADLINE - 5:
        lsU, ls = _local_search(best_U, _DEADLINE - 5)
        if ls > best_score:
            best_U, best_score = lsU, ls

    # ensure 0 present
    if 0 not in best_U:
        best_U = [0] + best_U

    return best_U
