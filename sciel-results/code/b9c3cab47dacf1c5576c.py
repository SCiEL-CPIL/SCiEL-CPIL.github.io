"""Fresh design: density-seeded SA with triple-swap moves over distinct 3-AP differences."""
import time
import random
import numpy as np

N = 32          # set size
UB = 512        # upper bound of the universe
DEADLINE = 145.0  # safety margin before the 150s kill


def _distinct_diffs(S):
    """Return set of d>0 with a, a+d, a+2d all in S for some a."""
    diffs = set()
    for i in range(N):
        a = S[i]
        for j in range(i + 1, N):
            b = S[j]
            d = b - a
            if d > 0 and (a + 2 * d) in S:
                diffs.add(d)
    return diffs


def _score(S):
    return len(_distinct_diffs(S))


def _greedy_seed(rng):
    """Pick 32 elements maximizing 3-AP coverage via greedy density selection."""
    S = []
    for _ in range(N):
        best_x, best_cov = None, -1
        for x in range(1, UB + 1):
            if x in S:
                continue
            # count APs that would be completed by adding x
            cov = 0
            for a in S:
                d = x - a
                if d > 0 and (a + 2 * d) in S:
                    cov += 1
            for b in S:
                d = b - x
                if d > 0 and (b + 2 * d) in S:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_x = x
        S.append(best_x)
    return sorted(S)


def _sa(S, rng, deadline):
    """Simulated annealing with triple-swap moves."""
    S = list(S)
    cur_score = _score(S)
    best = S[:]
    best_score = cur_score
    T = 10.0
    T_min = 0.05
    cooling = 0.9995
    n = len(S)
    S_set = set(S)

    while T > T_min:
        if time.time() > deadline:
            break
        # triple swap: pick 3 positions, replace with 3 new random values
        idx = rng.sample(range(n), 3)
        old_vals = [S[i] for i in idx]
        new_vals = []
        valid = True
        for _ in range(3):
            v = rng.randint(1, UB)
            if v in S_set and v not in old_vals:
                valid = False
                break
            new_vals.append(v)
        if not valid:
            continue
        # apply
        for i, v in zip(idx, new_vals):
            S_set.discard(S[i])
            S[i] = v
            S_set.add(v)
        S.sort()
        new_score = _score(S)
        delta = new_score - cur_score
        if delta >= 0 or rng.random() < np.exp(delta / T):
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best = S[:]
        else:
            # revert
            for i, v in zip(idx, old_vals):
                S_set.discard(S[i])
                S[i] = v
                S_set.add(v)
            S.sort()
        T *= cooling
    return best, best_score


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = random.Random(12345)
    best = None
    best_score = -1
    n_restarts = 0
    while time.time() < deadline:
        seed = _greedy_seed(rng)
        s, sc = _sa(seed, rng, deadline)
        if sc > best_score:
            best_score = sc
            best = s
        n_restarts += 1
    if best is None:
        best = list(range(1, 33))
    return best
