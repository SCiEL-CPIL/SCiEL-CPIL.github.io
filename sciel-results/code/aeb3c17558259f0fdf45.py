# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def _build_pool():
    """Pool of integer vectors with norm^2 in {1, 3}.
    Norm^2 = 1: the 11 unit vectors e_i.
    Norm^2 = 3: vectors with one coordinate +-2 and one coordinate +-1 (132 of them).
    For this pool, any two distinct vectors have integer dot product <= 1,
    so a subset with all pairwise dot <= 1 has min pairwise dist^2 >= 2*Nmax = 6 >= 6.
    """
    d = 11
    pool = []
    # Unit vectors (norm^2 = 1)
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        pool.append(v)
    # Norm^2 = 3 vectors: one +-2, one +-1
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for s2 in (-2, 2):
                for s1 in (-1, 1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s2
                    v[j] = s1
                    pool.append(v)
    return np.array(pool)

def _greedy(pool, order, max_dot):
    """Greedy: take vectors in 'order' if dot with all chosen <= max_dot.
    Returns (chosen_indices, chosen_matrix)."""
    n = len(pool)
    chosen = []
    for idx in order:
        v = pool[idx]
        ok = True
        for c in chosen:
            if np.dot(v, pool[c]) > max_dot:
                ok = False
                break
        if ok:
            chosen.append(idx)
    return chosen

def _try_swap(chosen, pool, max_dot, iters, rng):
    """Local search: try swapping one chosen vector for an unchosen one to gain size.
    Returns new chosen if an improving swap is found, else None."""
    chosen_set = set(chosen)
    chosen_arr = np.array(chosen)
    for _ in range(iters):
        # pick a random chosen index to remove
        ci = rng.randrange(len(chosen))
        # pick a random unchosen candidate
        if len(chosen) >= len(pool):
            break
        cand = rng.randrange(len(pool))
        if cand in chosen_set:
            continue
        # check cand compatible with all chosen except ci
        v = pool[cand]
        ok = True
        for k, c in enumerate(chosen):
            if k == ci:
                continue
            if np.dot(v, pool[c]) > max_dot:
                ok = False
                break
        if ok:
            # swap: remove ci, add cand  (size unchanged, but may enable more later)
            # Only accept if it allows adding at least one more vector afterwards.
            new_chosen = [c for k, c in enumerate(chosen) if k != ci] + [cand]
            # try to greedily add more
            improved = False
            new_set = set(new_chosen)
            for c in range(len(pool)):
                if c in new_set:
                    continue
                vv = pool[c]
                good = True
                for nc in new_chosen:
                    if np.dot(vv, pool[nc]) > max_dot:
                        good = False
                        break
                if good:
                    new_chosen.append(c)
                    new_set.add(c)
                    improved = True
            if len(new_chosen) > len(chosen):
                return new_chosen
    return None

def kissing_number11() -> np.ndarray:
    deadline = time.time() + 560  # safety margin before 630s
    pool = _build_pool()
    n = len(pool)
    Nmax = 3
    max_dot = Nmax // 2  # = 1

    best = []
    best_len = 0

    rng = random.Random(12345)

    # Precompute dot matrix for speed (n x n)
    D = pool @ pool.T

    def greedy_from_order(order):
        chosen = []
        for idx in order:
            ok = True
            for c in chosen:
                if D[idx, c] > max_dot:
                    ok = False
                    break
            if ok:
                chosen.append(idx)
        return chosen

    def try_swap2(chosen, iters):
        chosen_set = set(chosen)
        for _ in range(iters):
            ci = rng.randrange(len(chosen))
            cand = rng.randrange(n)
            if cand in chosen_set:
                continue
            ok = True
            for k, c in enumerate(chosen):
                if k == ci:
                    continue
                if D[cand, c] > max_dot:
                    ok = False
                    break
            if ok:
                new_chosen = [c for k, c in enumerate(chosen) if k != ci] + [cand]
                new_set = set(new_chosen)
                for c in range(n):
                    if c in new_set:
                        continue
                    good = True
                    for nc in new_chosen:
                        if D[c, nc] > max_dot:
                            good = False
                            break
                    if good:
                        new_chosen.append(c)
                        new_set.add(c)
                if len(new_chosen) > len(chosen):
                    return new_chosen
        return None

    # Phase 1: greedy with several orderings
    orderings = []
    # natural order
    orderings.append(list(range(n)))
    # random shuffles
    for _ in range(40):
        o = list(range(n))
        rng.shuffle(o)
        orderings.append(o)
    # sorted by degree (fewest conflicts first) - approximate by norm then index
    orderings.append(sorted(range(n), key=lambda i: (int(pool[i].max()), i)))

    for o in orderings:
        if time.time() > deadline:
            break
        chosen = greedy_from_order(o)
        if len(chosen) > best_len:
            best = chosen
            best_len = len(chosen)
        # local search to improve
        for _ in range(3):
            if time.time() > deadline:
                break
            improved = try_swap2(chosen, iters=200)
            if improved is not None:
                chosen = improved
                if len(chosen) > best_len:
                    best = chosen
                    best_len = len(chosen)

    # Phase 2: more aggressive local search from best
    if time.time() < deadline - 30:
        chosen = list(best)
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = try_swap2(chosen, iters=500)
            if improved is not None:
                chosen = improved
                if len(chosen) > best_len:
                    best = chosen
                    best_len = len(chosen)
            else:
                # perturb: random restart around current
                if rng.random() < 0.3:
                    o = list(range(n))
                    rng.shuffle(o)
                    # seed with current best prefix
                    chosen = greedy_from_order(o)
                    if len(chosen) < best_len:
                        chosen = list(best)

    # Finalize: convert best indices to array
    arr = np.array(pool[best], dtype=np.int64)
    arr = np.around(arr).astype(np.int64)
    return arr

# EVOLVE-BLOCK-END
