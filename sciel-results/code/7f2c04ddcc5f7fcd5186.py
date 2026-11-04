"""Ternary covering code K_3(6,1): greedy set-cover + time-guarded add-then-remove local search."""
import time
import random
import numpy as np


def _int_to_word(x, n, q):
    w = np.empty(n, dtype=np.int8)
    for j in range(n - 1, -1, -1):
        w[j] = x % q
        x //= q
    return w


def _tetracode(q, n):
    if n < 4:
        return []
    return [
        [a, b, (a + b) % 3, (a + 2 * b) % 3]
        for a in range(3)
        for b in range(3)
    ]


def _ball_masks(n, q, radius):
    """ball_masks[i] = 1D bool array (size q**n) marking words within `radius` of word i."""
    total = q ** n
    ball_masks = np.zeros(total, dtype=bool)
    for i in range(total):
        c = _int_to_word(i, n, q)
        for r in range(radius + 1):
            for idx in np.ndindex(*([q] * r)):
                for perm in np.ndindex(*([n - r] * r)):
                    pos = np.array(perm, dtype=int)
                    w = c.copy()
                    for k in range(r):
                        w[pos[k]] = (c[pos[k]] + 1 + idx[k]) % q
                    x = 0
                    for j in range(n):
                        x = x * q + w[j]
                    ball_masks[x] = True
    return ball_masks


def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = t0 + 118.0
    total = q ** n

    ball_masks = _ball_masks(n, q, radius)
    if total == 0:
        return []

    # ---- Greedy set cover (vectorized) ----
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    for _ in range(total):
        if not uncovered.any():
            break
        cov = (ball_masks & uncovered).sum(axis=1)
        best = int(np.argmax(cov))
        if cov[best] == 0:
            break
        chosen.append(best)
        uncovered[ball_masks[best]] = False
        if time.time() > DEADLINE:
            break

    if not uncovered.any():
        return [_int_to_word(x, n, q).tolist() for x in chosen]

    # ---- Local search: add-then-remove heuristic ----
    covered = np.zeros(total, dtype=bool)
    for x in chosen:
        covered[ball_masks[x]] = True
    covcount = np.zeros(total, dtype=np.int32)
    for x in chosen:
        covcount[ball_masks[x]] += 1

    rng = random.Random(12345)
    best_len = len(chosen)
    best_chosen = list(chosen)

    while time.time() < DEADLINE:
        improved = False
        # Add phase: repeatedly add the word that covers the most still-uncovered.
        while True:
            if time.time() > DEADLINE:
                break
            if not uncovered.any():
                break
            cov = (ball_masks & uncovered).sum(axis=1)
            cand = int(np.argmax(cov))
            if cov[cand] == 0:
                break
            chosen.append(cand)
            uncovered[ball_masks[cand]] = False
            covered[ball_masks[cand]] = True
            covcount[ball_masks[cand]] += 1
            improved = True
        if not improved:
            break
        # Remove phase: drop any chosen word whose removal keeps full coverage.
        while True:
            if time.time() > DEADLINE:
                break
            removed_any = False
            for i in range(len(chosen) - 1, -1, -1):
                if covcount[ball_masks[chosen[i]]] > 1:
                    covcount[ball_masks[chosen[i]]] -= 1
                    covered[ball_masks[chosen[i]]] = False
                    chosen.pop(i)
                    removed_any = True
            if not removed_any:
                break
        # Update uncovered from coverage map.
        uncovered = ~covered
        if len(chosen) < best_len:
            best_len = len(chosen)
            best_chosen = list(chosen)
            if best_len <= 71:
                break

    out = [_int_to_word(x, n, q).tolist() for x in best_chosen]
    # Deduplicate (defensive).
    seen = set()
    res = []
    for w in out:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            res.append(w)
    return res
