# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

_DEADLINE = 150.0
_DEADLINE_START = time.time()


def _remaining_budget():
    return _DEADLINE - (time.time() - _DEADLINE_START)


def construct(q, n, radius):
    global _DEADLINE_START
    _DEADLINE_START = time.time()
    rng = random.Random(12345)

    if n == 0:
        return [[]]
    if radius == 0:
        return [list(c) for c in itertools.product(range(q), repeat=n)]

    total = q ** n
    # precompute powers
    powq = [1]
    for _ in range(n):
        powq.append(powq[-1] * q)
    # powq[i] = q**i, so powq[n] = total

    # base digits for the full index space
    base = np.zeros(total, dtype=np.int8)
    for i in range(n):
        p = powq[i]
        base[:, i] = (np.arange(total) // p) % q
    base = np.ascontiguousarray(base)

    # build all candidate codewords (the whole space) as digit matrix
    cands = np.ascontiguousarray(base)  # (total, n)

    # radius-1 ball digit sets per coordinate
    def ball1_digits(i):
        d = np.empty((q, n), dtype=np.int8)
        d[:, :] = base[i]
        for j in range(n):
            for v in range(q):
                if v != base[i, j]:
                    d[q * j + v, j] = v
        return d

    # precompute ball1 for every index (memory: total*q*n ints)
    # For n=6,q=3: 729*3*6 = 13122 ints -> tiny
    ball1_all = np.empty((total, q, n), dtype=np.int8)
    for i in range(total):
        ball1_all[i] = ball1_digits(i)

    # radius-1 ball membership for each candidate index: (total, total) bool
    # ball[i, k] = True if k is within dist 1 of i
    ball1 = np.zeros((total, total), dtype=bool)
    # ball1[i,i] = True
    np.fill_diagonal(ball1, True)
    for i in range(total):
        for j in range(n):
            for v in range(q):
                if v != base[i, j]:
                    k = i + (v - base[i, j]) * powq[j]
                    ball1[i, k] = True

    # radius-2 ball membership (only needed if radius >= 2)
    if radius >= 2:
        # ball2[i, k] = dist(i,k) <= 2
        # compute via digit diff count
        # build pairwise diff count efficiently: for each coord, count matches
        # dist = n - matches
        # We can build matches matrix as sum over coords of (base[:,j] == base[k,j])
        matches = np.zeros((total, total), dtype=np.int8)
        for j in range(n):
            col = base[:, j]
            matches += (col[:, None] == col[None, :])
        ball2 = matches >= (n - radius)
        ball = ball2
    else:
        ball = ball1

    uncovered = np.ones(total, dtype=bool)

    # Greedy set cover
    chosen = []
    chosen_set = set()
    while uncovered.any():
        if _remaining_budget() < 5.0:
            break
        # gain = number of uncovered covered by each candidate
        gains = ball @ uncovered.astype(np.int8)
        # choose candidate with max gain, tie-break by min overlap (fewer total covered)
        # overlap = row sum of ball
        overlaps = ball.sum(axis=1)
        # combined score: maximize gain, minimize overlap
        # find max gain
        maxgain = int(gains.max())
        best_mask = (gains == maxgain)
        # among best, pick min overlap
        ov = overlaps[best_mask]
        minov = int(ov.min())
        cand_mask = best_mask & (overlaps == minov)
        idx = int(np.argmax(cand_mask))
        if idx < 0 or not cand_mask[idx]:
            idx = int(np.argmax(gains))
        chosen.append(idx)
        chosen_set.add(idx)
        uncovered &= ~ball[idx]
        if not uncovered.any():
            break

    # Ensure we have a full cover: if greedy stopped early due to budget, finish greedily
    while uncovered.any() and _remaining_budget() > 1.0:
        gains = ball @ uncovered.astype(np.int8)
        idx = int(np.argmax(gains))
        if idx in chosen_set:
            break
        chosen.append(idx)
        chosen_set.add(idx)
        uncovered &= ~ball[idx]

    # If still not covered (shouldn't happen), force-add remaining
    if uncovered.any():
        for idx in range(total):
            if uncovered.any() and idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
                uncovered &= ~ball[idx]

    # Local search: try to remove redundant codewords, then add+remove improvements
    def try_remove(chosen_list, chosen_set, uncovered):
        # returns new list/set/uncovered or None if no change
        m = len(chosen_list)
        if m == 0:
            return None
        # coverage count per word: how many chosen codewords cover it
        # cov = sum of ball[i] over i in chosen
        cov = np.zeros(total, dtype=np.int8)
        for i in chosen_list:
            cov += ball[i]
        # a codeword i is redundant if cov >= 2 for all words it uniquely... 
        # Actually i is redundant if for every word w in ball[i], there is another chosen covering w
        # i.e., cov[w] >= 2 for all w in ball[i]
        # equivalently: no word w where cov[w]==1 and ball[i,w]
        # compute: for each i, is there a w with ball[i,w] and cov[w]==1?
        # unique words covered by i = ball[i] & (cov==1)
        improved = False
        new_list = list(chosen_list)
        new_set = set(chosen_set)
        for pos, i in enumerate(chosen_list):
            if not (ball[i] & (cov == 1)).any():
                # redundant
                cov -= ball[i]
                new_list.pop(pos)
                new_set.discard(i)
                improved = True
                # recompute cov from scratch to be safe
                cov = np.zeros(total, dtype=np.int8)
                for j in new_list:
                    cov += ball[j]
                pos = -1
        if improved:
            # recompute uncovered
            unc = cov == 0
            return new_list, new_set, unc
        return None

    # Iterative improvement loop
    improved = True
    while improved and _remaining_budget() > 3.0:
        improved = False
        # remove pass
        res = try_remove(chosen, set(chosen), None)
        if res is not None:
            chosen, chosen_set, uncovered = res
            improved = True
            continue
        # add+remove pass: try adding a candidate that allows removing more than 1
        # For efficiency, only try candidates not already chosen, limited attempts
        # gain of adding i = ball[i] & uncovered (newly covered)
        # but we want to reduce total count: after adding i, can we remove >1 codewords?
        # Approximate: add i, then do a remove pass, check if net size decreased
        # To keep it fast, sample a subset of candidate indices
        unc = np.zeros(total, dtype=bool)
        cov = np.zeros(total, dtype=np.int8)
        for i in chosen:
            cov += ball[i]
        unc = cov == 0

        # candidate indices to try: those with high gain on uncovered
        if unc.any():
            gains = ball @ unc.astype(np.int8)
        else:
            gains = np.zeros(total, dtype=np.int8)
        # exclude already chosen
        for i in chosen:
            gains[i] = -1
        # top candidates
        top = np.argsort(gains)[::-1][:200]
        for i in top:
            if _remaining_budget() < 2.0:
                break
            if i < 0 or gains[i] < 0:
                continue
            # tentatively add
            new_list = list(chosen) + [int(i)]
            new_set = set(chosen) | {int(i)}
            new_cov = cov + ball[i]
            # remove pass on new_cov
            new_list2, new_set2, new_unc = try_remove(new_list, new_set, None)
            if new_list2 is not None and len(new_list2) < len(chosen):
                chosen = new_list2
                chosen_set = new_set2
                # recompute uncovered
                cov = np.zeros(total, dtype=np.int8)
                for j in chosen:
                    cov += ball[j]
                unc = cov == 0
                improved = True
                break

    # Final safety: ensure valid cover
    cov = np.zeros(total, dtype=np.int8)
    for i in chosen:
        cov += ball[i]
    if (cov == 0).any():
        # add missing
        for idx in range(total):
            if (cov == 0).any() and idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
                cov += ball[idx]

    return [list(base[i]) for i in chosen]
# EVOLVE-BLOCK-END
