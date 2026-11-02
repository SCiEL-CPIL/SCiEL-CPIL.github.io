import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    timeout = 118.0  # safety margin vs 150s wall-clock kill

    # General fallback (not the target case)
    if not (q == 3 and n == 7 and radius == 1):
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                         for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode
                    for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n  # 2187

    def to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def from_idx(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # --- Precompute the candidate (word -> covered indices) matrix once ---
    # cand[i] = list of indices covered by codeword i (the word itself + radius-1 flips)
    cand = np.empty((N, 1 + n * (q - 1)), dtype=np.int32)
    for i in range(N):
        w = from_idx(i)
        cand[i, 0] = i
        col = 1
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                cand[i, col] = to_idx(w)
                col += 1
            w[pos] = orig

    # --- Phase 1: fast vectorized greedy seed ---
    def greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        covered = np.zeros(N, dtype=bool)
        remaining = np.ones(N, dtype=np.int32)   # uncovered count per candidate
        selected = []
        while True:
            if covered.all():
                break
            if seed is not None:
                # randomize among the top candidates
                top = 12
                idx = np.argpartition(-remaining, top)[:top]
                c = idx[random.randrange(top)]
            else:
                c = int(np.argmax(remaining))
            selected.append(c)
            covered[cand[c]] = True
            remaining -= cand @ covered  # vectorized update of all remaining counts
        return selected

    # --- Phase 2: local search (add / remove / exchange) on the chosen set ---
    def local_search(cw, max_iters=12000, seed=None):
        if seed is not None:
            random.seed(seed)
        cw = list(cw)
        m = len(cw)
        counts = cand @ np.array(cw, dtype=np.int32)  # how many chosen cover each word
        order = list(range(m))
        it = 0
        while it < max_iters:
            it += 1
            if it % 512 == 0 and time.time() - start_time > timeout - 1.5:
                break
            # try to remove a redundant codeword (counts>=2 everywhere it covers)
            removable = [j for j in order
                         if (counts[cand[cw[j]]] >= 2).all()]
            if removable:
                j = removable[random.randrange(len(removable))]
                counts[cand[cw[j]]] -= 1
                cw.pop(j)
                order.pop(j)
                m -= 1
                continue
            # try a 1-for-1 exchange
            improved = False
            for _ in range(30):
                i = random.randrange(m)
                ci = cw[i]
                # candidates whose removal would not break coverage
                safe = [k for k in order if k != i
                        and (counts[cand[cw[k]]] >= 2).all()]
                if not safe:
                    continue
                k = safe[random.randrange(len(safe))]
                ck = cw[k]
                # words that only cw[k] covers (would be lost on removal)
                lost = cand[ck] * (counts[cand[ck]] == 1)
                # candidate ci covers 'lost' iff ci's row has 1 at those indices
                if (cand[ci][lost] > 0).all():
                    counts[cand[ck]] -= 1
                    counts[cand[ci]] += 1
                    cw[k] = ci
                    improved = True
                    break
            if improved:
                continue
            # no removal and no exchange found -> try a random add (escape plateau)
            if m < N - 1:
                j = random.randrange(m)
                cj = cw[j]
                # candidate j covers words only cw[cj] covers -> adding it makes cj removable
                need = cand[cj] * (counts[cand[cj]] == 1)
                cand_j = [x for x in range(N)
                          if (cand[x][need] > 0).all() and x not in cw]
                if cand_j:
                    newc = cand_j[random.randrange(len(cand_j))]
                    counts[cand[cj]] -= 1
                    counts[cand[newc]] += 1
                    cw.append(newc)
                    order.append(m)
                    m += 1
                    continue
            break
        return cw

    # --- Orchestrate: one deterministic seed + many randomized seeds ---
    best = None
    best_size = N
    cw = greedy(seed=None)
    cw = local_search(cw)
    best = cw
    best_size = len(cw)

    seed = 0
    while time.time() - start_time < timeout - 2.0:
        cw = greedy(seed=seed)
        cw = local_search(cw, seed=seed)
        if len(cw) < best_size:
            best = cw
            best_size = len(cw)
        seed += 1

    return [from_idx(c) for c in best]

# EVOLVE-BLOCK-END
