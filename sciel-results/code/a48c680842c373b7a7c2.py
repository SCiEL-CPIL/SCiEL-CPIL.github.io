"""Search for a small ternary covering code K_3(6,1)."""
import time
import itertools
import numpy as np


def _all_words(q, n):
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)


def _balls(words, q, n):
    # For each word, the set of words within Hamming distance 1.
    # words: (m, n) int8. Returns (m, 1+q*n) int8 matrix of covered word indices.
    m = words.shape[0]
    base = words.reshape(m, 1, n)  # (m,1,n)
    # center
    centers = words  # (m,n)
    # neighbors: for each position, each other symbol
    neigh = []
    for pos in range(n):
        for s in range(q):
            w = words.copy()
            w[:, pos] = s
            neigh.append(w)
    neigh = np.array(neigh, dtype=np.int8)  # (n*(q-1), m, n) -> actually (m, n*(q-1), n)
    # reshape to (m, n*(q-1), n)
    neigh = neigh.reshape(m, -1, n)
    allw = _all_words(q, n)  # (q^n, n)
    # encode each word to an index via base-q
    powq = (q ** np.arange(n))[::-1]  # (n,)
    def encode(W):
        return (W.astype(np.int64) * powq).sum(axis=1)
    centers_idx = encode(centers)
    neigh_idx = encode(neigh.reshape(-1, n)).reshape(m, -1)
    return np.hstack([centers_idx[:, None], neigh_idx])


def _greedy_cover(q, n, radius, words_pool, time_limit, deadline):
    # words_pool: list of candidate codewords (as int8 arrays), or None to use all words.
    allw = _all_words(q, n)
    total = allw.shape[0]
    if radius == 1:
        # precompute balls for the pool
        if words_pool is None:
            pool = allw
        else:
            pool = np.array(words_pool, dtype=np.int8)
        balls = _balls(pool, q, n)  # (m, 1+q*n)
        m = pool.shape[0]
        covered = np.zeros(total, dtype=bool)
        chosen = []
        # initial uncovered count
        while True:
            if time.time() > deadline:
                break
            if covered.all():
                break
            # for each candidate, count newly covered
            b = balls  # (m, B)
            # newly = ~covered[b]
            newly = (~covered)[b].sum(axis=1)  # (m,)
            # only consider candidates not already chosen
            chosen_set = set(chosen)
            best = -1
            best_idx = -1
            for i in range(m):
                if i in chosen_set:
                    continue
                if newly[i] > best:
                    best = newly[i]
                    best_idx = i
            if best_idx < 0 or best <= 0:
                break
            chosen.append(best_idx)
            covered[balls[best_idx]] = True
        return [pool[i] for i in chosen]


def construct(q, n, radius):
    start = time.time()
    # hard wall-clock limit with safety margin
    deadline = start + 100.0  # 100s of the 120s budget

    # Baseline: product construction (valid, 9 * q**(n-4) for the ternary tetracode family)
    def product_baseline():
        if q == 3 and n == 6 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # generic fallback: single codeword won't cover; use all words (valid but large)
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    baseline = product_baseline()
    best = baseline
    best_len = len(best)

    # Try greedy set-cover from the full pool, then local improvement (remove redundant).
    allw = _all_words(q, n)
    total = allw.shape[0]

    # Precompute balls for ALL words (used for coverage checks and removals)
    # This is 729 * (1+18) entries — fine.
    if radius == 1:
        all_balls = _balls(allw, q, n)  # (total, 1+q*n)
    else:
        all_balls = None

    def coverage_count(code_list):
        # return number of covered words
        if radius != 1:
            # slow generic
            cov = np.zeros(total, dtype=bool)
            for w in code_list:
                w = np.array(w, dtype=np.int8)
                for cand in allw:
                    if np.sum(cand != w) <= radius:
                        cov[cand] = True
            return cov.sum()
        idx = (np.array(code_list, dtype=np.int64) * (q ** np.arange(n)[::-1])).sum(axis=1)
        cov = np.zeros(total, dtype=bool)
        cov[all_balls[idx]] = True
        return cov.sum()

    def is_valid(code_list):
        return coverage_count(code_list) == total

    # Greedy from full pool
    try:
        greedy = _greedy_cover(q, n, radius, None, 30, deadline - 20)
        if greedy is not None and len(greedy) < best_len:
            # local improvement: try removing words that are redundant
            improved = list(greedy)
            changed = True
            while changed and time.time() < deadline - 10:
                changed = False
                cov = np.zeros(total, dtype=bool)
                idx = (np.array(improved, dtype=np.int64) * (q ** np.arange(n)[::-1])).sum(axis=1)
                cov[all_balls[idx]] = True
                if cov.sum() != total:
                    break
                # try removing each
                for i in range(len(improved)):
                    if time.time() > deadline - 10:
                        break
                    test = improved[:i] + improved[i+1:]
                    if is_valid(test):
                        improved = test
                        changed = True
                        break
            if len(improved) < best_len and is_valid(improved):
                best = improved
                best_len = len(best)
    except Exception:
        pass

    # Try a few random restarts of greedy with a random initial seed to beat baseline
    rng = np.random.RandomState(0)
    for attempt in range(4):
        if time.time() > deadline - 15:
            break
        # start with a small random seed
        seed_size = max(1, best_len - 6 - attempt * 2)
        seed_idx = rng.choice(total, size=min(seed_size, total), replace=False)
        seed = [allw[i] for i in seed_idx]
        # greedy add
        cov = np.zeros(total, dtype=bool)
        idx = (np.array(seed, dtype=np.int64) * (q ** np.arange(n)[::-1])).sum(axis=1)
        cov[all_balls[idx]] = True
        chosen = list(seed)
        chosen_set = set(map(tuple, seed))
        while time.time() < deadline - 10:
            if cov.all():
                break
            # pick the word (from all) maximizing newly covered
            best_i = -1
            best_gain = 0
            # sample a subset of candidates for speed
            cand_idx = rng.choice(total, size=min(200, total), replace=False)
            for ci in cand_idx:
                w = allw[ci]
                if tuple(w) in chosen_set:
                    continue
                gain = (~cov)[all_balls[ci]].sum()
                if gain > best_gain:
                    best_gain = gain
                    best_i = ci
            if best_i < 0:
                break
            chosen.append(allw[best_i])
            chosen_set.add(tuple(allw[best_i]))
            cov[all_balls[best_i]] = True
        if len(chosen) < best_len and is_valid(chosen):
            best = chosen
            best_len = len(best)

    # Final safety: ensure validity; if not, fall back to baseline
    if not is_valid(best):
        best = baseline

    return [list(map(int, w)) for w in best]
