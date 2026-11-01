"""Rewritten solver for K_3(6,1) covering code. Bitset-based greedy + local search."""
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 112.0  # safety margin before 120s (evaluator kills at 150s)

    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    return _greedy_set_cover(q, n, radius, deadline)


def _solve_q3_n6_r1(deadline):
    q, n, r = 3, 6, 1
    total = q ** n  # 729

    # Precompute base-3 digit positions (word index -> digit at position p)
    pos = np.array([q ** p for p in range(n)], dtype=np.int64)
    digits = (np.arange(total, dtype=np.int64)[:, None] // pos[None, :]) % q  # (total, n)

    # Ball of radius 1: each word + its 2n neighbors. Precompute as (total, total) bool matrix.
    ball = np.zeros((total, total), dtype=bool)
    for p in range(n):
        d = digits[:, p]  # (total,)
        for v in range(q):
            mask = d == v  # words with digit v at position p
            for w in range(q):
                if w == v:
                    continue
                # shift: change digit p from v to w
                shift = (w - v) * pos[p]
                src = np.nonzero(mask)[0]
                dst = src + shift
                ball[dst, src] = True
    # add self
    np.fill_diagonal(ball, True)

    # Each column c: ball[:, c] is the set covered by choosing codeword c.
    # coverage_count[i] = number of chosen codewords whose ball covers word i.
    cov = np.zeros(total, dtype=np.int64)

    # ---- Greedy initialization: pick codeword covering most uncovered words ----
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    chosen_set = set()

    # symmetry score for tie-breaking: prefer codewords with low digit-sum mod 3? Use a
    # regularity heuristic: prefer codewords whose digits are "balanced".
    def sym_score(c):
        # lower = more symmetric (sum of digits mod 3 == 0 preferred)
        s = int(digits[c].sum())
        return abs(s % 3 - 1)  # 0 if s%3==1? just a deterministic tie-break
    # simpler deterministic tie-break: prefer smaller index among ties -> stable.

    while uncovered.any():
        # candidate gain = number of uncovered words covered by each codeword
        gains = ball[uncovered, :].sum(axis=0)  # (total,)
        # exclude already chosen
        for c in chosen_set:
            gains[c] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        chosen.append(best)
        chosen_set.add(best)
        cov[ball[:, best]] += 1
        uncovered[ball[:, best]] = False
        if time.time() > deadline - 20:
            break

    code = np.array(chosen, dtype=np.int64)

    # ---- Local search 1: pure removal pass (drop redundant codewords) ----
    def try_remove(i):
        nonlocal code
        idx = code[i]
        b = ball[:, idx]
        if np.any(cov[b] <= 1):
            return False
        cov[b] -= 1
        code = np.delete(code, i)
        return True

    improved = True
    while improved and time.time() < deadline - 25:
        improved = False
        for i in range(len(code)):
            if time.time() > deadline - 25:
                break
            res = try_remove(i)
            if res:
                improved = True
                break

    # ---- Local search 2: remove-then-repair (drop a codeword, re-cover gaps greedily) ----
    best_code = code.copy()
    best_size = len(code)

    rng = random.Random(12345)

    def recompute_cov(arr):
        c = np.zeros(total, dtype=np.int64)
        for idx in arr:
            c[ball[:, idx]] += 1
        return c

    for it in range(400):
        if time.time() > deadline - 8:
            break
        # pick a candidate codeword to remove (random order to escape local minima)
        order = list(range(len(code)))
        rng.shuffle(order)
        for i in order:
            if time.time() > deadline - 8:
                break
            idx = code[i]
            b = ball[:, idx]
            # find words that would become uncovered
            at_risk = b & (cov <= 1)
            if not at_risk.any():
                # free removal
                cov[b] -= 1
                code = np.delete(code, i)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
                break
            # tentatively remove
            old_cov = cov.copy()
            cov[b] -= 1
            new_code = np.delete(code, i)
            new_set = set(new_code.tolist())
            uncovered = cov == 0
            added = 0
            while uncovered.any():
                gains = ball[uncovered, :].sum(axis=0)
                for c in new_set:
                    gains[c] = -1
                best = int(np.argmax(gains))
                if gains[best] <= 0:
                    # cannot cover; revert
                    cov = old_cov
                    break
                new_code = np.append(new_code, best)
                new_set.add(best)
                cov[ball[:, best]] += 1
                uncovered[ball[:, best]] = False
                added += 1
                if added >= 2:
                    break
            if added == 0 and uncovered.any():
                # revert (failed)
                cov = old_cov
                continue
            # accept if smaller
            if len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code
                code = new_code
                # recompute cov to be safe
                cov = recompute_cov(code)
                break

    # Final: ensure we return best found
    out = best_code
    # Validate coverage fully (safety)
    covf = np.zeros(total, dtype=np.int64)
    for idx in out:
        covf[ball[:, idx]] += 1
    if not np.all(covf >= 1):
        # fallback to greedy re-cover
        uncovered = covf == 0
        while uncovered.any() and time.time() < deadline - 2:
            gains = ball[uncovered, :].sum(axis=0)
            best = int(np.argmax(gains))
            if gains[best] <= 0:
                break
            out = np.append(out, best)
            covf[ball[:, best]] += 1
            uncovered[ball[:, best]] = False

    # Convert indices to words
    result = []
    for idx in out:
        w = [(int(idx) // pos[p]) % q for p in range(n)]
        result.append(w)
    return result


def _greedy_set_cover(q, n, radius, deadline):
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    pos = np.array([q ** p for p in range(n)], dtype=np.int64)
    digits = (np.arange(total, dtype=np.int64)[:, None] // pos[None, :]) % q

    # ball of radius `radius`
    ball = np.zeros((total, total), dtype=bool)
    # radius 1 fast path; general: enumerate
    if radius == 1:
        for p in range(n):
            d = digits[:, p]
            for v in range(q):
                mask = d == v
                for w in range(q):
                    if w == v:
                        continue
                    shift = (w - v) * pos[p]
                    src = np.nonzero(mask)[0]
                    dst = src + shift
                    ball[dst, src] = True
        np.fill_diagonal(ball, True)
    else:
        # brute force for small spaces
        for i in range(total):
            for j in range(total):
                if np.count_nonzero(digits[i] != digits[j]) <= radius:
                    ball[i, j] = True

    cov = np.zeros(total, dtype=np.int64)
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    chosen_set = set()
    while uncovered.any() and time.time() < deadline - 5:
        gains = ball[uncovered, :].sum(axis=0)
        for c in chosen_set:
            gains[c] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        chosen.append(best)
        chosen_set.add(best)
        cov[ball[:, best]] += 1
        uncovered[ball[:, best]] = False

    # removal pass
    code = np.array(chosen, dtype=np.int64)
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(len(code)):
            idx = code[i]
            b = ball[:, idx]
            if not np.any(cov[b] <= 1):
                cov[b] -= 1
                code = np.delete(code, i)
                improved = True
                break

    result = []
    for idx in code:
        w = [(int(idx) // pos[p]) % q for p in range(n)]
        result.append(w)
    return result
