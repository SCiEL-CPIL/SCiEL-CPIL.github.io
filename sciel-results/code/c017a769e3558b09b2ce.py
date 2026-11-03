# EVOLVE-BLOCK-START
import time
import numpy as np

_BUDGET = 118.0


def _ball_indices(base, q, n, r):
    """All indices (base-10 over alphabet q) within Hamming distance r of base.
    Returns a 1-D int64 array of unique indices."""
    if r >= n:
        return np.arange(q ** n, dtype=np.int64)
    r = min(r, n)
    base = int(base)
    out = [base]
    for d in range(1, r + 1):
        # all ways to pick exactly d positions to change (0/1 mask)
        for mask in range(1 << n):
            if bin(mask).count('1') != d:
                continue
            # cartesian product of q choices over the selected positions
            vals = [list(range(q))] * d
            for combo in np.ndindex(*([q] * d)):
                idx = base
                sel = 0
                for pos in range(n):
                    if (mask >> pos) & 1:
                        idx += (combo[sel] - (base // (q ** pos)) % q) * (q ** pos)
                        sel += 1
                out.append(idx)
    return np.array(out, dtype=np.int64)


def construct(q, n, radius):
    t0 = time.time()
    N = q ** n
    r = min(int(radius), n)

    # Precompute the set of all indices and a coverage boolean array.
    all_idx = np.arange(N, dtype=np.int64)
    covered = np.zeros(N, dtype=bool)

    # --- initial seed: a small random starting set, then greedy ---
    rng = np.random.default_rng(12345)

    def ball_of(idx):
        return _ball_indices(idx, q, n, r)

    code = []
    # seed with one random word
    covered[ball_of(int(rng.integers(0, N)))] = True
    code.append(int(rng.integers(0, N)))

    # Greedy: repeatedly add the word covering the most currently-uncovered points.
    # Candidate pool: a random sample of the space (plus all words if small).
    if N <= 4096:
        candidates = all_idx.copy()
    else:
        candidates = np.unique(rng.integers(0, N, size=min(N, 60000)))

    # Precompute balls for candidates once (they are fixed).
    cand_balls = [ball_of(int(c)) for c in candidates]

    while True:
        if time.time() - t0 > _BUDGET - 25.0:
            break
        if covered.all():
            break
        # pick candidate with max newly-covered
        best = -1
        best_gain = -1
        for i, b in enumerate(cand_balls):
            gain = int(np.count_nonzero(covered[b] == False))
            if gain > best_gain:
                best_gain = gain
                best = i
            if gain <= 0:
                continue
        if best_gain <= 0:
            # no candidate helps (shouldn't happen if pool large enough); expand pool
            newc = np.unique(rng.integers(0, N, size=min(N, 20000)))
            for c in newc:
                if c not in candidates:
                    candidates = np.concatenate([candidates, [c]])
                    cand_balls.append(ball_of(int(c)))
            continue
        b = cand_balls[best]
        covered[b] = True
        code.append(int(candidates[best]))

    # --- redundancy removal: drop any codeword whose ball is fully covered by others ---
    def idx_to_word(idx):
        w = [0] * n
        x = idx
        for pos in range(n):
            w[pos] = x % q
            x //= q
        return w

    balls = [ball_of(int(c)) for c in code]
    keep = [True] * len(code)
    for i in range(len(code)):
        if not keep[i]:
            continue
        if time.time() - t0 > _BUDGET - 12.0:
            break
        # is ball[i] covered by union of all other kept balls?
        others = np.zeros(N, dtype=bool)
        for j in range(len(code)):
            if j != i and keep[j]:
                others[balls[j]] = True
        if others[balls[i]].all():
            keep[i] = False

    final_code = [int(c) for c, k in zip(code, keep) if k]

    # --- safety: verify coverage; if broken, fall back to full code ---
    if time.time() - t0 < _BUDGET - 6.0:
        chk = np.zeros(N, dtype=bool)
        for c in final_code:
            chk[ball_of(int(c))] = True
        if not chk.all():
            final_code = list(code)

    return [idx_to_word(c) for c in final_code]
# EVOLVE-BLOCK-END
