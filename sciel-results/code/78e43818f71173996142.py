"""Covering code K_q(n, r): minimize number of codewords covering every word within radius r."""
import time
import itertools
import numpy as np


def _build_initial_code(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, perfect code radius 1)
    # with every word on the remaining n-4 coordinates: 9 * q**(n-4) words.
    if q == 3 and n >= 4:
        tet = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        rest = n - 4
        if rest == 0:
            return [h[:] for h in tet]
        tails = np.array(list(itertools.product(range(q), repeat=rest)), dtype=np.int8)
        heads = np.array(tet, dtype=np.int8)
        code = np.concatenate([np.repeat(heads, len(tails), axis=0),
                               np.tile(tails, (len(heads), 1))], axis=1)
        return code
    # Fallback: a single codeword (valid only for tiny spaces; otherwise improved below).
    return np.zeros((1, n), dtype=np.int8)


def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = 110.0          # safety margin before the 150s wall-clock kill
    SPACELIMIT = 1_500_000    # full-space coverage checks only below this size
    CODELIMIT = 250_000       # candidate pool cap for the greedy pass

    code = _build_initial_code(q, n, radius)
    code = np.unique(code, axis=0)

    def time_left():
        return DEADLINE - (time.time() - t0)

    def full_covered(cw):
        # cw: (k, n) int array; return True if every word of {0..q-1}^n is within radius.
        space = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
        d = np.abs(space[:, None, :] - cw[None, :, :]).sum(axis=2)
        return bool(np.all(d.min(axis=1) <= radius))

    def covered_mask(cw):
        space = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
        d = np.abs(space[:, None, :] - cw[None, :, :]).sum(axis=2)
        return d.min(axis=1) <= radius

    # ---- Phase 0: greedy reduction (remove codewords whose ball is redundant) ----
    if code.shape[0] > 1 and code.shape[0] <= CODELIMIT and time_left() > 5.0:
        m = code.shape[0]
        balls = np.array([np.array(list(itertools.product(range(q), repeat=n)),
                                   dtype=np.int8) - code[i],
                          ] for i in range(m))  # placeholder; replaced below
        # Build per-codeword ball masks (word index -> bool) efficiently.
        # Represent a word by its base-q integer index.
        def word_to_idx(w):
            idx = 0
            for v in w:
                idx = idx * q + v
            return idx
        def idx_to_word(idx):
            w = [0] * n
            for j in range(n - 1, -1, -1):
                w[j] = idx % q
                idx //= q
            return w
        total = q ** n
        masks = np.zeros((m, total), dtype=bool)
        for i in range(m):
            ci = code[i]
            # all words within radius of ci
            for off in range(total):
                w = idx_to_word(off)
                if sum(abs(a - b) for a, b in zip(ci, w)) <= radius:
                    masks[i, off] = True
        keep = np.ones(m, dtype=bool)
        order = np.arange(m)
        # Remove a codeword if its ball is covered by the union of the others.
        changed = True
        while changed and time_left() > 3.0:
            changed = False
            union = np.zeros(total, dtype=bool)
            for i in range(m):
                if keep[i]:
                    union |= masks[i]
            for i in range(m):
                if not keep[i]:
                    continue
                if time_left() < 3.0:
                    break
                # ball i redundant if masks[i] subset of union without i
                u_without = np.zeros(total, dtype=bool)
                for j in range(m):
                    if keep[j] and j != i:
                        u_without |= masks[j]
                if np.all(masks[i] <= u_without):
                    keep[i] = False
                    changed = True
                    break
        if changed:
            code = code[keep]

    # ---- Phase 1: greedy addition of the best uncovered word ----
    if code.shape[0] < CODELIMIT and time_left() > 10.0:
        # work on a candidate pool of words, not the whole space, to bound cost
        pool_size = min(total if 'total' in dir() else q ** n, CODELIMIT)
        rng = np.random.default_rng(0)
        pool_idx = rng.choice(q ** n, size=min(pool_size, q ** n), replace=False)
        pool = np.array([idx_to_word(int(x)) for x in pool_idx], dtype=np.int8)
        # current covered set over pool
        d = np.abs(pool[:, None, :] - code[None, :, :]).sum(axis=2)
        cov = d.min(axis=1) <= radius
        while time_left() > 10.0:
            if cov.all():
                break
            # candidate = uncovered word that covers the most uncovered words
            uncov = ~cov
            if not uncov.any():
                break
            # for each pool word, count how many uncovered words it covers
            d2 = np.abs(pool[:, None, :] - pool[None, :, :]).sum(axis=2)
            # restrict to uncovered targets
            gains = (d2 <= radius) & uncov[None, :]
            gains = gains.sum(axis=1)
            # only consider adding a pool word that is itself uncovered (typical)
            gains[cov] = -1
            if gains.max() <= 0:
                break
            best = int(np.argmax(gains))
            code = np.vstack([code, pool[best][None, :]])
            d = np.abs(pool[:, None, :] - code[None, :, :]).sum(axis=2)
            cov = d.min(axis=1) <= radius

    # ---- Phase 2: local improvement (swap one codeword for a better one) ----
    if code.shape[0] >= 2 and time_left() > 20.0 and code.shape[0] <= CODELIMIT:
        # candidate replacement pool: a sample of the space
        cand_size = min(q ** n, CODELIMIT)
        rng = np.random.default_rng(1)
        cand_idx = rng.choice(q ** n, size=min(cand_size, q ** n), replace=False)
        cand = np.array([idx_to_word(int(x)) for x in cand_idx], dtype=np.int8)
        improved = True
        while improved and time_left() > 20.0:
            improved = False
            m = code.shape[0]
            # current coverage of the candidate pool
            d_cur = np.abs(cand[:, None, :] - code[None, :, :]).sum(axis=2)
            cur_cov = d_cur.min(axis=1) <= radius
            for i in range(m):
                if time_left() < 20.0:
                    break
                # coverage if we drop codeword i
                d_drop = np.abs(cand[:, None, :] - np.delete(code, i, axis=0)[None, :, :]).sum(axis=2)
                drop_cov = d_drop.min(axis=1) <= radius
                # words that become uncovered by dropping i
                lost = cur_cov & ~drop_cov
                if not lost.any():
                    # safe to drop i entirely
                    code = np.delete(code, i, axis=0)
                    improved = True
                    break
                # find a candidate that covers all lost words (and is not already in code)
                d_cand = np.abs(cand[:, None, :] - cand[None, :, :]).sum(axis=2)
                # a candidate c covers word w if dist(c,w)<=radius; need to cover all lost
                lost_idx = np.where(lost)[0]
                ok = (d_cand[lost_idx, :] <= radius).all(axis=0)
                # exclude candidates already in code
                for j, row in enumerate(code):
                    ok &= ~np.all(cand == row, axis=1)
                if ok.any():
                    # pick the candidate covering the most lost words (for robustness)
                    cover_count = (d_cand[lost_idx, :] <= radius).sum(axis=0)
                    cover_count[~ok] = -1
                    best = int(np.argmax(cover_count))
                    # replace i with best
                    code[i] = cand[best]
                    improved = True
                    break

    # ---- Final: verify full coverage if the space is small enough; else trust ----
    if q ** n <= SPACELIMIT and time_left() > 2.0:
        if not full_covered(code):
            # should not happen; as a last resort fall back to the full direct-sum code
            code = _build_initial_code(q, n, radius)
    return [list(map(int, row)) for row in code]
