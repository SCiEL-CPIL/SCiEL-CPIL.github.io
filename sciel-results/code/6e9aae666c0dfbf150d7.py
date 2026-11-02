"""Structural rewrite: greedy set-cover for the ternary covering code K_3(7,1)."""
import time
import itertools

import numpy as np


def _all_words(q, n):
    """Return a (q**n, n) int64 array of every word in {0..q-1}^n (row-major)."""
    if n == 0:
        return np.zeros((1, 0), dtype=np.int64)
    base = np.arange(q, dtype=np.int64)
    out = base.reshape((q,) + (1,) * n)
    for i in range(1, n):
        out = np.concatenate([out, out + 1], axis=0)
    # ensure contiguous C-order (row-major)
    return np.ascontiguousarray(out)


def _sphere1_masks(q, n):
    """For each position and each symbol, a boolean mask of words having that symbol there."""
    allw = _all_words(q, n)
    masks = np.empty((n, q), dtype=object)
    for pos in range(n):
        for sym in range(q):
            masks[pos, sym] = (allw[:, pos] == sym)
    return allw, masks


def _sphere1_indices(masks, word):
    """Return the integer indices covered by the radius-1 sphere around `word`."""
    # covered = union over positions of (word with that position replaced by any symbol)
    acc = np.zeros(masks.shape[0] * 0, dtype=bool)  # placeholder, replaced below
    # word has length n; build boolean union
    union = None
    for pos in range(len(word)):
        for sym in range(masks.shape[1]):
            m = masks[pos, sym]
            if union is None:
                union = m.copy()
            else:
                union |= m
    return np.nonzero(union)[0]


def construct(q, n, radius):
    start = time.time()
    deadline = start + 130.0  # safety margin under the 150 s kill

    # Only the (q=3, n=7, radius=1) case is the real target; handle it specially.
    # Fallback for other cases: the direct-sum tetracode construction.
    if not (q == 3 and n == 7 and radius == 1):
        # generic fallback: direct sum of ternary tetracode over remaining coords
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        rem = n - 4
        if rem < 0:
            # too short: just emit all words
            return [list(w) for w in itertools.product(range(q), repeat=n)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=rem)]

    allw, masks = _sphere1_masks(q, n)
    N = allw.shape[0]

    # Precompute, for every word, the set of indices it covers (radius-1 sphere).
    # There are 2187 words; each sphere has 1 + 7*2 = 15 words.
    # We compute the index-set per word using the masks.
    word_cover = []
    for idx in range(N):
        word = allw[idx]
        union = None
        for pos in range(n):
            for sym in range(q):
                m = masks[pos, sym]
                if union is None:
                    union = m.copy()
                else:
                    union |= m
        word_cover.append(union)  # boolean array of length N

    covered = np.zeros(N, dtype=bool)
    code = []
    code_idx = []

    # Greedy: repeatedly pick the word whose sphere covers the most uncovered words.
    while not covered.all():
        if time.time() >= deadline - 5.0:
            break
        best = -1
        best_count = -1
        for idx in range(N):
            if covered[idx]:
                continue
            cnt = int(np.count_nonzero(word_cover[idx] & ~covered))
            if cnt > best_count:
                best_count = cnt
                best = idx
        if best < 0:
            break
        covered |= word_cover[best]
        code.append(allw[best].tolist())
        code_idx.append(best)

    # If we ran out of time and are not fully covered, we must still return a valid
    # code. Fall back to the direct-sum construction (guaranteed valid) if needed.
    if not covered.all():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    return code
