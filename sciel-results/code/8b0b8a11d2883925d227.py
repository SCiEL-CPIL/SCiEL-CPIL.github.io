# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 125.0  # hard wall-clock kill is 150s; keep ~25s safety margin
_START = time.time()


def _remaining():
    return _DEADLINE - time.time()


def _seed_words(q, n, radius):
    """Direct-sum of the ternary tetracode (9 words, length 4, radius-1 perfect)
    with all words on the remaining coordinates."""
    if n < 4:
        return [list(c) for c in np.ndindex(*([q] * n))]
    tet = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    rest = n - 4
    out = []
    for h in tet:
        for tail in np.ndindex(*([q] * rest)):
            out.append(h + list(tail))
    return out


def _build_neighborhoods(q, n, radius):
    """Return (cells, neigh): cells = int index of every word in {0..q-1}^n;
    neigh = list of cell-index arrays; neigh[i] covers word i plus all words
    within Hamming distance radius."""
    cells = np.arange(q ** n, dtype=np.int64)
    npos = n * (q - 1)  # number of 1-position variants per word
    neigh = np.empty((q ** n, npos + 1), dtype=np.int64)
    neigh[:, 0] = cells
    k = 1
    for d in range(n):
        base = cells - d * q ** (n - 1 - d)
        for v in range(q):
            if v == d % q:
                continue
            neigh[:, k] = base + v * q ** (n - 1 - d)
            k += 1
    return cells, neigh


def _covered_mask(words, neigh):
    """Bitset (bool array over all cells) of cells covered by the given words."""
    mask = np.zeros(neigh.shape[0], dtype=bool)
    for w in words:
        mask[neigh[w]] = True
    return mask


def _redundant_mask(words, neigh):
    """For each codeword index, True if removing it leaves all its cells still
    covered by the other codewords."""
    m = len(words)
    if m == 0:
        return np.zeros(0, dtype=bool)
    covered = _covered_mask(words, neigh)
    cnt = np.zeros(neigh.shape[0], dtype=np.int64)
    for w in words:
        cnt[neigh[w]] += 1
    red = np.zeros(m, dtype=bool)
    for i, w in enumerate(words):
        if np.all(covered[neigh[w]] & (cnt[neigh[w]] > 1)):
            red[i] = True
    return red


def _best_add(uncovered, neigh, rng, sample_cap=20000):
    """Return (index, gain) of the word covering the most uncovered cells,
    or (None, 0) if none. Samples to keep it fast when space is large."""
    total = neigh.shape[0]
    if total <= sample_cap:
        cand = np.arange(total)
    else:
        cand = rng.choice(total, size=sample_cap, replace=False)
    gains = np.zeros(len(cand), dtype=np.int64)
    for j, w in enumerate(cand):
        gains[j] = uncovered[neigh[w]].sum()
    j = int(np.argmax(gains))
    return int(cand[j]), int(gains[j])


def _best_swap(uncovered, neigh, words, rng, sample_cap=8000):
    """Return (out_i, in_w, net) best single swap, or (None, None, 0)."""
    m = len(words)
    if m == 0:
        return None, None, 0
    covered = _covered_mask(words, neigh)
    cnt = np.zeros(neigh.shape[0], dtype=np.int64)
    for w in words:
        cnt[neigh[w]] += 1
    total = neigh.shape[0]
    if total <= sample_cap:
        cand = np.arange(total)
    else:
        cand = rng.choice(total, size=sample_cap, replace=False)
    best = (None, None, 0)
    for w in cand:
        if w in words:
            continue
        add = uncovered[neigh[w]].sum()
        if add <= 0:
            continue
        for i, wi in enumerate(words):
            # net gain if we drop wi and add w
            # cells that become newly covered: w's cells not covered by others
            # cells lost: wi's cells that only wi covered
            others = cnt[neigh[wi]] - 1
            lost = np.sum((cnt[neigh[wi]] == 1) & ~covered[neigh[wi]])  # 0 normally
            # a cell of wi is lost if cnt==1 (only wi) -> after removal uncovered
            lost = np.sum(cnt[neigh[wi]] == 1)
            net = int(add) - int(lost)
            if net > best[2]:
                best = (i, int(w), net)
    return best


def construct(q, n, radius):
    global _START
    _START = time.time()
    rng = random.Random(12345)
    rng_np = np.random.default_rng(12345)

    cells, neigh = _build_neighborhoods(q, n, radius)
    total = neigh.shape[0]

    words = _seed_words(q, n, radius)
    # de-duplicate
    seen = set()
    uniq = []
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    words = uniq

    best_words = list(words)
    best_size = len(words)

    def fully_covered(wlist):
        mask = _covered_mask(wlist, neigh)
        return bool(np.all(mask))

    # ensure best is valid
    if not fully_covered(best_words):
        # fallback: full space (always valid)
        best_words = [list(c) for c in np.ndindex(*([q] * n))]
        best_size = len(best_words)

    # Greedy set-cover refinement: start from seed, but greedily add to fix any
    # gaps, then remove redundancies.
    work = list(words)
    if _remaining() > 5:
        # remove redundancies repeatedly
        for _ in range(50):
            if _remaining() < 5:
                break
            red = _redundant_mask(work, neigh)
            if not red.any():
                break
            work = [w for i, w in enumerate(work) if not red[i]]
        # greedily add until covered
        for _ in range(2000):
            if _remaining() < 5:
                break
            mask = _covered_mask(work, neigh)
            if np.all(mask):
                break
            uncovered = ~mask
            w, gain = _best_add(uncovered, neigh, rng_np)
            if w is None or gain <= 0:
                break
            work.append(w)
    if len(work) < best_size and fully_covered(work):
        best_words, best_size = work, len(work)

    # Local search polish on the best valid solution
    if _remaining() > 8:
        cur = list(best_words)
        improved = True
        while improved and _remaining() > 8:
            improved = False
            # 1) remove redundancies
            red = _redundant_mask(cur, neigh)
            if red.any():
                cur = [w for i, w in enumerate(cur) if not red[i]]
                improved = True
                if len(cur) < best_size and fully_covered(cur):
                    best_words, best_size = cur, len(cur)
                continue
            # 2) try swaps
            mask = _covered_mask(cur, neigh)
            if not np.all(mask):
                uncovered = ~mask
                w, gain = _best_add(uncovered, neigh, rng_np)
                if w is not None and gain > 0:
                    cur.append(w)
                    improved = True
                    if len(cur) < best_size and fully_covered(cur):
                        best_words, best_size = cur, len(cur)
                continue
            out_i, in_w, net = _best_swap(~mask, neigh, cur, rng_np)
            if out_i is not None and net > 0:
                cur[out_i] = in_w
                improved = True
                if len(cur) < best_size and fully_covered(cur):
                    best_words, best_size = cur, len(cur)

    # Final safety: if we're still over the deadline, return best valid.
    if not fully_covered(best_words):
        best_words = [list(c) for c in np.ndindex(*([q] * n))]
    return best_words
# EVOLVE-BLOCK-END
