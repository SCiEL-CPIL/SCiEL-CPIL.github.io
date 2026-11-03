"""Fresh design: vectorized greedy + local-search covering-code search for K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _cover_mask(cw, q, n):
    """Return a boolean array over the full space: True if within Hamming distance 1 of cw."""
    base = np.array(cw, dtype=np.int64)
    mask = np.zeros(q ** n, dtype=bool)
    mask[base] = True
    for pos in range(n):
        for sym in range(q):
            if sym == base[pos]:
                continue
            v = base.copy()
            v[pos] = sym
            mask[v] = True
    return mask


def _encode_space(q, n):
    """Enumerate all q^n words as an (N, n) int64 array plus base index per row."""
    coords = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int64)
    base = np.zeros(coords.shape[0], dtype=np.int64)
    for i in range(n):
        base = base * q + coords[:, i]
    return coords, base


def _dist_to_code(coords, base, code):
    """For each space word, the min Hamming distance to any codeword (code: list of n-vectors)."""
    c = np.array(code, dtype=np.int64)
    # diff matrix: N x M (can be large; for 2187 x ~100 fine)
    diff = np.abs(coords[:, None, :] - c[None, :, :])
    d = (diff != 0).sum(axis=2)
    return d.min(axis=1)


def _is_covering(coords, base, code, radius):
    return bool((_dist_to_code(coords, base, code) <= radius).all())


def _greedy_init(q, n, radius, seed):
    """Greedy: repeatedly add the word covering the most still-uncovered words."""
    rng = np.random.default_rng(seed)
    coords, base = _encode_space(q, n)
    N = coords.shape[0]
    # Precompute each word's radius-1 neighborhood as a set of indices for speed
    neigh = {}
    for idx in range(N):
        b = base[idx]
        words = [b]
        for pos in range(n):
            val = (b // (q ** (n - 1 - pos))) % q
            for sym in range(q):
                if sym == val:
                    continue
                nb = b + (sym - val) * (q ** (n - 1 - pos))
                words.append(nb)
        neigh[idx] = np.array(words, dtype=np.int64)
    covered = np.zeros(N, dtype=bool)
    code = []
    # start with a random word
    start = int(rng.integers(0, N))
    code.append(coords[start].tolist())
    covered[neigh[start]] = True
    while not covered.all():
        # gain of each candidate = number of newly-covered words
        best_gain = -1
        best_idx = -1
        for idx in range(N):
            g = np.sum(~covered[neigh[idx]])
            if g > best_gain:
                best_gain = g
                best_idx = idx
            elif g == best_gain and best_idx != -1 and rng.random() < 0.05:
                best_idx = idx  # tie-breaking randomization
        if best_gain <= 0:
            break
        code.append(coords[best_idx].tolist())
        covered[neigh[best_idx]] = True
    return code


def _local_search(q, n, radius, coords, base, code, max_passes, seed):
    """Reduce/swap codewords to lower count while keeping coverage."""
    rng = np.random.default_rng(seed)
    code = [list(c) for c in code]

    def counts():
        # count how many codewords cover each space word (within radius)
        cnt = np.zeros(base.shape[0], dtype=np.int64)
        for cw in code:
            b = np.array(cw, dtype=np.int64)
            words = [b]
            for pos in range(n):
                val = b[pos]
                for sym in range(q):
                    if sym == val:
                        continue
                    nb = b.copy()
                    nb[pos] = sym
                    words.append(nb)
            cnt[np.array(words, dtype=np.int64)] += 1
        return cnt

    for _ in range(max_passes):
        improved = False
        # try removing each codeword
        for i in range(len(code)):
            rest = code[:i] + code[i + 1:]
            if _is_covering(coords, base, rest, radius):
                code = rest
                improved = True
                break
        if improved:
            continue
        # try swaps: replace codeword i with a new word that improves coverage
        # (only useful if we first could add; we only swap to keep size)
        # Try replacing a codeword with another that allows a later removal.
        # Simple: try replacing codeword i with a random uncovered-adjacent word and check.
        cnt = counts()
        single = np.where(cnt == 1)[0]
        if len(single) == 0:
            break
        # find a codeword that uniquely covers some word; try swapping it.
        # For each codeword, compute its unique coverage set.
        # We'll attempt: replace codeword i with a word w such that coverage still holds.
        # Heuristic: pick the codeword whose removal leaves the largest gap, then try
        # replacing it with a word that covers that gap.
        best_swap = None
        for i in range(len(code)):
            rest = code[:i] + code[i + 1:]
            d = _dist_to_code(coords, base, rest)
            uncovered = np.where(d > radius)[0]
            if len(uncovered) == 0:
                continue
            # We need a single word covering all uncovered words within radius.
            # Find a word w (in space) that is within radius of all uncovered words.
            # candidate w must be within radius of each uncovered word.
            # intersect neighborhoods.
            cand = None
            for uw in uncovered[:50]:
                b = coords[uw]
                words = [b]
                for pos in range(n):
                    val = b[pos]
                    for sym in range(q):
                        if sym == val:
                            continue
                        nb = b.copy()
                        nb[pos] = sym
                        words.append(nb)
                arr = set(tuple(w) for w in words)
                if cand is None:
                    cand = arr
                else:
                    cand = cand & arr
                if not cand:
                    break
            if cand:
                w = list(next(iter(cand)))
                # ensure w not already in rest
                if w not in rest:
                    newcode = rest + [w]
                    if _is_covering(coords, base, newcode, radius):
                        best_swap = newcode
                        break
        if best_swap is not None:
            code = best_swap
            improved = True
        if not improved:
            # try a few random perturbations to escape local minima
            if rng.random() < 0.3 and len(code) > 0:
                i = int(rng.integers(0, len(code)))
                w = coords[int(rng.integers(0, base.shape[0]))].tolist()
                cand = code[:i] + code[i + 1:] + [w]
                if _is_covering(coords, base, cand, radius):
                    code = cand
                    improved = True
        if not improved:
            break
    return code


def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = 145.0
    coords, base = _encode_space(q, n)
    best = None
    best_len = float('inf')
    seed = 0
    while time.time() < DEADLINE:
        # budget per restart: leave margin for final validation
        start = time.time()
        code = _greedy_init(q, n, radius, seed)
        # local search reduction
        remaining = DEADLINE - (time.time() - t0)
        max_passes = max(1, int(remaining / 0.5))
        code = _local_search(q, n, radius, coords, base, code, max_passes, seed)
        # validate
        if _is_covering(coords, base, code, radius):
            if len(code) < best_len:
                best_len = len(code)
                best = [list(c) for c in code]
        # early stop if we reached a strong bound
        if best_len <= 90:
            break
        seed += 1
    if best is None:
        # fallback: direct-sum construction (always valid)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            best = [list(t) for t in itertools.product(range(q), repeat=n)]
    return best
# EVOLVE-BLOCK-END
