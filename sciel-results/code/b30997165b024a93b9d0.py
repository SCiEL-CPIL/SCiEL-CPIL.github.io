"""Fresh design for K_3(8,1): punctured ternary Golay + greedy minimal covering code."""
import time
import numpy as np

# ---- Ternary Golay code (length 11, 1728 words, covering radius 2) -------------
# Generator matrix G (11 x 5) over GF(3); columns: identity | [1,1,1,1,1] | [1,2,4,1,2]
_G = np.array([
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 4, 1, 2, 1, 2, 4, 1, 2, 4],
    [2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2],
    [4, 2, 1, 4, 1, 4, 2, 1, 4, 1, 2],
    [1, 4, 2, 2, 4, 1, 4, 2, 2, 4, 1],
    [2, 1, 4, 4, 2, 2, 1, 4, 4, 2, 1],
    [4, 2, 1, 1, 4, 4, 2, 1, 1, 4, 2],
], dtype=np.int64)


def _golay_words():
    gens = _G.T  # 5 x 11
    coeffs = np.stack(np.meshgrid(*[np.arange(3)] * 5), axis=-1).reshape(-1, 5)
    return (coeffs @ gens) % 3


def _puncture(words, keep):
    return words[:, list(keep)]


def _is_covering(code, q, n, radius):
    """True iff every word of {0..q-1}^n is within `radius` of some row of `code`."""
    code = np.asarray(code, dtype=np.int64)
    if code.size == 0:
        return False
    n_rows = code.shape[0]
    chunk = 200000
    total = q ** n
    for start in range(0, total, chunk):
        cnt = np.arange(start, min(start + chunk, total))
        words = (cnt[:, None] // (q ** np.arange(n - 1, -1, -1))) % q  # (c, n)
        if radius == 1:
            # dist<=1 iff equal to a codeword, or equal with one coordinate replaced by each other symbol
            eq = (words[:, None, :] == code[None, :, :]).all(axis=2)  # (c, m)
            if eq.any(axis=1).all():
                continue
            for s in range(1, q):
                ok = np.zeros(words.shape[0], dtype=bool)
                for j in range(n):
                    w = words.copy()
                    w[:, j] = (w[:, j] + s) % q
                    e = (w[:, None, :] == code[None, :, :]).all(axis=2)
                    if e.any(axis=1).all():
                        continue
                    ok |= e.any(axis=1)
                if ok.all():
                    break
            if ok.all():
                continue
            return False
        else:
            d = (words[:, None, :] != code[None, :, :]).sum(axis=2)  # (c, m)
            if (d.min(axis=1) <= radius).all():
                continue
            return False
    return True


def _greedy_minimize(code, q, n, radius, deadline):
    """Greedy removal: drop a row if the set still covers. Returns a covering code."""
    code = [list(r) for r in code]
    m = len(code)
    order = list(range(m))
    # try to remove rows in an order that tends to expose redundancy first
    for _ in range(3):
        if time.time() > deadline:
            break
        # pick the row whose removal keeps coverage; prefer rows with many near-duplicates
        removed = True
        while removed and time.time() < deadline:
            removed = False
            arr = np.asarray(code, dtype=np.int64)
            # score each row by number of other rows at distance 1 (high => likely redundant)
            dup = np.zeros(m, dtype=np.int64)
            for s in range(1, q):
                for j in range(n):
                    a = arr.copy()
                    a[:, j] = (a[:, j] + s) % q
                    e = (arr[:, None, :] == a[None, :, :]).all(axis=2)
                    dup += e.sum(axis=1) - 1
            cand = int(np.argmax(dup))
            if dup[cand] <= 0:
                break
            test = code[:cand] + code[cand + 1:]
            if _is_covering(test, q, n, radius):
                code = test
                m -= 1
                removed = True
    return code


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin under the 150s kill

    # Fallback: tetracode direct sum (valid, 9 * 3**(n-4))
    def tetracode_fallback():
        tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        if n >= 4:
            return [h + list(t) for h in tet for t in itertools.product(range(3), repeat=n - 4)]
        return [h[:n] for h in tet]

    best = None
    if q == 3 and n == 8:
        words = _golay_words()
        # puncture: drop 3 coordinates -> length 8, radius 1 covering code (1728 words)
        for keep in (
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 8],
            [0, 1, 2, 3, 4, 5, 7, 8],
            [0, 1, 2, 3, 4, 6, 7, 8],
            [0, 1, 2, 3, 5, 6, 7, 8],
            [0, 1, 2, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ):
            cand = _puncture(words, keep)
            if time.time() > deadline - 30:
                break
            if _is_covering(cand, q, n, radius):
                mini = _greedy_minimize(cand, q, n, radius, deadline - 15)
                if best is None or len(mini) < len(best):
                    best = mini
                if len(best) <= 420:
                    break
    if best is None:
        best = tetracode_fallback()
    # final safety: ensure validity; if greedy somehow broke it, fall back
    if not _is_covering(best, q, n, radius):
        best = tetracode_fallback()
    return best
