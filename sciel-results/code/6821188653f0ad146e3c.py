"""Fresh design: fast NumPy covering oracle + randomized local search from the 81-word
direct-sum seed, with a greedy set-cover completion to guarantee validity."""
import numpy as np
import time

_DEADLINE = time.time() + 142.0  # safety margin under the 150 s wall-clock kill
_RNG = np.random.RandomState(12345)
_TETRA = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)], dtype=np.int8)


def _seed_code(q, n, radius):
    if n == 4:
        return _TETRA.copy()
    base = _TETRA
    if n < 4:
        base = base[:, :n]
    if n > 4:
        tails = np.array(np.meshgrid(*([np.arange(q)] * (n - 4))), dtype=np.int8).reshape(-1, n - 4)
        base = np.concatenate([np.repeat(base, len(tails), axis=0),
                               np.tile(tails, (len(base), 1))], axis=1)
    return base.copy()


def _coverage_matrix(C, q, n):
    """rows = codewords, cols = all q^n words; True if within Hamming distance <= radius."""
    N = q ** n
    idx = np.arange(N, dtype=np.int64)
    digits = (idx[:, None] // (q ** np.arange(n, dtype=np.int64)[::-1])) % q
    digits = digits.T.astype(np.int8)                      # (n, N)
    C = C.astype(np.int8)
    d = np.abs(C[:, None, :] - digits[None, :, :])
    d[d > (q - 1) // 2] = q - d[d > (q - 1) // 2]
    dist = d.sum(axis=2)                                    # (|C|, N)
    return dist <= radius


def _is_covered(C, q, n, radius):
    N = q ** n
    idx = np.arange(N, dtype=np.int64)
    digits = (idx[:, None] // (q ** np.arange(n, dtype=np.int64)[::-1])) % q
    digits = digits.T.astype(np.int8)
    C = C.astype(np.int8)
    d = np.abs(C[:, None, :] - digits[None, :, :])
    d[d > (q - 1) // 2] = q - d[d > (q - 1) // 2]
    dist = d.sum(axis=2)
    return bool((dist <= radius).any(axis=0).all())


def _greedy_complete(C, q, n, radius):
    """Add words until every word is within radius of some codeword (guaranteed valid)."""
    N = q ** n
    idx = np.arange(N, dtype=np.int64)
    digits = (idx[:, None] // (q ** np.arange(n, dtype=np.int64)[::-1])) % q
    digits = digits.T.astype(np.int8)
    C = C.astype(np.int8)
    d = np.abs(C[:, None, :] - digits[None, :, :])
    d[d > (q - 1) // 2] = q - d[d > (q - 1) // 2]
    dist = d.sum(axis=2)
    covered = (dist <= radius).any(axis=0)
    if covered.all():
        return C
    while not covered.all():
        d2 = np.abs(digits[:, None, :] - digits[None, :, :])
        d2[d2 > (q - 1) // 2] = q - d2[d2 > (q - 1) // 2]
        dist2 = d2.sum(axis=2)                              # (N, N)
        gains = ((dist2 <= radius) & (~covered)[None, :]).sum(axis=1)
        i = int(np.argmax(gains))
        if gains[i] <= 0:
            break
        covered |= (dist2[i] <= radius)
        C = np.vstack([C, digits[:, i].reshape(1, -1)])
    return C


def construct(q, n, radius):
    global _RNG
    t0 = time.time()
    C = _seed_code(q, n, radius)
    cur = len(C)
    best = C.copy()

    # Precompute the ball (set of words within radius of each position word) for fast add.
    N = q ** n
    idx = np.arange(N, dtype=np.int64)
    digits = (idx[:, None] // (q ** np.arange(n, dtype=np.int64)[::-1])) % q
    digits = digits.T.astype(np.int8)                       # (n, N)
    dpos = np.abs(digits[:, None, :] - digits[None, :, :])
    dpos[dpos > (q - 1) // 2] = q - dpos[dpos > (q - 1) // 2]
    distpos = dpos.sum(axis=2)
    ball = distpos <= radius                                # (N, N)

    # Coverage oracle over the full space.
    cover = _coverage_matrix(C, q, n, radius)
    covered = cover.any(axis=0)

    def try_accept(newC, newCover):
        nonlocal C, cur, best, covered
        if newC.shape[0] < cur and bool(newCover.any(axis=0).all()):
            C = newC
            cur = newC.shape[0]
            best = C.copy()
            covered = newCover.any(axis=0)
            return True
        return False

    # ---- Pass 1: random remove / move / add ------------------------------------
    while time.time() < _DEADLINE - 40.0 and cur > 70:
        m = len(C)
        if m == 0:
            break
        r = _RNG.random_sample()
        if r < 0.55:  # remove a random codeword
            i = _RNG.randint(m)
            newCover = cover.copy()
            newCover[i] = False
            if not bool(newCover.any(axis=0).all()):
                continue
            newC = np.delete(C, i, axis=0)
            if try_accept(newC, newCover):
                cover = newCover
        elif r < 0.85:  # move a codeword to a random new word
            i = _RNG.randint(m)
            j = _RNG.randint(N)
            C2 = C.copy()
            C2[i] = digits[:, j]
            if (C2 == C2[i]).any(axis=1).any() and i != 0:
                # duplicate check: ensure no other row equals the new row
                dup = ((C2 == C2[i]).all(axis=1))
                dup[i] = False
                if dup.any():
                    continue
            newCover = _coverage_matrix(C2, q, n, radius)
            if try_accept(C2, newCover):
                cover = newCover
        else:  # add a random word (only useful if it later enables a removal)
            j = _RNG.randint(N)
            C2 = C.copy()
            C2 = np.vstack([C2, digits[:, j].reshape(1, -1)])
            if len(C2) < cur:
                pass
            newCover = _coverage_matrix(C2, q, n, radius)
            # adding never shrinks; skip unless it directly helps nothing -> ignore
            continue

    # ---- Pass 2: greedy set-cover completion (guarantees validity) -------------
    if time.time() < _DEADLINE - 5.0:
        C = _greedy_complete(C, q, n, radius)
        if len(C) < cur:
            best = C.copy()

    # Final validity safeguard.
    if not _is_covered(best, q, n, radius):
        best = _greedy_complete(best, q, n, radius)

    return [list(map(int, row)) for row in best]
