import numpy as np
import time

_N = 17
_MIN_D = 6
_SEED = 32
_START = time.time()
_DEADLINE = 85.0  # safety margin before the 90s kill


def _qr_code():
    """[18,9,6] extended quadratic-residue code (256 words) shortened to length 17."""
    p = 17
    # quadratic residues mod 17
    QR = set((i * i) % p for i in range(1, p))
    # generator matrix of the [17,8,5] QR code: rows 1, x, x^2, ...
    G = np.zeros((8, p), dtype=np.uint8)
    for k in range(8):
        for i in range(p):
            if i == 0:
                G[k, i] = 1
            elif (i % p) in QR:
                G[k, i] = (k + 1) % 2
    # parity-check column (extend to length 18)
    par = np.array([G[:, i].sum() % 2 for i in range(p)], dtype=np.uint8)
    # build all 256 codewords of the [18,9,6] code
    full = np.zeros((256, p + 1), dtype=np.uint8)
    for m in range(256):
        bits = np.array([(m >> j) & 1 for j in range(8)], dtype=np.uint8)
        row = (bits @ G) % 2
        full[m, :p] = row
        full[m, p] = row.sum() % 2
    # shorten by one coordinate: drop the last column, keep words with that bit = 0
    mask = full[:, p] == 0
    short = full[mask, :p]
    # convert to integers
    powers = (1 << np.arange(p, dtype=np.int64))[::-1]
    ints = (short @ powers).astype(np.int64)
    return ints


def _hamming(a, b):
    return (a ^ b).bit_count()


def construct():
    t0 = time.time()
    deadline = t0 + (_DEADLINE - (time.time() - _START))
    if deadline - t0 < 1.0:
        deadline = t0 + 1.0

    # strong seed: 256-word shortened [18,9,6] code
    try:
        ints = _qr_code()
        ints = np.unique(ints)
    except Exception:
        ints = np.array([0], dtype=np.int64)

    code = ints.tolist()
    code_set = set(code)

    # precompute popcount table for fast distance
    # use numpy for the main checks
    code_arr = np.array(code, dtype=np.int64)

    # Greedy add: try to insert additional words
    # Candidate pool: all 2^17 words, but we can't iterate all (131072) cheaply against 256+
    # Instead, do a smarter search: random + local improvement
    # First, try adding words from a broad random candidate set

    rng = np.random.default_rng(12345)
    added = 0
    attempts = 0

    # We'll maintain code as a list and use vectorized distance checks.
    # For efficiency, keep code as a numpy array and check new candidates against all.

    def can_add(cand, current):
        # current: numpy array of int64
        diffs = current ^ cand
        # compute popcount for each
        # use a vectorized approach via unpacking bits
        # For a single cand, we can loop but that's slow; use numpy bit tricks
        # popcount of each diff: sum of bits
        # We'll use a lookup-free method: for each diff, count set bits
        # numpy doesn't have popcount; use a trick with bit shifting
        d = diffs.astype(np.int64)
        cnt = np.zeros(len(d), dtype=np.int32)
        for _ in range(_N):
            cnt += (d & 1).astype(np.int32)
            d >>= 1
        return (cnt >= _MIN_D).all()

    # Try to add random candidates
    max_rounds = 200000
    for _ in range(max_rounds):
        if time.time() > deadline - 2.0:
            break
        cand = int(rng.integers(0, 1 << _N))
        if cand in code_set:
            continue
        ca = np.array([cand], dtype=np.int64)
        diffs = code_arr ^ cand
        d = diffs.astype(np.int64)
        cnt = np.zeros(len(d), dtype=np.int32)
        for _ in range(_N):
            cnt += (d & 1).astype(np.int32)
            d >>= 1
        if (cnt >= _MIN_D).all():
            code.append(cand)
            code_set.add(cand)
            code_arr = np.append(code_arr, cand)
            added += 1

    return code


# Fallback if construct isn't the only entry point needed
def main():
    return construct()
