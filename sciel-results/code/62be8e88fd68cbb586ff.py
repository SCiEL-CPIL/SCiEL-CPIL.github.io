import numpy as np
import time

_N = 17
_MIN_D = 6
_SEED = 32
_START = time.time()
_DEADLINE = 85.0  # safety margin before the 90s kill

# Precompute a bit-unpack matrix: 17 -> 64 columns (4 bits per byte) for fast popcount
# For a vector of int64, we shift right by 0,2,4,...,62 and mask with 0b11, then sum.
# We'll do it in chunks of 4 bits to keep it fast.
def _popcount_vec(diffs):
    """Vectorized popcount for an array of int64 values (17-bit)."""
    d = diffs.astype(np.int64)
    # Use the standard bit-parallel popcount for small integers
    # d is 17 bits; we can do:
    # count = d; count = count - ((count >> 1) & 0x5555555555555555); ...
    # But simpler: unpack via shifts
    cnt = np.zeros(len(d), dtype=np.int32)
    for s in range(0, 17):
        cnt += (d >> s) & 1
    return cnt


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

    def _min_dist_to_code(cand, code_arr_local):
        """Return minimum Hamming distance from cand to any word in code_arr_local."""
        diffs = code_arr_local ^ cand
        cnt = _popcount_vec(diffs)
        return cnt.min()

    def try_add(cand, code_list, code_set_local, code_arr_local):
        """Try to add cand; returns (success, updated_code_arr)."""
        if cand in code_set_local:
            return False, code_arr_local
        if _min_dist_to_code(cand, code_arr_local) >= _MIN_D:
            code_list.append(cand)
            code_set_local.add(cand)
            return True, np.append(code_arr_local, cand)
        return False, code_arr_local

    # Phase 1: Random insertion
    max_rounds = 200000
    for _ in range(max_rounds):
        if time.time() > deadline - 2.0:
            break
        cand = int(rng.integers(0, 1 << _N))
        ok, code_arr = try_add(cand, code, code_set, code_arr)
        if ok:
            added += 1

    # Phase 2: Local search — try flipping 1 bit of each codeword to find a new valid word
    if time.time() < deadline - 3.0:
        for i in range(len(code)):
            if time.time() > deadline - 2.0:
                break
            base = code[i]
            for bit in range(_N):
                cand = base ^ (1 << bit)
                ok, code_arr = try_add(cand, code, code_set, code_arr)
                if ok:
                    added += 1

    # Phase 3: More random rounds with different seed
    if time.time() < deadline - 2.0:
        rng2 = np.random.default_rng(54321)
        for _ in range(100000):
            if time.time() > deadline - 1.5:
                break
            cand = int(rng2.integers(0, 1 << _N))
            ok, code_arr = try_add(cand, code, code_set, code_arr)
            if ok:
                added += 1

    return code


# Fallback if construct isn't the only entry point needed
def main():
    return construct()
