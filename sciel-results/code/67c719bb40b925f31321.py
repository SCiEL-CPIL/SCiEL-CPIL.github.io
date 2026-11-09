import numpy as np
import time
import random

_N = 17
_MIN_D = 6
_SEED = 32
_START = time.time()
_DEADLINE = 85.0  # safety margin before the 90s kill

# Precompute bit masks for faster operations
_MASKS = [1 << i for i in range(_N)]


def _popcount_vec(diffs):
    """Vectorized popcount for an array of int64 values (17-bit)."""
    d = diffs.astype(np.int64)
    cnt = np.zeros(len(d), dtype=np.int32)
    for s in range(_N):
        cnt += (d >> s) & 1
    return cnt


def _qr_code():
    """[18,9,6] extended quadratic-residue code (256 words) shortened to length 17."""
    p = 17
    # quadratic residues mod 17
    QR = set((i * i) % p for i in range(1, p))
    # generator matrix of the [17,8,5] QR code
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


def _min_dist_to_code(cand, code_arr):
    """Return minimum Hamming distance from cand to any word in code_arr."""
    diffs = code_arr ^ cand
    cnt = _popcount_vec(diffs)
    return int(cnt.min())


def _try_add(cand, code_list, code_set, code_arr):
    """Try to add cand; returns (success, updated_code_arr)."""
    if cand in code_set:
        return False, code_arr
    if _min_dist_to_code(cand, code_arr) >= _MIN_D:
        code_list.append(cand)
        code_set.add(cand)
        return True, np.append(code_arr, cand)
    return False, code_arr


def _local_improve(code_list, code_set, code_arr, deadline):
    """Try flipping 1-2 bits of codewords to find new valid words."""
    added = 0
    n = len(code_list)
    
    # Phase 1: single bit flips
    for i in range(n):
        if time.time() > deadline - 2.0:
            break
        base = code_list[i]
        for bit in range(_N):
            cand = base ^ (1 << bit)
            ok, code_arr = _try_add(cand, code_list, code_set, code_arr)
            if ok:
                added += 1
    
    # Phase 2: double bit flips (only if time remains)
    if time.time() < deadline - 3.0:
        for i in range(len(code_list)):
            if time.time() > deadline - 2.0:
                break
            base = code_list[i]
            for b1 in range(_N - 1):
                for b2 in range(b1 + 1, _N):
                    cand = base ^ (1 << b1) ^ (1 << b2)
                    ok, code_arr = _try_add(cand, code_list, code_set, code_arr)
                    if ok:
                        added += 1
    return added


def construct():
    t0 = time.time()
    deadline = t0 + (_DEADLINE - (time.time() - _START))
    if deadline - t0 < 1.0:
        deadline = t0 + 1.0

    # Strong seed: 256-word shortened [18,9,6] code
    try:
        ints = _qr_code()
        ints = np.unique(ints)
    except Exception:
        ints = np.array([0], dtype=np.int64)

    code = ints.tolist()
    code_set = set(code)
    code_arr = np.array(code, dtype=np.int64)

    added_total = 0

    # Phase 1: Random insertion with multiple seeds
    rng = np.random.default_rng(12345)
    max_rounds = 150000
    for _ in range(max_rounds):
        if time.time() > deadline - 3.0:
            break
        cand = int(rng.integers(0, 1 << _N))
        ok, code_arr = _try_add(cand, code, code_set, code_arr)
        if ok:
            added_total += 1

    # Phase 2: Local improvement (1-bit and 2-bit flips)
    if time.time() < deadline - 4.0:
        added_total += _local_improve(code, code_set, code_arr, deadline)

    # Phase 3: More random rounds with different seed
    if time.time() < deadline - 2.0:
        rng2 = np.random.default_rng(54321)
        for _ in range(100000):
            if time.time() > deadline - 1.5:
                break
            cand = int(rng2.integers(0, 1 << _N))
            ok, code_arr = _try_add(cand, code, code_set, code_arr)
            if ok:
                added_total += 1

    # Phase 4: Another local improvement pass if time remains
    if time.time() < deadline - 3.0:
        added_total += _local_improve(code, code_set, code_arr, deadline)

    # Phase 5: Final random sweep
    if time.time() < deadline - 1.0:
        rng3 = np.random.default_rng(77777)
        for _ in range(50000):
            if time.time() > deadline - 0.5:
                break
            cand = int(rng3.integers(0, 1 << _N))
            ok, code_arr = _try_add(cand, code, code_set, code_arr)
            if ok:
                added_total += 1

    return code


def main():
    return construct()
