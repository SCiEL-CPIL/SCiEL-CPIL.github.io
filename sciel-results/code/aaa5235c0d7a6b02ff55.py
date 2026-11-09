# EVOLVE-BLOCK-START
import time
import numpy as np

N = 10
DEMANDS = [
    (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
    (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
    (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
]
D = len(DEMANDS)


def _cw_mask(s, t):
    m = 0
    e = s
    while e != t:
        m |= (1 << e)
        e = (e + 1) % N
    return m


def _ccw_mask(s, t):
    m = 0
    e = (s - 1) % N
    while e != (t - 1) % N:
        m |= (1 << e)
        e = (e - 1) % N
    return m


CW = []
CCW = []
W = []
for (s, t, w) in DEMANDS:
    CW.append(_cw_mask(s, t))
    CCW.append(_ccw_mask(s, t))
    W.append(w)

FULL = (1 << N) - 1
CW_ARR = np.array(CW, dtype=np.int64)
CCW_ARR = np.array(CCW, dtype=np.int64)
W_ARR = np.array(W, dtype=np.int64)


def _max_load(bits):
    # bits: int, bit i => demand i routed CCW (1) else CW (0)
    loads = np.zeros(N, dtype=np.int64)
    for i in range(D):
        if (bits >> i) & 1:
            loads += W_ARR[i] * ((CCW_ARR[i] & FULL).view(np.uint64) if False else np.unpackbits(np.array(CCW_ARR[i], dtype=np.uint8) & 0xFF))
        else:
            loads += W_ARR[i] * np.unpackbits(np.array(CW_ARR[i], dtype=np.uint8) & 0xFF)
    return int(np.max(loads))


def _max_load_fast(bits):
    # scalar loop with precomputed edge bit-arrays per demand
    loads = [0] * N
    for i in range(D):
        if (bits >> i) & 1:
            m = CCW[i]
            w = W[i]
        else:
            m = CW[i]
            w = W[i]
        e = 0
        mm = m
        while mm:
            if mm & 1:
                loads[e] += w
            e += 1
            mm >>= 1
    return max(loads)


def _local_search(bits):
    # hill-climb: try flipping each bit, keep if improves; repeat until no gain
    cur = _max_load_fast(bits)
    improved = True
    while improved:
        improved = False
        for i in range(D):
            nb = bits ^ (1 << i)
            v = _max_load_fast(nb)
            if v < cur:
                bits = nb
                cur = v
                improved = True
    return bits, cur


def ring_routing() -> np.ndarray:
    deadline = time.time() + 600.0 - 30.0  # safety margin

    # Exhaustive search over all 2^15 routings using vectorized load computation.
    # For each routing bits, loads = sum_i W_i * (CCW_i if bit else CW_i).
    # We vectorize by iterating bits and using precomputed per-demand edge arrays.

    # Precompute per-demand edge presence as boolean arrays (N,)
    cw_edges = np.zeros((D, N), dtype=np.int64)
    ccw_edges = np.zeros((D, N), dtype=np.int64)
    for i in range(D):
        for e in range(N):
            if (CW[i] >> e) & 1:
                cw_edges[i, e] = 1
            if (CCW[i] >> e) & 1:
                ccw_edges[i, e] = 1

    best_bits = 0
    best_val = _max_load_fast(0)

    # Full enumeration with vectorized scoring per candidate
    total = 1 << D
    # Process in chunks to keep memory bounded
    chunk = 4096
    for start in range(0, total, chunk):
        if time.time() > deadline:
            break
        end = min(start + chunk, total)
        bits_arr = np.arange(start, end, dtype=np.int64)
        # For each bit position i, demand i is CCW if (bits >> i) & 1
        # loads[e] = sum_i W_i * (ccw_edges[i,e] if bit_i else cw_edges[i,e])
        # = sum_i W_i * cw_edges[i,e] + sum_i W_i * (ccw_edges[i,e]-cw_edges[i,e]) * bit_i
        base = W_ARR @ cw_edges  # shape (N,)
        delta = W_ARR @ (ccw_edges - cw_edges)  # shape (N,)
        # For each bits in bits_arr, contribution = sum_i bit_i * delta_i
        # bit_i for demand i in bits: (bits >> i) & 1
        contrib = np.zeros((end - start, N), dtype=np.int64)
        for i in range(D):
            mask_i = (bits_arr >> i) & 1
            contrib += mask_i[:, None] * delta[i, None]
        loads = base[None, :] + contrib  # (chunk, N)
        vals = np.max(loads, axis=1)
        idx = int(np.argmin(vals))
        if vals[idx] < best_val:
            best_val = int(vals[idx])
            best_bits = int(bits_arr[idx])

    # Local search polish
    if time.time() < deadline:
        best_bits, best_val = _local_search(best_bits)

    routing = np.zeros(D)
    for i in range(D):
        routing[i] = 1.0 if (best_bits >> i) & 1 else 0.0
    return routing


# EVOLVE-BLOCK-END
