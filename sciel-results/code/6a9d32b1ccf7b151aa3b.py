"""Rewritten solver: pack cosets of RM(1,4) into 17-bit words, d>=6."""
import time
import numpy as np

N = 17
MIN_DISTANCE = 6
DEADLINE = 78.0  # safety margin vs 90s kill


def _popcount16_rows(x):
    """Popcount of each uint8 row (vectorized)."""
    x = x.astype(np.uint8)
    x = x ^ (x >> 1)
    x = (x ^ (x >> 2)) & 0x33
    x = (x ^ (x >> 4)) & 0x0F
    return x.astype(np.int32)


def construct():
    t0 = time.time()
    pts = np.arange(16, dtype=np.uint8)
    slopes = np.arange(16, dtype=np.uint8)[:, None] & pts[None, :]
    rm = np.vstack([_popcount16_rows(slopes),
                    _popcount16_rows(slopes) ^ 0xFFFF])  # (32,16)

    rng = np.random.default_rng(12345)
    order = rng.permutation(1 << 16)
    labels = np.arange(1 << 16, dtype=np.uint32)[order]

    sel_cores = []   # list of (32,16) uint8
    sel_words = []   # list of (32,17) uint8

    for idx in range(len(labels)):
        if time.time() > DEADLINE:
            break
        label = int(labels[idx])
        row = np.frombuffer(label.to_bytes(2, 'little'), dtype=np.uint8)
        cores = rm ^ row  # (32,16)
        if sel_cores:
            sc = np.vstack(sel_cores)
            ok = True
            chunk = 4096
            for s0 in range(0, sc.shape[0], chunk):
                blk = sc[s0:s0 + chunk]
                d = _popcount16_rows(cores[:, None, :] ^ blk[None, :, :])
                if d.min() < MIN_DISTANCE:
                    ok = False
                    break
            if not ok:
                continue
        for bit in (0, 1):
            full = np.zeros((32, 17), dtype=np.uint8)
            full[:, :16] = cores
            full[:, 16] = bit
            sel_words.append(full)
            sel_cores.append(cores)

    if not sel_words:
        out = []
        for w in rm:
            val = 0
            for i, b in enumerate(w):
                if b:
                    val |= (1 << i)
            out.append(val)
        return out

    flat = np.vstack(sel_words)
    out = []
    for i in range(flat.shape[0]):
        w = flat[i]
        val = 0
        for b in range(17):
            if w[b]:
                val |= (1 << b)
        out.append(val)
    seen = set()
    res = []
    for v in out:
        if v not in seen:
            seen.add(v)
            res.append(v)
    return res
