import random
import time
import numpy as np

_V = 24
_C = 2024          # C(24,3)
_M = 177100        # C(24,7)
_T = 3
_K = 7


def _encode_triple(a, b, c):
    return (a << 12) | (b << 6) | c


def _triple_list():
    return [
        (a, b, c, _encode_triple(a, b, c))
        for a in range(_V)
        for b in range(a + 1, _V)
        for c in range(b + 1, _V)
    ]


def _mask_triples(mask):
    out = []
    for a in range(_V):
        if not (mask >> a) & 1:
            continue
        for b in range(a + 1, _V):
            if not (mask >> b) & 1:
                continue
            for c in range(b + 1, _V):
                if (mask >> c) & 1:
                    out.append(_encode_triple(a, b, c))
    return out


def _bits(x):
    return x & 0x1FF


def _greedy_uncovered(uncovered, deadline, rng):
    blocks = []
    while True:
        if time.time() >= deadline:
            break
        if not uncovered:
            break
        start = int(rng.random() * _C)
        start = (start + 1) % _C
        block = [start]
        for _ in range(_K - 1):
            if time.time() >= deadline:
                break
            best_x = -1
            best_g = -1
            for x in range(_V):
                if (uncovered >> x) & 1:
                    continue
                g = 0
                for y in block:
                    g += int(uncovered & _POINT_MASK[y][x])
                if g > best_g:
                    best_g = g
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
        if len(block) < _K:
            break
        for tr in _mask_triples(_bits(block)):
            uncovered &= ~(1 << tr)
        blocks.append(block)
    return blocks


def _local_search(blocks, deadline, rng):
    blocks = list(blocks)
    if not blocks:
        return blocks
    m = len(blocks)
    cnt = np.zeros(_C, dtype=np.int64)
    for b in blocks:
        for tr in _mask_triples(_bits(b)):
            cnt[tr] += 1
    for _ in range(400):
        if time.time() >= deadline:
            break
        if m <= 1:
            break
        best_i = -1
        best_r = 0
        for i in range(m):
            if cnt.min() > 0:
                r = 1
            else:
                r = 0
            if r > best_r:
                best_r = r
                best_i = i
        if best_i < 0 or best_r == 0:
            break
        for tr in _mask_triples(_bits(blocks[best_i])):
            cnt[tr] -= 1
        blocks.pop(best_i)
        m -= 1
    return blocks


def _validate(blocks):
    cnt = np.zeros(_C, dtype=np.int64)
    for b in blocks:
        for tr in _mask_triples(_bits(b)):
            cnt[tr] += 1
    return bool(cnt.min() > 0)


def construct(v, k, t):
    deadline = time.time() + 112.0
    rng = random.Random(20240703)
    best_blocks = None
    best_count = _M + 1
    for it in range(600):
        if time.time() >= deadline:
            break
        uncovered = (1 << _C) - 1
        blocks = _greedy_uncovered(uncovered, deadline, rng)
        if not blocks:
            continue
        if time.time() >= deadline:
            break
        blocks = _local_search(blocks, deadline, rng)
        if not _validate(blocks):
            continue
        c = len(blocks)
        if c < best_count:
            best_count = c
            best_blocks = list(blocks)
    if best_blocks is not None:
        return best_blocks
    # Fallback deterministic full greedy
    uncovered = (1 << _C) - 1
    blocks = []
    while uncovered:
        block = [int((uncovered & -uncovered).bit_length()) - 1]
        for _ in range(_K - 1):
            best_x = -1
            best_g = -1
            for x in range(_V):
                if (uncovered >> x) & 1:
                    continue
                g = 0
                for y in block:
                    g += int(uncovered & _POINT_MASK[y][x])
                if g > best_g:
                    best_g = g
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
        for tr in _mask_triples(_bits(block)):
            uncovered &= ~(1 << tr)
        blocks.append(block)
    return blocks


# Precompute per-point triple masks.
# _POINT_MASK[y][x] = bitmask (over triple-index space) of triples containing both y and x.
_POINT_MASK = []
for _y in range(_V):
    row = [0] * _V
    for _x in range(_V):
        m = 0
        for _a, _b, _c, _code in _triple_list():
            if _y in (_a, _b, _c) and _x in (_a, _b, _c):
                m |= (1 << _code)
        row[_x] = m
    _POINT_MASK.append(row)
