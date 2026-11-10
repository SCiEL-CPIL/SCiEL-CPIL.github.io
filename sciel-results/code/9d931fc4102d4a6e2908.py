import random
import time

N = 16

# ---- precompute Q3 subcube embeddings ----
# A Q3 embedding = 3 disjoint vertex pairs (a_i,b_i); 12 edges.
_PAIRS = [(a, b) for a in range(N) for b in range(a + 1, N)]
_TRIPLES = []
for i in range(len(_PAIRS)):
    a1, b1 = _PAIRS[i]
    for j in range(i + 1, len(_PAIRS)):
        a2, b2 = _PAIRS[j]
        if a2 in (a1, b1) or b2 in (a1, b1):
            continue
        for k in range(j + 1, len(_PAIRS)):
            a3, b3 = _PAIRS[k]
            if a3 in (a1, b1, a2, b2) or b3 in (a1, b1, a2, b2):
                continue
            _TRIPLES.append(((a1, b1), (a2, b2), (a3, b3)))

_SUBCUBE_EDGE_MASKS = []
_EDGE_SUBS = [[] for _ in range(N * N)]
for idx, ((a1, b1), (a2, b2), (a3, b3)) in enumerate(_TRIPLES):
    e = [(a1, b1), (a2, b2), (a3, b3),
         (a1, a2), (a1, a3), (a2, a3),
         (b1, b2), (b1, b3), (b2, b3),
         (a1, b2), (a2, b3), (a3, b1)]
    assert len(e) == 12
    m = 0
    for x, y in e:
        m |= (1 << (x * N + y)) | (1 << (y * N + x))
    _SUBCUBE_EDGE_MASKS.append(m)
    for x, y in e:
        _EDGE_SUBS[x * N + y].append(idx)
        _EDGE_SUBS[y * N + x].append(idx)


def _is_q3_free(mask):
    for m in _SUBCUBE_EDGE_MASKS:
        if (mask & m) == m:
            return False
    return True


def _would_create(mask, edge):
    for idx in _EDGE_SUBS[edge]:
        if (mask & _SUBCUBE_EDGE_MASKS[idx]) == _SUBCUBE_EDGE_MASKS[idx]:
            return True
    return False


def _add_all(mask):
    for x in range(N):
        for y in range(x + 1, N):
            if (mask >> (x * N + y)) & 1:
                continue
            nm = mask | (1 << (x * N + y)) | (1 << (y * N + x))
            if not _would_create(mask, x * N + y):
                mask = nm
    return mask


def _greedy(rng):
    order = [(x, y) for x in range(N) for y in range(x + 1, N)]
    rng.shuffle(order)
    mask = 0
    for x, y in order:
        nm = mask | (1 << (x * N + y)) | (1 << (y * N + x))
        if not _would_create(mask, x * N + y):
            mask = nm
    return mask


def _local(mask, rng, deadline):
    mask = _add_all(mask)
    best = mask
    best_count = bin(mask).count('1') // 2
    while time.time() < deadline:
        mask = _add_all(mask)
        c = bin(mask).count('1') // 2
        if c > best_count:
            best = mask
            best_count = c
        present = [(x, y) for x in range(N) for y in range(x + 1, N)
                   if (mask >> (x * N + y)) & 1]
        if not present:
            break
        x, y = rng.choice(present)
        mask = mask & ~(1 << (x * N + y)) & ~(1 << (y * N + x))
        mask = _add_all(mask)
        c = bin(mask).count('1') // 2
        if c > best_count:
            best = mask
            best_count = c
    return best


def _to_edges(mask):
    return [[x, y] for x in range(N) for y in range(x + 1, N)
            if (mask >> (x * N + y)) & 1]


def construct():
    deadline = time.time() + 110.0
    rng = random.Random(12345)
    best = 0
    best_count = -1
    seed_mask = 0
    for a in range(3):
        for b in range(3, 16):
            seed_mask |= (1 << (a * N + b)) | (1 << (b * N + a))
    seed_mask = _add_all(seed_mask)
    best = seed_mask
    best_count = bin(seed_mask).count('1') // 2

    while time.time() < deadline:
        g = _greedy(rng)
        g = _local(g, rng, deadline)
        c = bin(g).count('1') // 2
        if c > best_count:
            best = g
            best_count = c
    return _to_edges(best)
