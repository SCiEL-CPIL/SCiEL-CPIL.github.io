"""Q3-free dense graph on 16 vertices.
Structural change: precomputed Q3-embedding table + per-edge 'forbidden count'.
Adding edge e is safe iff forbidden_count[e] == 0 (no Q3 embedding is completed).
"""
import time
import random
from itertools import combinations

_N = 16
_ALL_EDGES = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_EDGE_INDEX = {e: i for i, e in enumerate(_ALL_EDGES)}
_N_EDGES = len(_ALL_EDGES)  # 120

# All labeled embeddings of Q3 into 16 labeled vertices.
_Q3_EMB_EDGES = []
_Q3_VERTS = []
for _vs in combinations(range(_N), 8):
    _L = list(_vs)
    for _a in range(8):
        for _b in range(_a + 1, 8):
            for _c in range(_b + 1, 8):
                for _d in range(_c + 1, 8):
                    for _e in range(_d + 1, 8):
                        for _f in range(_e + 1, 8):
                            for _g in range(_f + 1, 8):
                                for _h in range(_g + 1, 8):
                                    _L = (_a, _b, _c, _d, _e, _f, _g, _h)
                                    _Q3 = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),
                                           (3,7),(4,5),(4,6),(5,7),(6,7)]
                                    _es = sorted(set((_L[i], _L[j]) for (i, j) in _Q3))
                                    _Q3_EMB_EDGES.append(_es)
                                    _Q3_VERTS.append(_L)
_N_EMB = len(_Q3_EMB_EDGES)

_EDGE_EMB = [[] for _ in range(_N_EDGES)]
for _k in range(_N_EMB):
    for _u, _v in _Q3_EMB_EDGES[_k]:
        _EDGE_EMB[_EDGE_INDEX[(_u, _v)]].append(_k)

_EMB_MASKS = []
for _e in _Q3_EMB_EDGES:
    m = 0
    for u, v in _e:
        m |= (1 << _EDGE_INDEX[(u, v)])
    _EMB_MASKS.append(m)


def _count_bits(x):
    c = 0
    while x:
        c += 1
        x &= x - 1
    return c


def _forbidden_from(mask):
    """cnt[ei] = #embeddings present except possibly edge ei."""
    cnt = [0] * _N_EDGES
    for k in range(_N_EMB):
        miss = _EMB_MASKS[k] & ~mask
        if miss and (miss & (miss - 1)) == 0:
            cnt[miss.bit_length() - 1] += 1
    return cnt


def construct():
    t0 = time.time()
    DEADLINE = t0 + 108.0

    def now():
        return time.time()

    def local_search(mask):
        # greedy add to saturation
        cnt = _forbidden_from(mask)
        for _ in range(8):
            added = False
            for ei in range(_N_EDGES):
                if (mask & (1 << ei)) or cnt[ei] != 0:
                    continue
                mask |= (1 << ei)
                added = True
            if not added:
                break
            cnt = _forbidden_from(mask)

        # edge flips: remove 1, add 2
        for _ in range(1500):
            if now() >= DEADLINE - 3.0:
                break
            improved = False
            base = _count_bits(mask)
            for _ in range(60):
                if now() >= DEADLINE - 3.0:
                    break
                rm = random.randrange(_N_EDGES)
                if not (mask & (1 << rm)):
                    continue
                m2 = mask & ~(1 << rm)
                cnt2 = _forbidden_from(m2)
                adds = []
                for ei in range(_N_EDGES):
                    if (m2 & (1 << ei)) or cnt2[ei] != 0:
                        continue
                    adds.append(ei)
                    if len(adds) == 2:
                        break
                if len(adds) < 2:
                    continue
                m3 = m2 | (1 << adds[0]) | (1 << adds[1])
                if _count_bits(m3) > base:
                    mask = m3
                    improved = True
                    break
            if not improved:
                break
        return mask

    best_mask = 0
    best_count = 0

    for trial in range(4000):
        if now() >= DEADLINE - 4.0:
            break
        density = random.uniform(0.5, 0.85)
        mask = 0
        for ei in range(_N_EDGES):
            if random.random() < density:
                mask |= (1 << ei)
        # remove edges to destroy every fully-present Q3
        for _ in range(500):
            bad = -1
            for k in range(_N_EMB):
                if (_EMB_MASKS[k] & ~mask) == 0:
                    bad = k
                    break
            if bad == -1:
                break
            es = _Q3_EMB_EDGES[bad]
            ei = _EDGE_INDEX[random.choice(es)]
            mask &= ~(1 << ei)
        mask = local_search(mask)
        c = _count_bits(mask)
        if c > best_count:
            best_count = c
            best_mask = mask

    final_mask = local_search(best_mask)
    if _count_bits(final_mask) > best_count:
        best_mask = final_mask

    result = []
    for ei in range(_N_EDGES):
        if best_mask & (1 << ei):
            u, v = _ALL_EDGES[ei]
            result.append([u, v])
    return result
