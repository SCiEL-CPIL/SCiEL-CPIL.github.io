"""Linear triple systems on 16 points: minimize independence number."""
import time
import random
import itertools
import sys
sys.setrecursionlimit(100000)

N = 16
DEADLINE = 140.0  # wall-clock safety margin under 150s limit

# Precompute all 560 triples and pairwise compatibility
_ALL_TRIPLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            _ALL_TRIPLES.append((a, b, c))

_M = len(_ALL_TRIPLES)
_MASKS = [((1 << a) | (1 << b) | (1 << c)) for (a, b, c) in _ALL_TRIPLES]
# compat[i] = list of j != i such that masks intersect in <=1 bit
_COMPAT = []
for i in range(_M):
    mi = _MASKS[i]
    lst = []
    for j in range(_M):
        if i == j:
            continue
        if (mi & _MASKS[j]).bit_count() <= 1:
            lst.append(j)
    _COMPAT.append(lst)

# Precompute all point subsets as bitmasks for independence check
_ALL_SUBSETS = [0]
for i in range(N):
    _ALL_SUBSETS += [s | (1 << i) for s in _ALL_SUBSETS]


def _independence_number(triples):
    """Exact largest subset of points containing no triple."""
    t_masks = [((1 << a) | (1 << b) | (1 << c)) for (a, b, c) in triples]
    best = 0
    deadline = time.time() + 8.0
    # subsets sorted by descending size
    subs = sorted(_ALL_SUBSETS, key=lambda s: s.bit_count(), reverse=True)
    # quick upper bound: if no triple, independence = N
    if not triples:
        return N
    for s in subs:
        sz = s.bit_count()
        if sz <= best:
            break
        if time.time() > deadline:
            break
        # check if s contains any triple
        contains = False
        for tm in t_masks:
            if (tm & s) == tm:
                contains = True
                break
        if not contains:
            best = sz
    return best


def _greedy_build(rng, n_points=16, target=None):
    """Randomized greedy: repeatedly add a random compatible triple."""
    used = [False] * _M
    chosen = []
    # start by picking a random first triple
    first = rng.randrange(_M)
    used[first] = True
    chosen.append(first)
    # available candidates: those compatible with all chosen
    avail = list(_COMPAT[first])
    while avail:
        # pick a random candidate
        idx = rng.randrange(len(avail))
        t = avail[idx]
        used[t] = True
        chosen.append(t)
        # remove incompatible from avail
        bad = set(_COMPAT[t])
        new_avail = []
        for x in avail:
            if x != idx and x not in bad:
                new_avail.append(x)
        # careful: we removed t itself
        avail = new_avail
        if target is not None and len(chosen) >= target:
            break
    return [_ALL_TRIPLES[i] for i in chosen]


def _local_improve(triples, rng, deadline):
    """Try to add more triples or swap to increase count (proxy for lower alpha)."""
    used = [False] * _M
    for t in triples:
        used[_ALL_TRIPLES.index(t)] = True
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # find a compatible unused triple to add
        for i in range(_M):
            if used[i]:
                continue
            ok = True
            for j in range(_M):
                if used[j] and j != i and (i not in _COMPAT[j]):
                    ok = False
                    break
            if ok:
                used[i] = True
                triples.append(_ALL_TRIPLES[i])
                improved = True
                break
    return triples


def construct():
    start = time.time()
    rng = random.Random()
    best = None
    best_alpha = N + 1
    best_count = -1
    deadline = start + DEADLINE

    # Phase 1: many randomized greedy builds
    attempt = 0
    while time.time() < deadline - 20.0:
        attempt += 1
        triples = _greedy_build(rng)
        # local improvement
        triples = _local_improve(triples, rng, deadline - 20.0)
        count = len(triples)
        # alpha estimate: more triples -> smaller independence
        # compute exact alpha only for promising candidates
        if count > best_count:
            alpha = _independence_number(triples)
            if alpha < best_alpha:
                best_alpha = alpha
                best = triples
                best_count = count
        # early stop if we hit a very low alpha
        if best_alpha <= 6:
            break

    # Phase 2: if time remains, try to refine best with more random restarts
    if best is None:
        # fallback: simple disjoint triples
        best = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        best_alpha = _independence_number(best)

    # final safety: ensure we return within deadline
    return best
