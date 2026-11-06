import time
import random
import itertools
from collections import defaultdict

_N = 16
_PAIRS = list(itertools.combinations(range(_N), 2))
_NPAIRS = len(_PAIRS)          # 120
_PAIR_INDEX = {p: k for k, p in enumerate(_PAIRS)}
_QUADS = list(itertools.combinations(range(_N), 4))
_NQUADS = len(_QUADS)          # 1820
_QUAD_EDGE_MASK = [[_PAIR_INDEX[(a, b)] for a, b in
                    itertools.combinations(q, 2)] for q in _QUADS]
_DEADLINE = 150.0              # wall-clock kill limit
_MARGIN = 15.0                 # safety margin before returning


def _validity(c):
    """Return (ok, min_distinct) for colour list c of length 120."""
    min_d = 120
    for em in _QUAD_EDGE_MASK:
        s = set(c[i] for i in em)
        d = len(s)
        if d < 5:
            return False, d
        if d < min_d:
            min_d = d
    return True, min_d


def _build_initial():
    """Valid 30-colour construction: 4 groups of 4, disjoint intra + inter colours."""
    c = [0] * _NPAIRS
    groups = [list(range(4 * g, 4 * g + 4)) for g in range(4)]
    # intra colours: 6 per group, base 6*g  -> 24 colours
    for g in range(4):
        for e, (a, b) in enumerate(itertools.combinations(groups[g], 2)):
            c[_PAIR_INDEX[(a, b)]] = 6 * g + e
    # inter colours: one per group-pair, base 24 + gp -> 6 colours
    gp = 0
    for g1 in range(4):
        for g2 in range(g1 + 1, 4):
            for a in groups[g1]:
                for b in groups[g2]:
                    c[_PAIR_INDEX[(a, b)]] = 24 + gp
            gp += 1
    ok, _ = _validity(c)
    assert ok
    return c


def construct():
    random.seed(20240517)
    t0 = time.time()
    best = _build_initial()
    best_k = len(set(best))
    assert best_k == 30

    def elapsed():
        return time.time() - t0

    def try_merge(c, k):
        """Attempt to merge a random colour into another, checking validity.
        Returns new list if a valid merge is found, else None."""
        colours = sorted(set(c))
        if len(colours) <= k:
            return None
        # pick a source colour to eliminate (prefer rarest)
        counts = defaultdict(int)
        for v in c:
            counts[v] += 1
        # candidate source colours (those that appear)
        src = random.choice([x for x in colours if counts[x] > 0])
        # choose a target colour different from src
        targets = [x for x in colours if x != src]
        random.shuffle(targets)
        for tgt in targets:
            nc = list(c)
            for i in range(_NPAIRS):
                if nc[i] == src:
                    nc[i] = tgt
            ok, _ = _validity(nc)
            if ok:
                return nc
        return None

    # Greedy colour-merging local search, deadline-guarded.
    cur = list(best)
    cur_k = best_k
    stall = 0
    while elapsed() < _DEADLINE - _MARGIN:
        if cur_k <= 13:
            break
        nc = try_merge(cur, cur_k)
        if nc is not None:
            cur = nc
            cur_k = len(set(cur))
            stall = 0
            if cur_k < best_k:
                best = list(cur)
                best_k = cur_k
        else:
            stall += 1
            if stall > 40:
                # restart from best with a fresh randomisation of a few edges
                cur = list(best)
                # lightly perturb: recolor a few edges to random used colours
                used = list(set(cur))
                for _ in range(8):
                    i = random.randrange(_NPAIRS)
                    cur[i] = random.choice(used)
                # ensure still valid, else revert
                ok, _ = _validity(cur)
                if not ok:
                    cur = list(best)
                stall = 0
            if elapsed() >= _DEADLINE - _MARGIN:
                break

    # Final safety: return best valid solution (guaranteed valid by construction).
    ok, _ = _validity(best)
    assert ok
    return best
