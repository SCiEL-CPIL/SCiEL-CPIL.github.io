"""Rewritten program: greedy randomized edge-disjoint packing of T1..T20 into K20."""
from random import Random
from time import time

# ---------- fixed trees (deterministic, seed=611) ----------
def _make_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

_TREES = _make_trees()
_N = 20
# order to embed: bigger (more edges) first -> harder to fit later
_ORDER = sorted(range(_N), key=lambda i: len(_TREES[i]), reverse=True)


def _greedy(rng, trees, order, used):
    """Embed each tree in `order` greedily. Returns (ok, result_list, value)."""
    res = [None] * _N
    for ti in order:
        edges = trees[ti]
        k = ti + 1
        # candidate pool: all vertices, but prefer those with fewer used edges
        verts = list(range(_N))
        rng.shuffle(verts)
        # map tree-vertex -> host
        assign = {}
        ok = True
        for (a, b) in edges:
            placed = []
            # try to place a and b on free host vertices (not yet used by this tree)
            used_here = set(assign.values())
            pool = [v for v in verts if v not in used_here]
            if not pool:
                ok = False
                break
            # pick a host for a (random), then for b (random, distinct)
            ha = pool[rng.randrange(len(pool))]
            pool2 = [v for v in pool if v != ha]
            if not pool2:
                ok = False
                break
            hb = pool2[rng.randrange(len(pool2))]
            # check edge-disjointness
            key = (ha, hb) if ha < hb else (hb, ha)
            if key in used:
                ok = False
                break
            assign[a] = ha
            assign[b] = hb
            used.add(key)
        if not ok:
            return False, None, 0
        # build list: position i = host of vertex i
        lst = [int(assign[i]) for i in range(k)]
        # sanity: distinct
        if len(set(lst)) != k:
            return False, None, 0
        res[ti] = lst
    # compute value = total embedded edges
    val = sum(len(trees[i]) for i in range(_N) if res[i] is not None)
    return True, res, val


def construct():
    deadline = time() + 110.0   # safety margin vs 120/150s
    base_seed = 12345
    best_val = -1
    best_res = None

    # try a few seeds / strategies; keep best
    attempt = 0
    while time() < deadline:
        rng = Random(base_seed + attempt)
        # occasionally try a different order (still mostly size-desc)
        order = list(_ORDER)
        if attempt % 5 == 0:
            rng.shuffle(order)
        # used-edge set
        used = set()
        ok, res, val = _greedy(rng, _TREES, order, used)
        if ok and val > best_val:
            best_val = val
            best_res = res
        attempt += 1
        # if we already packed all trees, that's the maximum possible
        if best_val == 190:
            break

    # Fallback: identity embedding of T20 only (always valid, value 19)
    if best_res is None:
        best_res = [None] * 19 + [list(range(_N))]
        best_val = 19

    # ensure pure Python ints and lists
    out = []
    for i in range(_N):
        e = best_res[i]
        if e is None:
            out.append(None)
        else:
            out.append([int(x) for x in e])
    return out
