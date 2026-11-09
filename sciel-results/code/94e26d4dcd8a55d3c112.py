import time
import random
from itertools import combinations

# Precompute LCM(1..24) based integer weights: weight(x) = L // x
# 1/x = sum(1/y)  <=>  weight(x) = sum(weight(y))
_L = 5354228880
W = [0] + [_L // x for x in range(1, 25)]
UNIVERSE = list(range(1, 25))
UNIV_SET = frozenset(UNIVERSE)

# ---- enumerate all minimal forbidden subsets (hyperedges) -----------------
def _enumerate_edges():
    edges = []
    for x in UNIVERSE:
        t = W[x]
        others = [y for y in UNIVERSE if y != x]
        # candidate subsets must have total weight == t and each weight <= t
        cand = [y for y in others if W[y] <= t]
        n = len(cand)
        # bound subset size by min weight
        minw = min(W[y] for y in cand)
        maxk = min(n, t // minw)
        for k in range(1, maxk + 1):
            for combo in combinations(cand, k):
                if sum(W[y] for y in combo) == t:
                    edges.append(frozenset(combo))
    return edges

EDGES = _enumerate_edges()
# map element -> list of edges it participates in
EDGE_OF = {x: [] for x in UNIVERSE}
for e in EDGES:
    for x in e:
        EDGE_OF[x].append(e)

# ---- smooth (LP-relaxation) upper bound on independent-set size ----------
def _lp_bound(S):
    """Upper bound on #independent-set elements within S via LP relaxation.
    Smooth surrogate: fractional packing max sum v_i s.t. sum_{i in e} v_i <= 1, 0<=v<=1.
    Greedy fractional solve gives an admissible upper bound."""
    S = list(S)
    if not S:
        return 0.0
    # candidate edges restricted to S (non-empty intersection)
    cand = [e for e in EDGES if e & S]
    # greedy: repeatedly pick the edge with most remaining capacity usage,
    # distribute its budget evenly among its still-unbounded members
    v = {i: 1.0 for i in S}
    cap = {id(e): 1.0 for e in cand}
    changed = True
    guard = 0
    while changed and guard < 5000:
        changed = False
        guard += 1
        # for each edge, if sum v over members > cap, scale them down
        for e in cand:
            mem = [i for i in e if i in v]
            if not mem:
                continue
            s = sum(v[i] for i in mem)
            c = cap[id(e)]
            if s > c + 1e-12:
                scale = c / s
                for i in mem:
                    v[i] *= scale
                changed = True
    return sum(v.values())

# ---- conflict test (fast) -------------------------------------------------
def _conflicts(S):
    """Return (x, T) if x in S is subsumed by subset T of S\{x}, else None."""
    Sset = set(S)
    for x in S:
        t = W[x]
        # candidate subset of others with weights <= t
        cand = [y for y in S if y != x and W[y] <= t]
        # meet-in-the-middle subset sum
        n = len(cand)
        if n == 0:
            continue
        mid = n // 2
        left = cand[:mid]
        right = cand[mid:]
        left_sums = {0: frozenset()}
        for y in left:
            wy = W[y]
            add = {}
            for s, fs in left_sums.items():
                ns = s + wy
                if ns <= t:
                    add[ns] = fs | {y}
            for ns, fs in add.items():
                if ns not in left_sums:
                    left_sums[ns] = fs
        right_sums = {0: frozenset()}
        for y in right:
            wy = W[y]
            add = {}
            for s, fs in right_sums.items():
                ns = s + wy
                if ns <= t:
                    add[ns] = fs | {y}
            for ns, fs in add.items():
                if ns not in right_sums:
                    right_sums[ns] = fs
        for s, fs in left_sums.items():
            rem = t - s
            if rem in right_sums:
                T = fs | right_sums[rem]
                if T:
                    return x, T
    return None

def _is_valid(S):
    return _conflicts(S) is None

# ---- branch and bound -----------------------------------------------------
def _bnb(deadline):
    global _best, _best_size
    _best = frozenset()
    _best_size = 0
    nodes = [0]

    def improve_add(S):
        # greedily add any element that keeps validity
        Sset = set(S)
        for y in UNIVERSE:
            if y in Sset:
                continue
            Sset.add(y)
            if _is_valid(Sset):
                Sset.add(y)
            else:
                Sset.discard(y)
        return frozenset(Sset)

    def dfs(S, avail):
        nodes[0] += 1
        if time.time() > deadline:
            return
        # prune: even adding all avail can't beat best
        if len(S) + len(avail) <= _best_size:
            return
        # smooth bound
        if len(S) + _lp_bound(avail) <= _best_size + 1e-9:
            return
        if not avail:
            if len(S) > _best_size:
                _best_size = len(S)
                _best = S
            return
        # pick element with most conflicts in avail (most constrained)
        def conf_count(y):
            c = 0
            for e in EDGE_OF[y]:
                if e & avail:
                    c += 1
            return c
        y = max(avail, key=lambda z: (conf_count(z), -W[z]))
        # branch include y
        new_avail = avail - {y}
        # exclude y from consideration: S stays, y removed from avail
        # include branch
        S1 = S | {y}
        # check y against S1\{y} (only new conflicts involving y)
        ok = True
        for e in EDGE_OF[y]:
            if e - {y} <= S1:
                ok = False
                break
        if ok:
            # also ensure no existing x in S becomes subsumed by S1 (y added)
            for x in S1 - {y}:
                for e in EDGE_OF[x]:
                    if e - {x} <= S1:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            dfs(S1, new_avail)
        # exclude branch
        dfs(S, new_avail)

    # initial greedy seed
    seed = improve_add(frozenset())
    _best_size = len(seed)
    _best = seed

    # order elements: include small first tends to grow; start with all available
    dfs(frozenset(), frozenset(UNIVERSE))

    # local improvement from best
    b = set(_best)
    for y in UNIVERSE:
        if y in b:
            continue
        b.add(y)
        if _is_valid(b):
            pass
        else:
            b.discard(y)
    _best = frozenset(b)
    _best_size = len(_best)
    return _best

def construct():
    start = time.time()
    deadline = start + 80  # safety margin under 90s
    best = _bnb(deadline)
    if not best:
        return [1]
    return list(best)

if __name__ == "__main__":
    r = construct()
    print(len(r), sorted(r))
