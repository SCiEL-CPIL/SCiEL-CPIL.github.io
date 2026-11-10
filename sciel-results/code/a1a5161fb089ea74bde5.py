"""Rewritten program: maximize pairwise-disjoint residue classes with distinct moduli 2..128.

Two classes (r1,m1),(r2,m2) are disjoint iff r1 != r2 (mod m1) AND r1 != r2 (mod m2).
We maximize the number of classes via precomputed O(1) compatibility tables,
greedy initialization, randomized restarts, and forward-checking backtracking,
all under an explicit 80s deadline guard (limit is 90s).
"""
import random
import time

MAXIMUM = 128
MODULI = list(range(2, MAXIMUM + 1))
N = len(MODULI)

# TABLES[i][j][b] = list of residues r in [0, MODULI[i]) that are disjoint-compatible
# with the class (b, MODULI[j]).
def _build_tables():
    tables = [[None] * N for _ in range(N)]
    for i, mi in enumerate(MODULI):
        for j, mj in enumerate(MODULI):
            if i == j:
                continue
            per_b = []
            for b in range(mj):
                lst = []
                for r in range(mi):
                    if (r - b) % mi != 0 and (r - b) % mj != 0:
                        lst.append(r)
                per_b.append(lst)
            tables[i][j] = per_b
    return tables

TABLES = _build_tables()

# Fast membership sets for each compatibility list.
_ALLOWED_SET = {}
for _i in range(N):
    for _j in range(N):
        if _i == _j:
            continue
        for _b in range(MODULI[_j]):
            _lst = TABLES[_i][_j][_b]
            _ALLOWED_SET[_lst] = set(_lst)


def _greedy(order, rng):
    chosen = {}
    for i in order:
        mi = MODULI[i]
        cand = set(range(mi))
        for j, bj in chosen.items():
            if not cand:
                break
            allowed = TABLES[i][j][bj]
            aset = _ALLOWED_SET[allowed]
            if len(aset) < len(cand):
                cand = cand.intersection(aset)
            else:
                cand = cand.intersection(set(allowed))
        if not cand:
            continue
        cand_sorted = sorted(cand)
        r = rng.choice(cand_sorted) if rng else cand_sorted[0]
        chosen[i] = r
    return chosen


def _backtrack(order, rng, deadline, max_nodes):
    chosen = {}
    state = {"best": 0, "assign": {}, "nodes": 0}

    def rec(pos):
        if time.time() > deadline or state["nodes"] > max_nodes:
            return
        state["nodes"] += 1
        if pos == len(order):
            if len(chosen) > state["best"]:
                state["best"] = len(chosen)
                state["assign"] = dict(chosen)
            return
        i = order[pos]
        mi = MODULI[i]
        cand = list(range(mi))
        for j, bj in chosen.items():
            allowed = TABLES[i][j][bj]
            aset = _ALLOWED_SET[allowed]
            cand = [r for r in cand if r in aset]
            if not cand:
                break
        if not cand:
            rec(pos + 1)
            return
        rng.shuffle(cand)
        for r in cand:
            chosen[i] = r
            rec(pos + 1)
            del chosen[i]
            if time.time() > deadline or state["nodes"] > max_nodes:
                return

    rec(0)
    return state["best"], state["assign"]


def construct():
    start = time.time()
    deadline = start + 80.0
    rng = random.Random(12345)

    order_asc = list(range(N))
    order_desc = list(range(N - 1, -1, -1))

    best_result = None
    best_count = 0

    for order in (order_asc, order_desc):
        chosen = _greedy(order, rng)
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_result = chosen

    # Backtracking attempts with varied orderings.
    for attempt in range(80):
        if time.time() > deadline:
            break
        order = order_asc[:]
        if attempt % 2 == 1:
            a = rng.randrange(N)
            b = rng.randrange(a, N)
            order[a:b + 1] = order[a:b + 1][::-1]
        max_nodes = 120000 + attempt * 25000
        count, assign = _backtrack(order, rng, deadline, max_nodes)
        if count > best_count:
            best_count = count
            best_result = assign

    # Extra randomized greedy passes to fill remaining time.
    for _ in range(5000):
        if time.time() > deadline:
            break
        order = order_asc[:]
        if rng.random() < 0.5:
            a = rng.randrange(N)
            b = rng.randrange(a, N)
            order[a:b + 1] = order[a:b + 1][::-1]
        chosen = _greedy(order, rng)
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_result = chosen

    if best_result is None:
        best_result = _greedy(order_asc, rng)
        best_count = len(best_result)

    result = [[r, MODULI[i]] for i, r in best_result.items()]
    if not result:
        result = [[0, 2]]
    return result
