"""Rewritten: exact edge-decomposition of K20 into the 20 fixed trees via
unified backtracking with symmetry breaking + heuristic ordering, plus a
greedy-partial fallback. Returns 20 entries (None or list of k distinct ints)."""
import time
import random
from collections import deque


def instance_trees(n=20, seed=611):
    rng = Random(seed) if False else _Random(seed)
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


# Use the standard library Random exactly as in the task's generator.
from random import Random as _Random


def _root_tree(k, edges):
    if k == 1:
        return None
    adj = {v: [] for v in range(k)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    root = None
    for v in range(k):
        if len(adj[v]) == 1:
            root = v
            break
    if root is None:
        root = 0
    parent = {root: -1}
    children = {v: [] for v in range(k)}
    order = []
    visited = {root}
    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for w in adj[u]:
            if w not in visited:
                visited.add(w)
                parent[w] = u
                children[u].append(w)
                q.append(w)
    return root, parent, children, order


def _verify(result, trees, N=20):
    seen = set()
    total = 0
    for k in range(1, 21):
        e = result[k - 1]
        if e is None:
            continue
        if len(e) != k:
            return False, 0
        for x in e:
            if type(x) is not int or x < 0 or x >= N:
                return False, 0
        if len(set(e)) != k:
            return False, 0
        for u, v in trees[k - 1]:
            a, b = e[u], e[v]
            key = (a, b) if a < b else (b, a)
            if key in seen:
                return False, 0
            seen.add(key)
            total += 1
    return True, total


def construct():
    global_start = time.time()
    GLOBAL_DEADLINE = global_start + 110.0
    N = 20
    trees = instance_trees()

    info = {}
    for k in range(2, 21):
        info[k] = _root_tree(k, trees[k - 1])

    def make_used():
        return [[False] * N for _ in range(N)]

    def avail_deg(used, h):
        c = 0
        row = used[h]
        for x in range(N):
            if x != h and not row[x]:
                c += 1
        return c

    def greedy_partial(order_trees, seed):
        rng = random.Random(seed)
        used = make_used()
        mapping = {k: [-1] * k for k in range(1, 21)}
        local_used = {k: set() for k in range(1, 21)}
        result = [None] * 20
        total = 0

        def embed(k):
            root, parent, children, preorder = info[k]
            L = len(preorder)

            def rec(pi):
                if pi == L:
                    return True
                v = preorder[pi]
                if v == root:
                    cands = [h for h in range(N) if h not in local_used[k]]
                else:
                    p = parent[v]
                    ph = mapping[k][p]
                    cands = [h for h in range(N)
                             if h not in local_used[k] and not used[ph][h]]
                if not cands:
                    return False
                cands.sort(key=lambda h: (avail_deg(used, h), rng.random()))
                for h in cands:
                    mapping[k][v] = h
                    local_used[k].add(h)
                    if v != root:
                        p = parent[v]
                        ph = mapping[k][p]
                        used[ph][h] = used[h][ph] = True
                    if rec(pi + 1):
                        return True
                    if v != root:
                        p = parent[v]
                        ph = mapping[k][p]
                        used[ph][h] = used[h][ph] = False
                    local_used[k].discard(h)
                    mapping[k][v] = -1
                return False

            return rec(0)

        for k in order_trees:
            if embed(k):
                result[k - 1] = [mapping[k][v] for v in range(k)]
                total += len(trees[k - 1])
            else:
                result[k - 1] = None
        return result, total

    def full_backtrack(order_trees, seed, node_budget, time_budget):
        rng = random.Random(seed)
        used = make_used()
        mapping = {k: [-1] * k for k in range(1, 21)}
        local_used = {k: set() for k in range(1, 21)}
        result = [None] * 20
        nodes = 0
        deadline = time.time() + time_budget

        def rec(ti, pi):
            nonlocal nodes
            if time.time() > deadline:
                return False
            nodes += 1
            if nodes > node_budget:
                return False
            if ti == len(order_trees):
                return True
            k = order_trees[ti]
            root, parent, children, preorder = info[k]
            L = len(preorder)
            if pi == L:
                result[k - 1] = [mapping[k][v] for v in range(k)]
                return rec(ti + 1, 0)
            v = preorder[pi]
            if v == root:
                if ti == 0 and pi == 0:
                    cands = [0]  # symmetry break: K20 is vertex-transitive
                else:
                    cands = [h for h in range(N) if h not in local_used[k]]
            else:
                p = parent[v]
                ph = mapping[k][p]
                cands = [h for h in range(N)
                         if h not in local_used[k] and not used[ph][h]]
            if not cands:
                return False
            cands.sort(key=lambda h: (avail_deg(used, h), rng.random()))
            for h in cands:
                mapping[k][v] = h
                local_used[k].add(h)
                if v != root:
                    p = parent[v]
                    ph = mapping[k][p]
                    used[ph][h] = used[h][ph] = True
                if rec(ti, pi + 1):
                    return True
                if v != root:
                    p = parent[v]
                    ph = mapping[k][p]
                    used[ph][h] = used[h][ph] = False
                local_used[k].discard(h)
                mapping[k][v] = -1
            return False

        if rec(0, 0):
            return result
        return None

    # Baseline greedy partial (fast).
    best_result = [None] * 20
    best_total = 0
    desc = list(range(20, 1, -1))
    asc = list(range(2, 21))
    for seed in (0, 1, 2):
        if time.time() > GLOBAL_DEADLINE - 25:
            break
        r, t = greedy_partial(desc, seed)
        ok, tv = _verify(r, trees)
        if ok and t > best_total:
            best_total = t
            best_result = r
        if t == 190:
            return r

    # Try exact full backtracking to reach 190.
    configs = [(desc, 0), (desc, 1), (asc, 0), (desc, 2)]
    for order, seed in configs:
        if time.time() > GLOBAL_DEADLINE - 8:
            break
        tb = min(28.0, max(4.0, GLOBAL_DEADLINE - time.time() - 6.0))
        res = full_backtrack(order, seed, node_budget=2_500_000, time_budget=tb)
        if res is not None:
            ok, tv = _verify(res, trees)
            if ok and tv == 190:
                return res

    # Extra greedy attempts with varied orders for a better partial.
    for seed in (0, 1, 2, 3, 4, 5):
        if time.time() > GLOBAL_DEADLINE - 6:
            break
        for order in (asc, random.Random(seed).sample(range(2, 21), 19)):
            r, t = greedy_partial(order, seed)
            ok, tv = _verify(r, trees)
            if ok and t > best_total:
                best_total = t
                best_result = r
            if t == 190:
                return r

    # Final safety: ensure a valid result.
    ok, tv = _verify(best_result, trees)
    if not ok:
        # Fallback: embed only T20 by identity (always valid, 19 edges).
        best_result = [None] * 19 + [list(range(20))]
    out = []
    for entry in best_result:
        if entry is None:
            out.append(None)
        else:
            out.append([int(x) for x in entry])
    return out
