"""Combined program: greedy-partial seeding + unified exact backtracking with
symmetry breaking, heuristic ordering, and a verify/repair loop."""
import time
import random
from collections import deque
from random import Random


def instance_trees(n=20, seed=611):
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def _root_tree(k, edges):
    if k == 1:
        return None
    adj = {v: [] for v in range(k)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Pick a leaf as root for better BFS ordering
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
    GLOBAL_DEADLINE = global_start + 140.0  # safety margin before 150s kill
    N = 20
    trees = instance_trees(20, 611)

    # Precompute tree structures
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
        """Fast greedy embedding to get a partial solution."""
        rng = random.Random(seed)
        used = make_used()
        mapping = {k: [-1] * k for k in range(1, 21)}
        local_used = {k: set() for k in range(1, 21)}
        result = [None] * 20
        total = 0

        def embed(k):
            if k == 1:
                # Single vertex, no edges
                for h in range(N):
                    if h not in local_used[1]:
                        mapping[1][0] = h
                        local_used[1].add(h)
                        result[0] = [h]
                        return True
                return False
            if k == 2:
                # Single edge
                for u in range(N):
                    if u in local_used[2]:
                        continue
                    for w in range(N):
                        if w == u or w in local_used[2]:
                            continue
                        if not used[u][w]:
                            mapping[2][0] = u
                            mapping[2][1] = w
                            local_used[2].add(u)
                            local_used[2].add(w)
                            used[u][w] = used[w][u] = True
                            result[1] = [u, w]
                            return True
                return False
            
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
                # Sort by available degree (most constrained first)
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
                if k == 1:
                    total += 0
                else:
                    total += k - 1
            else:
                result[k - 1] = None
        return result, total

    def full_backtrack(order_trees, seed, node_budget, time_budget):
        """Exact backtracking to find a complete edge-decomposition."""
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
            
            if k == 1:
                # Single vertex
                for h in range(N):
                    if h not in local_used[1]:
                        mapping[1][0] = h
                        local_used[1].add(h)
                        result[0] = [h]
                        if rec(ti + 1, 0):
                            return True
                        local_used[1].discard(h)
                        mapping[1][0] = -1
                return False
            
            if k == 2:
                # Single edge
                for u in range(N):
                    if u in local_used[2]:
                        continue
                    for w in range(N):
                        if w == u or w in local_used[2]:
                            continue
                        if not used[u][w]:
                            mapping[2][0] = u
                            mapping[2][1] = w
                            local_used[2].add(u)
                            local_used[2].add(w)
                            used[u][w] = used[w][u] = True
                            result[1] = [u, w]
                            if rec(ti + 1, 0):
                                return True
                            used[u][w] = used[w][u] = False
                            local_used[2].discard(u)
                            local_used[2].discard(w)
                            mapping[2][0] = -1
                            mapping[2][1] = -1
                return False

            root, parent, children, preorder = info[k]
            L = len(preorder)
            if pi == L:
                result[k - 1] = [mapping[k][v] for v in range(k)]
                return rec(ti + 1, 0)
            v = preorder[pi]
            if v == root:
                if ti == 0 and pi == 0:
                    cands = [0]  # Symmetry break: K20 is vertex-transitive
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

    # Try greedy partial solutions first (fast)
    best_result = [None] * 20
    best_total = 0
    desc = list(range(20, 1, -1))  # T20 down to T2
    asc = list(range(2, 21))       # T2 up to T20
    
    for seed in (0, 1, 2, 3, 4):
        if time.time() > GLOBAL_DEADLINE - 30:
            break
        r, t = greedy_partial(desc, seed)
        ok, tv = _verify(r, trees)
        if ok and t > best_total:
            best_total = t
            best_result = r
        if t == 190:
            return r

    # Try exact full backtracking to reach 190
    configs = [
        (desc, 0, 5_000_000),
        (desc, 1, 5_000_000),
        (asc, 0, 3_000_000),
        (desc, 2, 4_000_000),
        (desc, 3, 4_000_000),
    ]
    
    for order, seed, budget in configs:
        elapsed = time.time() - global_start
        remaining = GLOBAL_DEADLINE - elapsed
        if remaining < 10:
            break
        tb = min(25.0, max(3.0, remaining - 8.0))
        res = full_backtrack(order, seed, node_budget=budget, time_budget=tb)
        if res is not None:
            ok, tv = _verify(res, trees)
            if ok and tv > best_total:
                best_total = tv
                best_result = res
                if tv == 190:
                    return res

    # Final fallback: return best partial solution found
    # Ensure all entries are either None or valid lists
    final = [None] * 20
    for i in range(20):
        if best_result[i] is not None:
            final[i] = list(best_result[i])
    
    # Verify the final result
    ok, tv = _verify(final, trees)
    if not ok:
        # If verification fails, try to return a simpler valid result
        # At minimum, embed T1 and T2 if possible
        final = [None] * 20
        used = make_used()
        # T1: single vertex
        final[0] = [0]
        # T2: single edge
        for u in range(20):
            for w in range(u + 1, 20):
                if not used[u][w]:
                    final[1] = [u, w]
                    used[u][w] = used[w][u] = True
                    break
            if final[1] is not None:
                break
    
    return final
