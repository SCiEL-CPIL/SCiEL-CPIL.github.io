import time
import random
from itertools import combinations


def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _embed_tree(adj, tree_edges, n, rng, time_limit, max_nodes):
    """Embed tree (tree_edges on 0..k-1) into host graph adj (n nodes).
    Returns a list of k distinct host vertices (image of tree vertex i), or None."""
    if not tree_edges:
        return [0]

    k = len(tree_edges)
    if k == 2:
        for u in range(n):
            if adj[u]:
                return [u, min(adj[u])]
        return None

    # host adjacency of each tree vertex (as a frozenset of candidate host vertices)
    cand = [set(range(n)) for _ in range(k)]

    # adjacency among tree vertices
    tree_adj = [set() for _ in range(k)]
    for a, b in tree_edges:
        tree_adj[a].add(b)
        tree_adj[b].add(a)

    # order: highest tree-degree first, then smallest candidate set
    order = list(range(k))
    order.sort(key=lambda v: (-len(tree_adj[v]), len(cand[v])))

    # pick a root with the fewest candidates
    root = min(order, key=lambda v: len(cand[v]))

    # build tree rooted at root: parent + children lists
    parent = [-1] * k
    children = [set() for _ in range(k)]
    stack = [root]
    parent[root] = root
    while stack:
        v = stack.pop()
        for nb in tree_adj[v]:
            if parent[nb] == -1:
                parent[nb] = v
                children[v].add(nb)
                stack.append(nb)

    # candidate sets: host vertices adjacent to the parent's host vertex
    # We'll do a DFS over the tree.
    assigned = [None] * k
    used = [False] * n
    # choose root host
    root_cands = list(cand[root])
    rng.shuffle(root_cands)

    # Precompute, for each tree vertex, its children (to prune candidate sets)
    # We'll assign in a DFS from root.

    def dfs(v, host_v):
        assigned[v] = host_v
        used[host_v] = True
        chs = list(children[v])
        rng.shuffle(chs)
        for c in chs:
            # candidates for c: host neighbors of host_v, unused, in cand[c]
            cands = [h for h in adj[host_v] if not used[h] and h in cand[c]]
            if not cands:
                used[host_v] = False
                assigned[v] = None
                return False
            rng.shuffle(cands)
            # heuristic: try candidates with fewer options for their children first
            def score(h):
                s = 0
                for cc in children[c]:
                    s += sum(1 for x in adj[h] if not used[x] and x in cand[cc])
                return s
            cands.sort(key=score)
            for h in cands:
                if dfs(c, h):
                    return True
            used[host_v] = False
            assigned[v] = None
            return False
        return True

    for rv in root_cands:
        used = [False] * n
        assigned = [None] * k
        if dfs(root, rv):
            return [int(assigned[i]) for i in range(k)]
    return None


def _hamiltonian_cycle(adj, n):
    """Find a Hamiltonian cycle in host graph adj (n nodes). Returns list of n
    vertices in cycle order, or None."""
    if n < 3:
        return list(range(n)) if n <= 2 else None
    for start in range(n):
        path = [start]
        visited = {start}
        if _hc_dfs(adj, n, start, path, visited):
            return path
    return None


def _hc_dfs(adj, n, start, path, visited):
    if len(path) == n:
        return start in adj[path[-1]]
    last = path[-1]
    nbrs = [v for v in adj[last] if v not in visited]
    # Warnsdorff: fewest onward moves first
    nbrs.sort(key=lambda v: sum(1 for x in adj[v] if x not in visited))
    for v in nbrs:
        path.append(v)
        visited.add(v)
        if _hc_dfs(adj, n, start, path, visited):
            return True
        path.pop()
        visited.remove(v)
    return False


def _try_once(trees, n, seed, deadline):
    rng = random.Random(seed)
    remaining = set()
    for u in range(n):
        for v in range(u + 1, n):
            remaining.add((u, v))

    result = [None] * n
    # embed small trees first (indices 1..n-1), leaving T_n (Hamiltonian) last
    order = list(range(1, n))
    rng.shuffle(order)
    # sort by tree size ascending to place easy ones first
    order.sort(key=lambda i: len(trees[i]))

    for i in order:
        if time.time() > deadline:
            break
        te = trees[i]
        if not te:
            result[i] = None
            continue
        adj = _build_adj(list(remaining), n)
        # a few random restarts for this tree
        found = None
        for attempt in range(30):
            if time.time() > deadline:
                break
            emb = _embed_tree(adj, te, n, rng, deadline, max_nodes=2000000)
            if emb is not None:
                found = emb
                break
        if found is None:
            # cannot place this tree now; skip it (leave None) but keep going
            continue
        # consume edges
        for a, b in te:
            u, v = found[a], found[b]
            e = (min(u, v), max(u, v))
            if e not in remaining:
                # should not happen
                continue
            remaining.discard(e)
        result[i] = [int(x) for x in found]

    # Hamiltonian cycle for the last tree (T_n) on remaining edges
    i = n - 1
    if time.time() <= deadline:
        te = trees[i]
        if te:
            adj = _build_adj(list(remaining), n)
            cyc = _hamiltonian_cycle(adj, n)
            if cyc is not None:
                # verify all cycle edges are in remaining
                ok = True
                for idx in range(n):
                    u, v = cyc[idx], cyc[(idx + 1) % n]
                    e = (min(u, v), max(u, v))
                    if e not in remaining:
                        ok = False
                        break
                if ok:
                    result[i] = [int(x) for x in cyc]
    return result


def _value(result, trees):
    total = 0
    for i, emb in enumerate(result):
        if emb is None:
            continue
        te = trees[i]
        # count embedded edges (should equal len(te))
        total += len(te)
    return total


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s
    n = 20
    trees = instance_trees(n, 611)

    best = [None] * n
    best_val = 0
    # include the trivial seed solution as baseline
    seed_res = [None] * (n - 1) + [list(range(n))]
    best = seed_res
    best_val = _value(best, trees)

    rng_outer = random.Random(12345)
    base_seed = 611
    trial = 0
    while time.time() < deadline - 5.0:
        s = (base_seed + trial * 7919) % (2**31)
        trial += 1
        res = _try_once(trees, n, s, deadline - 3.0)
        val = _value(res, trees)
        if val > best_val:
            best_val = val
            best = res
            if best_val >= 190:
                break
    return best
