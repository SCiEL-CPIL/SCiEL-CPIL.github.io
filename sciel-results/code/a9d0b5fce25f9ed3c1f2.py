import time
import random

def construct():
    n = 15
    start = time.time()
    timeout = 55.0

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_idx = {}
    for idx, (u, v) in enumerate(all_pairs):
        pair_idx[(u, v)] = idx

    def make_adj(edges_set):
        adj = [set() for _ in range(n)]
        for (u, v) in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_valid(edges_set):
        adj = make_adj(edges_set)
        deg = [len(adj[i]) for i in range(n)]
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def greedy_add(edges_set):
        edges = set(edges_set)
        adj = make_adj(edges)
        deg = [len(adj[i]) for i in range(n)]
        changed = True
        while changed:
            changed = False
            for (u, v) in all_pairs:
                if (u, v) in edges:
                    continue
                if not is_valid(edges | {(u, v)}):
                    continue
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                changed = True
        return edges

    def local_search(edges_set, max_iter=200):
        cur = set(edges_set)
        adj = make_adj(cur)
        deg = [len(adj[i]) for i in range(n)]
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            # Try adding one edge
            for (u, v) in all_pairs:
                if (u, v) in cur:
                    continue
                trial = cur | {(u, v)}
                if is_valid(trial):
                    cur = trial
                    adj = make_adj(cur)
                    deg = [len(adj[i]) for i in range(n)]
                    improved = True
                    break
            if improved:
                continue
            # Try remove one, then greedy add
            for e in list(cur):
                if time.time() - start > timeout:
                    break
                trial = cur - {e}
                trial = greedy_add(trial)
                if len(trial) > len(cur):
                    cur = trial
                    adj = make_adj(cur)
                    deg = [len(adj[i]) for i in range(n)]
                    improved = True
                    break
        return cur

    # Start with K_{7,8}
    L = list(range(7))
    R = list(range(7, 15))
    base = set()
    for u in L:
        for v in R:
            base.add((u, v))

    best = None
    best_count = -1

    # First, greedy from base
    res = greedy_add(base)
    best = res
    best_count = len(res)

    # Then local search
    res = local_search(base)
    if len(res) > best_count:
        best = res
        best_count = len(res)

    # Try random restarts with different initial graphs
    random.seed(12345)
    for restart in range(30):
        if time.time() - start > timeout:
            break
        # Random dense graph
        edges = set()
        for (u, v) in all_pairs:
            if random.random() < 0.7:
                edges.add((u, v))
        # Make sure it's valid, if not, remove edges
        while not is_valid(edges):
            e = random.choice(list(edges))
            edges.remove(e)
        # Greedy add
        res = greedy_add(edges)
        # Local search
        res = local_search(res, max_iter=50)
        if len(res) > best_count:
            best = res
            best_count = len(res)

    # Final local search on best
    if best is not None and time.time() - start < timeout - 5:
        res = local_search(best, max_iter=100)
        if len(res) > best_count:
            best = res
            best_count = len(res)

    # Ensure return type is correct
    result = []
    for u, v in best:
        result.append([int(u), int(v)])
    return result
