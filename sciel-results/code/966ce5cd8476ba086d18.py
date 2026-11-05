"""Decompose three fixed graphs into few cycles + edges (Erdős-Gallai)."""
import time
import random
import sys

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, n, deadline, rnd):
    """Greedy cycle extraction. Returns list of pieces (edge or cycle)."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)
    progress = 0
    guard = 0
    guard_limit = 20000000

    while remaining > 0:
        if time.time() > deadline:
            break
        guard += 1
        if guard > guard_limit:
            break

        # pick a start vertex (randomize to vary across restarts)
        start = None
        for _ in range(400):
            v = rnd.randrange(n)
            if adj[v]:
                start = v
                break
        if start is None:
            for v in range(n):
                if adj[v]:
                    start = v
                    break
            if start is None:
                break

        # random DFS to find a cycle through start
        stack = [(start, iter(sorted(adj[start])))]
        on_path = {start}
        parent = {}
        found_cycle = None
        steps = 0
        while stack and found_cycle is None:
            steps += 1
            if steps > 500000:
                break
            node, it = stack[-1]
            advanced = False
            for nb in it:
                if nb in on_path:
                    if nb != node:
                        # cycle found: walk back from node to nb
                        cyc = [node]
                        cur = node
                        while cur != nb:
                            cur = parent[cur]
                            cyc.append(cur)
                        found_cycle = cyc
                        break
                else:
                    on_path.add(nb)
                    parent[nb] = node
                    stack.append((nb, iter(sorted(adj[nb]))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                on_path.discard(node)

        if found_cycle and len(found_cycle) >= 3:
            cyc = found_cycle
            # remove cycle edges
            m = len(cyc)
            for i in range(m):
                a, b = cyc[i], cyc[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cyc)
            progress = 0
        else:
            # no cycle: emit the incident edge(s) of start as single edges
            nbrs = list(adj[start])
            if not nbrs:
                break
            # try to find any edge and emit it
            if len(nbrs) == 1:
                b = nbrs[0]
                adj[start].discard(b)
                adj[b].discard(start)
                remaining -= 1
                pieces.append([start, b])
            else:
                # emit one edge, prefer one leading to a leaf
                b = nbrs[0]
                for cand in nbrs:
                    if len(adj[cand]) == 1:
                        b = cand
                        break
                adj[start].discard(b)
                adj[b].discard(start)
                remaining -= 1
                pieces.append([start, b])
            progress += 1
            if progress > 3000:
                # safety: stop spinning on non-cycles
                break

    # any leftover edges (shouldn't happen) emitted as single edges
    for u in range(n):
        for v in list(adj[u]):
            if u < v:
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])

    return pieces


def _solve_one(edges, n, deadline):
    """Solve one graph; restart with different seeds until deadline."""
    best = None
    best_count = None
    base_seed = 1000
    attempt = 0
    while True:
        rnd = random.Random(base_seed + attempt)
        pieces = _decompose(edges, n, deadline, rnd)
        cnt = len(pieces)
        if best is None or cnt < best_count:
            best = pieces
            best_count = cnt
        attempt += 1
        if time.time() > deadline:
            break
        # stop early if we reached a very good value
        if best_count is not None and best_count <= n // 2:
            break
        if attempt > 400:
            break
    return best


def construct():
    start_time = time.time()
    total_budget = 150.0
    # leave a safety margin for returning + any overhead
    per_graph_budget = (total_budget - 8.0) / 3.0

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        # cap the deadline at the global limit
        if graph_deadline > start_time + total_budget - 4.0:
            graph_deadline = start_time + total_budget - 4.0
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        # if we're running out of time, fall back to all-edges for the rest
        if time.time() > start_time + total_budget - 6.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
    return results
