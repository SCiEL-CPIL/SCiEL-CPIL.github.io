"""Decompose three fixed random graphs into edge-disjoint cycles and edges, minimizing pieces."""
import time
import random
import sys

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _edge_set(cycle):
    s = set()
    for i in range(len(cycle)):
        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
        s.add((a, b) if a < b else (b, a))
    return s


def _greedy_find_cycles(adj, edge_set, deadline, seed):
    """Greedy: repeatedly find a long simple cycle via DFS, remove its edges."""
    cycles = []
    remaining = set(edge_set)
    adj_rem = {v: set() for v in adj}
    for e in edge_set:
        u, v = e
        adj_rem[u].add(v)
        adj_rem[v].add(u)
    rng = random.Random(seed)
    while remaining:
        if time.time() >= deadline:
            break
        # start vertex: pick a random remaining edge's endpoint
        start_edge = next(iter(remaining))
        start = start_edge[rng.randrange(2)]
        # DFS for a cycle back to start, maximizing length
        best_cycle = None
        # iterative DFS with backtracking limited by a step budget
        step_budget = 40000
        steps = 0
        stack = [(start, [start])]
        visited_in_path = {start}
        while stack and steps < step_budget:
            cur, path = stack.pop()
            steps += 1
            # try to extend
            nbrs = list(adj_rem.get(cur, ()))
            rng.shuffle(nbrs)
            found_extend = False
            for nb in nbrs:
                if nb == start and len(path) >= 3:
                    cyc = list(path)
                    if best_cycle is None or len(cyc) > len(best_cycle):
                        best_cycle = cyc
                    found_extend = True
                    break
                if nb not in visited_in_path:
                    stack.append((nb, path + [nb]))
                    visited_in_path.add(nb)
            if found_extend:
                # we found a cycle; but keep searching a bit more for longer? keep best so far
                pass
        # fallback: if DFS didn't find, do a simpler randomized cycle hunt
        if best_cycle is None:
            for _ in range(60):
                path = [start]
                seen = {start}
                cur = start
                ok = False
                for _ in range(200):
                    nbrs = list(adj_rem.get(cur, ()))
                    rng.shuffle(nbrs)
                    nxt = None
                    for nb in nbrs:
                        if nb == start and len(path) >= 3:
                            nxt = start
                            break
                        if nb not in seen:
                            nxt = nb
                            break
                    if nxt is None:
                        break
                    if nxt == start:
                        ok = True
                        break
                    path.append(nxt)
                    seen.add(nxt)
                    cur = nxt
                if ok and (best_cycle is None or len(path) > len(best_cycle)):
                    best_cycle = path
        if best_cycle is None:
            # cannot find any cycle through start; leave its edges as singletons later
            # move on by not removing anything; to avoid infinite loop, break
            break
        # remove best_cycle edges from remaining
        ec = _edge_set(best_cycle)
        for e in ec:
            if e in remaining:
                remaining.discard(e)
                a, b = e
                adj_rem[a].discard(b)
                adj_rem[b].discard(a)
        cycles.append(list(best_cycle))
    return cycles, remaining


def _local_search(adj, cycles, singletons, deadline):
    """Improve by: (a) merging two cycles sharing nothing but recombining? not valid.
       Instead do 2-opt style: try to extend a cycle by swapping with another cycle's
       structure. Simpler: try to find a longer cycle by combining two existing cycles
       that share a vertex, or by replacing a short cycle with a longer one."""
    # Represent pieces: cycles list and singleton edges set
    # Try improvement: for each pair of cycles, if they share a vertex, attempt to
    # splice into one longer cycle (if edge-disjoint and vertex-mergeable).
    # A valid merge: cycle A = [a0..ak-1], cycle B = [b0..bm-1]. If they share exactly
    # one vertex v, we can make one cycle by walking A from v around to v, but that just
    # repeats. Instead, splicing two cycles at a shared vertex is not a simple cycle.
    # So skip merge. Instead, try to find a longer cycle in the remaining structure by
    # re-running greedy on a subset? Keep it simple: try to upgrade singletons into
    # cycles by searching for a cycle that uses some singleton edges.
    # We'll just attempt: for each singleton edge, see if it can be part of a cycle
    # formed with other singletons; if a cycle of >=3 singletons exists, replace them.
    # Build adjacency of singletons.
    sing_adj = {v: set() for v in adj}
    for e in singletons:
        u, v = e
        sing_adj[u].add(v)
        sing_adj[v].add(u)
    # find cycles in the singleton-only graph
    used = set()
    new_singles = set(singletons)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in list(new_singles):
            if e in used:
                continue
            u, v = e
            # BFS for a cycle containing e in singleton graph
            # find path from v to u avoiding e
            from collections import deque
            parent = {v: None}
            q = deque([v])
            found = None
            while q:
                x = q.popleft()
                if x == u:
                    found = u
                    break
                for nb in sing_adj[x]:
                    if nb == x:
                        continue
                    if (min(x, nb), max(x, nb)) in used:
                        continue
                    if nb not in parent:
                        parent[nb] = x
                        q.append(nb)
            if found is not None:
                # reconstruct path u<-...<-v, plus edge (u,v) forms cycle
                path = []
                cur = u
                while cur is not None:
                    path.append(cur)
                    if cur == v:
                        break
                    cur = parent[cur]
                # path is u..v; cycle = path (closing edge v-u is e)
                if len(path) >= 3:
                    cyc = list(path)
                    for i in range(len(cyc)):
                        a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                        ee = (a, b) if a < b else (b, a)
                        used.add(ee)
                        new_singles.discard(ee)
                    cycles.append(cyc)
                    improved = True
                    break
    return cycles, new_singles


def _solve_one(n, p, seed, deadline):
    edges = _instance(n, p, seed)
    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    cycles, remaining = _greedy_find_cycles(adj, set(edges), deadline, seed)
    singletons = set(remaining)
    cycles, singletons = _local_search(adj, cycles, singletons, deadline)
    pieces = [list(c) for c in cycles] + [list(e) for e in singletons]
    return pieces


def construct():
    t0 = time.time()
    total_budget = 120.0
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for i, spec in enumerate(specs):
        # reserve margin: ensure at least ~1s left for last
        per = (total_budget - (time.time() - t0)) / (len(specs) - i)
        deadline = time.time() + max(1.0, per - 2.0)
        try:
            pieces = _solve_one(*spec, deadline)
        except Exception:
            # fallback: every edge as its own piece
            pieces = [list(e) for e in _instance(*spec)]
        results.append(pieces)
    return results


if __name__ == "__main__":
    res = construct()
    print([len(r) for r in res])
