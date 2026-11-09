"""Decompose each fixed instance graph into edge-disjoint cycles + single edges."""
import time
from random import Random


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges):
    # Build adjacency (symmetric) with edge tracking
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    # Edge availability set
    avail = set(tuple(sorted(e)) for e in edges)
    cycles = []
    singletons = []

    # Keep removing edge-disjoint cycles while any exists
    while True:
        # Find a cycle via randomized DFS from a few starting vertices
        found = None
        start_order = sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True)
        # Try a handful of random starts + greedy high-degree starts
        starts = start_order[:6]
        rnd = Random(12345)
        extra = rnd.sample(list(adj.keys()), min(6, len(adj)))
        starts = starts + [s for s in extra if s not in starts]

        for s in starts:
            if s not in adj or not adj[s]:
                continue
            # DFS to find a cycle back to s using only available edges
            parent = {}
            stack = [(s, iter(sorted(adj[s], key=lambda x: -len(adj[x]))))]
            visited = {s}
            found = None
            while stack:
                node, it = stack[-1]
                advanced = False
                for nb in it:
                    if (min(node, nb), max(node, nb)) not in avail:
                        continue
                    if nb == s and node != s:
                        # cycle found: path s..node + edge node-s
                        # reconstruct path from parent
                        path = [node]
                        cur = node
                        while cur != s:
                            cur = parent[cur]
                            path.append(cur)
                        path.reverse()  # s ... node
                        found = path
                        break
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = node
                        stack.append((nb, iter(sorted(adj[nb], key=lambda x: -len(adj[x])))))
                        advanced = True
                        break
                if found:
                    break
                if not advanced:
                    stack.pop()
            if found:
                break

        if not found:
            break

        # Remove the cycle's edges from avail, record cycle
        cyc_edges = []
        for i in range(len(found)):
            a, b = found[i], found[(i + 1) % len(found)]
            e = (min(a, b), max(a, b))
            cyc_edges.append(e)
            avail.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        cycles.append(found)

    # Remaining edges are singletons
    for e in avail:
        singletons.append(list(e))

    return cycles + singletons


def construct():
    deadline = time.time() + 140.0
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # safety: fall back to all-singletons for remaining
            results.append([list(e) for e in _instance(*spec)])
            continue
        edges = _instance(*spec)
        results.append(_decompose(edges))
    return results
