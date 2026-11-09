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
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    
    avail = set(tuple(sorted(e)) for e in edges)
    cycles = []
    
    while True:
        best_cycle = None
        best_len = 0
        
        for s in list(adj.keys()):
            if len(adj[s]) < 2:
                continue
            
            parent = {}
            visited = {s}
            stack = [(s, iter(sorted(adj[s], key=lambda x: -len(adj[x]))))]
            
            while stack:
                node, it = stack[-1]
                advanced = False
                for nb in it:
                    e = (min(node, nb), max(node, nb))
                    if e not in avail:
                        continue
                    if nb == s and node != s:
                        # Cycle found
                        path = [node]
                        cur = node
                        while cur != s:
                            cur = parent[cur]
                            path.append(cur)
                        path.reverse()
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path
                        break
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = node
                        stack.append((nb, iter(sorted(adj[nb], key=lambda x: -len(adj[x])))))
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
        
        if not best_cycle:
            break
        
        for i in range(len(best_cycle)):
            a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
            e = (min(a, b), max(a, b))
            avail.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        
        cycles.append(best_cycle)
        
    singletons = [list(e) for e in avail]
    return cycles + singletons


def construct():
    deadline = time.time() + 100.0
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # safety: fall back to all-singletons for remaining
            results.append([list(e) for e in _instance(*spec)])
            continue
        edges = _instance(*spec)
        results.append(_decompose(edges))
    return results
