import time
from collections import defaultdict

def instance(n, p, seed):
    import random
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose(edges):
    if not edges:
        return []
    
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    paths = []
    
    while remaining:
        # Pick edge incident to vertex with highest remaining degree
        # This helps create longer paths
        best_edge = None
        best_degree_sum = -1
        for e in remaining:
            u, v = e
            deg_sum = len(adj[u]) + len(adj[v])
            if deg_sum > best_degree_sum:
                best_degree_sum = deg_sum
                best_edge = e
        
        u, v = best_edge
        remaining.remove((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend from v
        curr = v
        while True:
            next_v = None
            for w in adj[curr]:
                if w not in path_set:
                    next_v = w
                    break
            if next_v is None:
                break
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.append(next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        # Extend from u
        curr = u
        while True:
            next_v = None
            for w in adj[curr]:
                if w not in path_set:
                    next_v = w
                    break
            if next_v is None:
                break
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.insert(0, next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        paths.append(path)
        
    return paths

def construct():
    deadline = time.time() + 140
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline:
             pass
             
    return results
