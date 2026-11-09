"""Greedy edge-disjoint path cover with merge improvement."""

import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140

    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def greedy_cover(edges):
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        while remaining:
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            start = next(iter(remaining))[0]
            path = [start]
            visited = {start}
            current = start
            
            while True:
                candidates = [n for n in adj[current] if n not in visited]
                if not candidates:
                    break
                best = min(candidates, key=lambda n: len(adj[n] - visited))
                path.append(best)
                visited.add(best)
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i+1])))
                    remaining.discard(e)
                current = best
            
            paths.append(path)
        
        for e in remaining:
            paths.append(list(e))
        
        return paths

    def try_merge(paths, edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    p1, p2 = paths[i], paths[j]
                    if set(p1) & set(p2):
                        continue
                    # Try p1_end -> p2_start
                    if p1[-1] in adj and p2[0] in adj and p2[0] in adj[p1[-1]]:
                        paths[i] = p1 + p2
                        paths.pop(j)
                        changed = True
                        break
                    # Try p2_end -> p1_start
                    if p2[-1] in adj and p1[0] in adj and p1[0] in adj[p2[-1]]:
                        paths[i] = p2 + p1
                        paths.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        paths = greedy_cover(edges)
        if time.time() < deadline - 5:
            paths = try_merge(paths, edges)
        results.append(paths)
    
    return results

# EVOLVE-BLOCK-END
