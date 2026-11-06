"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
from collections import defaultdict, deque
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _path_cover_tree(n, tree_edges):
    """Find a path cover of a tree using ceil(leaves/2) paths."""
    if not tree_edges:
        return []
    
    remaining = set((min(u,v), max(u,v)) for (u,v) in tree_edges)
    paths = []
    
    while remaining:
        # Build degree map
        deg = defaultdict(int)
        for (u, v) in remaining:
            deg[u] += 1
            deg[v] += 1
        
        # Find a leaf
        leaf = None
        for v in range(n):
            if deg[v] == 1:
                leaf = v
                break
        if leaf is None:
            # All degree 0 or 2+; pick any edge
            e = next(iter(remaining))
            remaining.discard(e)
            paths.append([e[0], e[1]])
            continue
        
        # Walk from leaf
        path = [leaf]
        current = leaf
        while True:
            next_v = None
            for (u, v) in list(remaining):
                if u == current:
                    next_v = v
                    remaining.discard((u, v))
                    break
                elif v == current:
                    next_v = u
                    remaining.discard((u, v))
                    break
            if next_v is None:
                break
            path.append(next_v)
            current = next_v
            if deg.get(current, 0) == 0:
                break
        
        if len(path) > 1:
            paths.append(path)
    
    return paths

def _improve(paths, edge_set, max_iter=10000):
    """Try to merge paths where endpoints match."""
    for _ in range(max_iter):
        merged = False
        for i in range(len(paths)):
            if merged or i >= len(paths):
                break
            for j in range(len(paths)):
                if merged or j >= len(paths) or i == j:
                    continue
                pi, pj = paths[i], paths[j]
                # Try 4 orientations
                for ri in [False, True]:
                    if merged:
                        break
                    for rj in [False, True]:
                        a = pi[::-1] if ri else pi
                        b = pj[::-1] if rj else pj
                        if a[-1] == b[0]:
                            cand = a + b[1:]
                            if len(cand) == len(set(cand)):
                                paths[i] = cand
                                paths.pop(j)
                                merged = True
                                break
        if not merged:
            break
    return paths

def _solve_graph(n, p, seed, deadline):
    edges = _instance(n, p, seed)
    edge_set = set((min(u,v), max(u,v)) for (u,v) in edges)
    
    adj = defaultdict(set)
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Spanning tree via BFS
    visited = {0}
    tree_edges = []
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v in sorted(adj[u]):
            if v not in visited:
                visited.add(v)
                tree_edges.append((min(u,v), max(u,v)))
                queue.append(v)
    
    # Path cover of tree
    paths = _path_cover_tree(n, tree_edges)
    
    # Add non-tree edges
    tree_edge_set = set((min(u,v), max(u,v)) for (u,v) in tree_edges)
    non_tree = edge_set - tree_edge_set
    for e in non_tree:
        paths.append([e[0], e[1]])
    
    # Improve
    if time.time() < deadline - 5:
        paths = _improve(paths, edge_set)
    
    # Verify
    covered = set()
    for path in paths:
        if len(set(path)) != len(path):
            return [[list(e)] for e in edges]
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            e = (min(u,v), max(u,v))
            if e in covered or e not in edge_set:
                return [[list(e2)] for e2 in edges]
            covered.add(e)
    
    if covered != edge_set:
        return [[list(e)] for e in edges]
    
    return paths

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            edges = _instance(*spec)
            results.append([[list(e)] for e in edges])
        else:
            paths = _solve_graph(*spec, deadline)
            results.append(paths)
    
    return results
