import time
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle(adj, n):
    """Find a cycle in the graph using DFS. Returns list of vertices or None."""
    color = [0] * n  # 0=white, 1=gray, 2=black
    parent = [-1] * n
    found = [None]
    
    def dfs(u):
        if found[0] is not None:
            return
        color[u] = 1
        for v in adj[u]:
            if found[0] is not None:
                return
            if color[v] == 1 and v != parent[u]:
                # Found a back edge u->v, extract cycle
                cycle = [u]
                cur = u
                while cur != v:
                    cur = parent[cur]
                    cycle.append(cur)
                found[0] = cycle
                return
            elif color[v] == 0:
                parent[v] = u
                dfs(v)
        color[u] = 2
    
    for i in range(n):
        if color[i] == 0:
            dfs(i)
            if found[0] is not None:
                break
    return found[0]

def _decompose(edges, n):
    """Decompose edges into cycles and single edges, minimizing pieces."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for e in edges:
        u, v = e[0], e[1]
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    pieces = []
    remaining = set(edge_set)
    
    # Repeatedly find and remove cycles
    while True:
        # Rebuild adjacency from remaining
        adj = [set() for _ in range(n)]
        for (u, v) in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle = _find_cycle(adj, n)
        if cycle is None:
            break
        
        # Remove cycle edges from remaining
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = (min(u, v), max(u, v))
            remaining.discard(edge)
        
        pieces.append(cycle)
    
    # Add remaining edges as individual pieces
    for (u, v) in sorted(remaining):
        pieces.append([u, v])
    
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    results = []
    for (n, p, seed) in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Fallback: all edges as individual pieces
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n)
        results.append(pieces)
    
    return results
