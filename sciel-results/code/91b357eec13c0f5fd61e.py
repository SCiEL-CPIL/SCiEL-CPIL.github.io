import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_fast(edges, n, deadline, rng):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Try to find a cycle
        cycle_found = None
        
        # Pick a random start vertex with degree > 0
        candidates = [i for i in range(n) if adj[i]]
        if not candidates:
            break
            
        # Try a few random starts to find a cycle
        for _ in range(min(10, len(candidates))):
            start = rng.choice(candidates)
            if not adj[start]:
                continue
                
            visited = {start}
            path = [start]
            
            # Use recursion with a helper that takes rng
            def dfs(u):
                if time.time() > deadline:
                    return None
                if u == start and len(path) >= 3:
                    return path[:]
                
                neighbors = list(adj[u])
                rng.shuffle(neighbors)
                
                for v in neighbors:
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        res = dfs(v)
                        if res:
                            return res
                        path.pop()
                        visited.remove(v)
                return None
            
            res = dfs(start)
            if res:
                cycle_found = res
                break
        
        if cycle_found:
            m = len(cycle_found)
            for i in range(m):
                a = cycle_found[i]
                b = cycle_found[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cycle_found)
        else:
            # No cycle found in this attempt, take a single edge
            edge_found = False
            for u in range(n):
                if adj[u]:
                    v = next(iter(adj[u]))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    remaining -= 1
                    pieces.append([u, v])
                    edge_found = True
                    break
            if not edge_found:
                break

    # Add any remaining edges
    if remaining > 0:
        for u in range(n):
            for v in list(adj[u]):
                if u < v:
                    pieces.append([u, v])

    return pieces


def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    # Try a few random seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        pieces = _decompose_fast(edges, n, deadline, rng)
        cnt = len(pieces)
        if cnt < best_count:
            best_count = cnt
            best = pieces
            
    if best is None:
        # Fallback: all edges
        best = [[u, v] for u, v in edges]
        
    return best


def construct():
    start_time = time.time()
    total_budget = 140.0  # Safety margin
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results
