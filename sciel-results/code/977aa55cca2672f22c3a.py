import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle_shortest(adj, start, n, max_len=10):
    """Find a short cycle starting from 'start' using BFS/DFS hybrid."""
    # Use BFS to find the shortest cycle involving 'start'
    # But we need a cycle that is a simple cycle.
    # Standard BFS finds shortest path, not necessarily a cycle.
    # We can use DFS with a depth limit.
    
    visited = [False] * n
    parent = [-1] * n
    depth = [0] * n
    
    def dfs(u, d):
        if d > max_len:
            return None
        for v in adj[u]:
            if v == start and d >= 2:
                # Found a cycle
                cyc = [v]
                cur = u
                while cur != start:
                    cyc.append(cur)
                    cur = parent[cur]
                cyc.append(start)
                # Remove duplicate start at end if present? 
                # The cycle is start -> ... -> u -> start
                # The list is [start, ..., u, start]
                # We want unique vertices.
                return cyc
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = d + 1
                res = dfs(v, d + 1)
                if res:
                    return res
                visited[v] = False
        return None

    visited[start] = True
    depth[start] = 0
    return dfs(start, 0)

def _decompose(edges, n, deadline, rnd):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Find a vertex with highest degree
        start = -1
        max_deg = 0
        for v in range(n):
            if len(adj[v]) > max_deg:
                max_deg = len(adj[v])
                start = v
        if start == -1:
            break
            
        # Try to find a short cycle
        cyc = _find_cycle_shortest(adj, start, n, max_len=8)
        
        if cyc and len(cyc) >= 3:
            # Remove cycle edges
            m = len(cyc) - 1 # Last is start again
            # The cycle is [start, v1, v2, ..., start]
            # Edges are (start, v1), (v1, v2), ..., (v_last, start)
            for i in range(m):
                a = cyc[i]
                b = cyc[i+1]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            # Store cycle as list of vertices without the closing duplicate
            pieces.append(cyc[:-1])
        else:
            # No cycle found, remove an edge
            if not adj[start]:
                continue
            # Pick a neighbor, preferably one with low degree
            nbrs = list(adj[start])
            nbrs.sort(key=lambda x: len(adj[x]))
            b = nbrs[0]
            adj[start].discard(b)
            adj[b].discard(start)
            remaining -= 1
            pieces.append([start, b])

    # Leftover edges
    for u in range(n):
        for v in list(adj[u]):
            if u < v:
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])

    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    base_seed = 1000
    attempt = 0
    while time.time() < deadline:
        rnd = random.Random(base_seed + attempt)
        pieces = _decompose(edges, n, deadline, rnd)
        cnt = len(pieces)
        if cnt < best_count:
            best = pieces
            best_count = cnt
        attempt += 1
        if best_count <= n // 2:
            break
        if attempt > 1000:
            break
    return best

def construct():
    start_time = time.time()
    total_budget = 140.0
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
