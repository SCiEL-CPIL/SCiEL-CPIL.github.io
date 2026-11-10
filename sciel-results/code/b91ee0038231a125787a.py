"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_triangle(adj, n):
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k > j and k in adj[i]:
                    return True
    return False

def _min_degree_ok(adj, n, min_deg):
    return all(len(adj[i]) >= min_deg for i in range(n))

def _girth_ok(adj, n, min_girth):
    """Check if all cycles have length >= min_girth using BFS from each vertex."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        for node in queue:
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node] and parent[neighbor] != node:
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _has_cycle(adj, n):
    """Check if graph has at least one cycle."""
    visited = [False] * n
    for start in range(n):
        if not visited[start]:
            stack = [(start, -1)]
            while stack:
                node, par = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                for nb in adj[node]:
                    if nb == par:
                        continue
                    if visited[nb]:
                        return True
                    stack.append((nb, node))
    return False

def _count_distinct_cycle_lengths(adj, n):
    """Count distinct simple cycle lengths using DFS from smallest vertex in cycle."""
    cycle_lengths = set()
    for start in range(n):
        visited = [False] * n
        visited[start] = True
        path = [start]
        
        def dfs(current):
            # Try to close cycle
            if len(path) >= 3 and start in adj[current]:
                cycle_lengths.add(len(path))
            for next_node in adj[current]:
                if next_node == start:
                    continue
                if next_node > start and not visited[next_node]:
                    visited[next_node] = True
                    path.append(next_node)
                    dfs(next_node)
                    path.pop()
                    visited[next_node] = False
        
        dfs(start)
    return len(cycle_lengths) if cycle_lengths else 0

def _evaluate(edges, n=12, min_deg=3, min_girth=4):
    """Returns (valid, num_distinct_cycle_lengths)."""
    edge_set = set()
    for u, v in edges:
        if u == v or u >= n or v >= n:
            return (False, 0)
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return (False, 0)
        edge_set.add(key)
    
    adj = _build_adj([(u, v) for u, v in edge_set], n)
    
    if not _min_degree_ok(adj, n, min_deg):
        return (False, 0)
    if _has_triangle(adj, n):
        return (False, 0)
    if not _girth_ok(adj, n, min_girth):
        return (False, 0)
    if not _has_cycle(adj, n):
        return (False, 0)
    
    ncl = _count_distinct_cycle_lengths(adj, n)
    return (True, ncl)

def construct():
    """
    Construct a graph on vertices 0..11 with:
    - Every vertex has degree at least 3
    - Every simple cycle has length at least 4 (triangle-free)
    - Minimize the number of distinct simple cycle lengths
    
    Strategy: Use two disjoint copies of K_{3,3}.
    K_{3,3} is 3-regular, bipartite (so girth 4, no triangles),
    and contains only cycles of length 4 and 6.
    Two disjoint copies give 12 vertices, all degree 3, girth 4,
    and only cycle lengths {4, 6} -> raw value = 2.
    
    This is believed to be the minimum possible value for this problem size
    and constraints.
    """
    n = 12
    edges = set()
    
    # First copy of K_{3,3}: vertices 0-5
    # Left partition: {0, 1, 2}, Right partition: {3, 4, 5}
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            edges.add(tuple(sorted((u, v))))
    
    # Second copy of K_{3,3}: vertices 6-11
    # Left partition: {6, 7, 8}, Right partition: {9, 10, 11}
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            edges.add(tuple(sorted((u, v))))
    
    # Convert to list of [u, v] lists
    result = [list(e) for e in sorted(edges)]
    
    # Verification step (optional but good for safety)
    # valid, ncl = _evaluate(result)
    # if not valid:
    #     # Fallback to a known good graph if verification fails (should not happen)
    #     pass
    
    return result

# EVOLVE-BLOCK-END
