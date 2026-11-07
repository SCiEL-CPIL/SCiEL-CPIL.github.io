"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bipartite_edges(n):
    """Return all possible cross-parity edges not in base path."""
    base = set()
    for i in range(n - 1):
        base.add((i, i + 1))
        base.add((i + 1, i))
    evens = [i for i in range(n) if i % 2 == 0]
    odds = [i for i in range(n) if i % 2 == 1]
    possible = []
    for e in evens:
        for o in odds:
            if (e, o) not in base and (o, e) not in base:
                possible.append((e, o))
    return possible

def _build_adj(n, added_edges):
    """Build adjacency list from base path + added edges."""
    adj = {i: set() for i in range(n)}
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _check_diameter(n, added_edges, max_diam=4):
    """Check if graph has diameter <= max_diam and is triangle-free."""
    adj = _build_adj(n, added_edges)
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in adj[u]:
                    return False
    # Check diameter via BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_diam:
                        return False
                    q.append(v)
        if any(d == -1 for d in dist):
            return False  # disconnected
    return True

def _greedy_min_edges(n, possible_edges):
    """Greedy + local search to find minimum edge set."""
    # Start with empty set, greedily add edges that fix most violations
    # But first, let's try small sets
    
    # Try k=4,5,6,7,8,9,10...
    # For small k, use combinations
    # For larger k, use heuristic
    
    # First, let's try to find a solution with few edges
    # Use a heuristic: start from the known 10-edge solution, try to remove edges
    
    # Known solution: hub 11 to evens {0,2,4,6,8,14,16,18,20,22}
    initial = [(11, 0), (11, 2), (11, 4), (11, 6), (11, 8), (11, 14), (11, 16), (11, 18), (11, 20), (11, 22)]
    
    if not _check_diameter(n, initial):
        return initial  # fallback
    
    current = list(initial)
    
    # Try to remove edges one by one
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            trial = current[:i] + current[i+1:]
            if _check_diameter(n, trial):
                current = trial
                improved = True
                break
    
    # Try to swap edges: remove one, add another
    for i in range(len(current)):
        for new_edge in possible_edges:
            if new_edge in current:
                continue
            trial = current[:i] + current[i+1:] + [new_edge]
            if _check_diameter(n, trial):
                current = trial
                break
    
    return current

def construct():
    n = P['n']
    possible = _bipartite_edges(n)
    result = _greedy_min_edges(n, possible)
    # Convert to list of lists with Python ints
    return [[int(u), int(v)] for u, v in result]
# EVOLVE-BLOCK-END
