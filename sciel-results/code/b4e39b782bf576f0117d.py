"""Rewritten program: maximum clique via Bron-Kerbosch with pivot + degeneracy ordering."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degeneracy_order(adj):
    n = P['n']
    remaining = list(range(n))
    order = []
    while remaining:
        # Pick vertex with minimum degree in remaining
        min_deg = min(len(adj[v] & set(remaining)) for v in remaining)
        # Find all with that degree, pick first
        v = next(v for v in remaining if len(adj[v] & set(remaining)) == min_deg)
        order.append(v)
        remaining.remove(v)
    return order

def _bron_kerbosch(adj, R, P_set, X, best_clique, deadline):
    """Bron-Kerbosch with pivot. Returns best clique found."""
    if time.time() > deadline:
        return best_clique
    if not P_set and not X:
        if len(R) > len(best_clique):
            best_clique = list(R)
        return best_clique
    if len(R) + len(P_set) <= len(best_clique):
        return best_clique
    
    # Choose pivot: vertex in P_set ∪ X with max neighbors in P_set
    PX = P_set | X
    if PX:
        pivot = max(PX, key=lambda v: len(adj[v] & P_set))
        candidates = P_set - adj[pivot]
    else:
        candidates = P_set
    
    for v in list(candidates):
        if time.time() > deadline:
            return best_clique
        new_R = R | {v}
        new_P = P_set & adj[v]
        new_X = X & adj[v]
        best_clique = _bron_kerbosch(adj, new_R, new_P, new_X, best_clique, deadline)
        P_set.discard(v)
        X.add(v)
    return best_clique

def _max_clique(adj):
    n = P['n']
    # Use degeneracy ordering for speed
    order = _degeneracy_order(adj)
    deadline = time.time() + 50  # safety margin
    
    best = []
    # Process in degeneracy order
    for i, v in enumerate(order):
        if time.time() > deadline:
            break
        # P = neighbors of v that come after v in order
        P_set = set()
        for w in adj[v]:
            if w in order[i+1:]:
                P_set.add(w)
        if not P_set:
            continue
        # X = neighbors of v that come before v in order
        X = set()
        for w in adj[v]:
            if w in order[:i]:
                X.add(w)
        best = _bron_kerbosch(adj, {v}, P_set, X, best, deadline)
    
    if not best:
        # Fallback: just try all vertices
        best = _bron_kerbosch(adj, set(), set(range(n)), set(), best, deadline)
    
    return best

def construct():
    adj = _build_adj()
    clique = _max_clique(adj)
    
    # Ensure we have at least 2 branches
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    
    clique = sorted(clique)
    branches = list(clique)
    
    # Build paths: for each pair of branches, find a simple path
    # Since they form a clique, each pair is directly adjacent, so path is just [u, v]
    paths = []
    for u, v in combinations(clique, 2):
        # Verify they are adjacent (they should be since it's a clique)
        if v in adj[u]:
            paths.append([u, v])
        else:
            # Shouldn't happen for a clique, but find a path via BFS
            from collections import deque
            visited = {u}
            queue = deque([(u, [u])])
            found = None
            while queue:
                node, path = queue.popleft()
                if node == v:
                    found = path
                    break
                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, path + [nb]))
            if found:
                paths.append(found)
            else:
                paths.append([u, v])  # fallback
    
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
