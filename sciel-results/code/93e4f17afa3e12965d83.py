# EVOLVE-BLOCK-START
import time
from random import Random

def _build_graph(n, p, seed):
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _find_triangles(adj):
    """Find all triangles in the graph."""
    n = len(adj)
    triangles = []
    for u in range(n):
        for v in sorted(adj[u]):
            if v <= u:
                continue
            common = adj[u] & adj[v]
            for w in sorted(common):
                if w > v:
                    triangles.append((u, v, w))
    return triangles

def _greedy_clique_partition(adj, deadline):
    """Greedy: find triangles, extend to maximal cliques, then cover remaining edges as pairs."""
    n = len(adj)
    # Track remaining edges
    remaining = set()
    for u in range(n):
        for v in adj[u]:
            if v > u:
                remaining.add((u, v))
    
    cliques = []
    
    # Try to find large cliques greedily
    # First pass: find all triangles and try to extend them
    triangles = _find_triangles(adj)
    
    # For each triangle, check if it can be extended; use greedy maximal clique
    # Sort triangles by potential (number of common neighbors)
    triangle_scores = []
    for (u, v, w) in triangles:
        # Check if all three edges are still remaining
        if (u, v) in remaining and (u, w) in remaining and (v, w) in remaining:
            # Count common neighbors that could extend
            common = adj[u] & adj[v] & adj[w]
            score = len(common)
            triangle_scores.append((score, u, v, w))
    
    triangle_scores.sort(reverse=True)
    
    # Greedily pick triangles/cliques
    for score, u, v, w in triangle_scores:
        if time.time() > deadline:
            break
        # Check edges still available
        if (u, v) not in remaining or (u, w) not in remaining or (v, w) not in remaining:
            continue
        # Try to extend to maximal clique starting from {u,v,w}
        clique = {u, v, w}
        candidates = adj[u] & adj[v] & adj[w]
        candidates = candidates - clique
        # Greedily add vertices
        while candidates:
            # Pick vertex with most connections to current clique
            best = None
            best_score = -1
            for c in candidates:
                s = len(clique & adj[c])
                if s > best_score:
                    best_score = s
                    best = c
            if best is None:
                break
            clique.add(best)
            candidates = candidates & adj[best]
            candidates = candidates - clique
        
        # Verify all edges in clique are in remaining
        valid = True
        clique_list = sorted(clique)
        for i in range(len(clique_list)):
            for j in range(i + 1, len(clique_list)):
                a, b = clique_list[i], clique_list[j]
                if (min(a,b), max(a,b)) not in remaining:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(clique) >= 2:
            # Remove all edges in this clique from remaining
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    remaining.discard((min(a,b), max(a,b)))
            cliques.append(clique_list)
    
    # Cover remaining edges as pairs
    for edge in list(remaining):
        if time.time() > deadline:
            break
        u, v = edge
        if (u, v) in remaining:
            cliques.append([u, v])
            remaining.discard((u, v))
    
    return cliques

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s limit
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback: trivial partition
            adj = _build_graph(n, p, seed)
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        cliques.append([u, v])
            results.append(cliques)
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _greedy_clique_partition(adj, deadline)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
