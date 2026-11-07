"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set(frozenset(e) for e in edges)
    total_edges = len(edge_set)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Generate candidate bicliques
    # We'll find maximal bicliques by trying all pairs of vertices as a seed
    # and expanding to maximal biclique containing that pair
    candidates = []
    
    # Also add trivial bicliques (single edges) as fallback
    for u, v in edges:
        candidates.append(([u], [v]))
    
    # Try to find larger bicliques
    # For each pair of edges sharing a vertex, try to expand
    # More systematically: for each subset of vertices of size 2-4, find common neighbors
    
    # For each pair of vertices (u, v) that are NOT connected (to avoid within-side issues)
    # Actually, let's think differently. A biclique [L, R] requires all cross edges present.
    # So L and R must be such that every vertex in L is connected to every vertex in R.
    
    # Strategy: for each vertex u, consider its neighborhood N(u).
    # Any subset of N(u) can form R with {u} in L, but we need all vertices in L to connect to all in R.
    
    # Let's enumerate small bicliques:
    # Size 1x1: all edges (already added)
    # Size 2x1: for each vertex v, and each pair of neighbors (u1, u2) of v, if u1-u2 not required (they're on same side L, so no constraint), then [u1,u2] x [v] is a biclique if u1-v and u2-v exist.
    # Size 1x2: symmetric
    # Size 2x2: for each pair (u1,u2) and (v1,v2), check all 4 cross edges
    
    # Let's be more systematic and efficient:
    # For each vertex v, let N(v) be its neighbors.
    # For any subset S of N(v), [S, {v}] is a biclique.
    # We want maximal ones, so for each v, the maximal is [N(v), {v}].
    # But we can also have [S, T] where |T| > 1.
    
    # Let's find maximal bicliques using a simple approach:
    # For each pair of non-adjacent vertices (or even adjacent, since same-side edges are allowed),
    # try to build a biclique with them on opposite sides.
    
    # Actually, let's just enumerate all possible bicliques of size up to 3x3 or so,
    # and also the "star" bicliques [N(v), {v}] and [{v}, N(v)].
    
    # Add star bicliques
    for v in range(n):
        if adj[v]:
            L = list(adj[v])
            R = [v]
            candidates.append((L, R))
            L2 = [v]
            R2 = list(adj[v])
            candidates.append((L2, R2))
    
    # Add 2x2 bicliques
    # For each pair of vertices (a, b) that are not connected (they could be on same side too, but let's put them on opposite sides)
    # Actually, for a biclique [L, R], vertices in L don't need to be connected to each other, and same for R.
    # So let's try: for each pair (u, v) where u != v, consider L containing u and R containing v.
    # The maximal such biclique has L = {u} ∪ (common neighbors of u and v that are not v) ... no.
    # L must be subset of N(v) ∪ {u} if u not in N(v), or subset of N(v) if u in N(v).
    # Actually, L ⊆ N(v) and R ⊆ N(u), and L ∩ R = ∅, and all cross edges present.
    # This is getting complex. Let's just use a greedy set cover with the candidates we have.
    
    # Let's add more candidates: for each vertex v, and each subset of size 2 of N(v)
    # This is O(n * deg^2) which might be too much. Let's just use the star bicliques and single edges.
    
    # Convert candidates to frozensets of edges for set cover
    candidate_edge_sets = []
    for L, R in candidates:
        es = set()
        valid = True
        for u in L:
            for v in R:
                if v not in adj[u]:
                    valid = False
                    break
                es.add(frozenset((u, v)))
            if not valid:
                break
        if valid and es:
            candidate_edge_sets.append((frozenset(es), L, R))
    
    # Greedy set cover
    uncovered = set(edge_set)
    selected = []
    
    while uncovered:
        best_set = None
        best_ratio = -1
        best_L = None
        best_R = None
        
        for es, L, R in candidate_edge_sets:
            covered = len(es & uncovered)
            if covered == 0:
                continue
            # Ratio: covered edges per biclique (each biclique costs 1)
            ratio = covered
            if ratio > best_ratio:
                best_ratio = ratio
                best_set = es
                best_L = L
                best_R = R
        
        if best_set is None:
            # Fallback: pick a single edge
            e = next(iter(uncovered))
            u, v = tuple(e)
            selected.append(([u], [v]))
            uncovered.discard(e)
        else:
            selected.append((best_L, best_R))
            uncovered -= best_set
    
    # Convert to required format: list of [left, right] where left and right are lists of ints
    result = []
    for L, R in selected:
        result.append([list(L), list(R)])
    
    return result
# EVOLVE-BLOCK-END
