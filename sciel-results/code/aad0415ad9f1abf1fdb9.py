"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def _q3_free_check(adj, n):
    """Check if graph with adjacency matrix adj (n x n boolean) is Q3-free."""
    # A Q3 has 8 vertices and 12 edges.
    # We check all 8-subsets of vertices to see if any induce a subgraph isomorphic to Q3.
    # For efficiency, we can use the fact that Q3 is 3-regular on 8 vertices with girth 4.
    # Actually, let's just check all 8-subsets. There are C(16,8) = 12870 subsets.
    # For each, check if the induced subgraph is isomorphic to Q3.
    
    # Q3 properties: 8 vertices, 12 edges, 3-regular, bipartite, diameter 3.
    # Let's precompute the Q3 adjacency matrix.
    # Vertices of Q3 are 3-bit strings. Edge between x,y if Hamming distance 1.
    q3_adj = np.zeros((8, 8), dtype=bool)
    for i in range(8):
        for j in range(8):
            if i != j:
                diff = bin(i ^ j).count('1')
                if diff == 1:
                    q3_adj[i, j] = True
                    q3_adj[j, i] = True
    
    # To check isomorphism of an 8-vertex induced subgraph to Q3:
    # 1. Must have exactly 12 edges
    # 2. Must be 3-regular
    # 3. Must be bipartite (which Q3 is)
    # 4. Girth 4 (no triangles)
    # Actually, a 3-regular bipartite graph on 8 vertices with 12 edges is Q3? 
    # The only 3-regular bipartite graph on 8 vertices is Q3 (since it's the unique 3-regular bipartite graph on 4+4 vertices... wait, K_{4,4} is 4-regular).
    # Let's just do a direct isomorphism check by comparing sorted degree sequences and edge counts, then verify structure.
    # Simpler: since 8 vertices is small, we can try all 8! = 40320 permutations? That's too slow for 12870 subsets.
    # Better: use the canonical form or just check if the subgraph matches Q3 structure.
    
    # Alternative: Q3 is the unique 3-regular bipartite graph on 8 vertices with no 4-cycles? No, Q3 has 4-cycles.
    # Let's use: a graph is Q3 iff it is 3-regular, bipartite, and has exactly 12 edges.
    # Is that sufficient? The 3-regular bipartite graphs on 8 vertices: the bipartition must be 4+4.
    # The only 3-regular bipartite graph with parts of size 4 is... well, there are multiple.
    # K_{4,4} minus a perfect matching is 3-regular bipartite on 8 vertices. Is that Q3? 
    # Q3 has girth 4. K_{4,4} minus perfect matching also has girth 4.
    # Are they isomorphic? Yes, Q3 is isomorphic to K_{4,4} minus a perfect matching.
    # So checking 3-regular + bipartite + 12 edges should suffice for 8-vertex graphs.
    
    vertices = range(n)
    for subset in combinations(vertices, 8):
        # Build induced subgraph
        sub_adj = np.zeros((8, 8), dtype=bool)
        for i, u in enumerate(subset):
            for j, v in enumerate(subset):
                if i != j:
                    sub_adj[i, j] = adj[u, v]
        
        # Check 3-regular
        degs = sub_adj.sum(axis=1)
        if not np.all(degs == 3):
            continue
        
        # Check bipartite
        # BFS coloring
        color = [-1] * 8
        color[0] = 0
        queue = [0]
        is_bipartite = True
        while queue:
            u = queue.pop(0)
            for v in range(8):
                if sub_adj[u, v]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
            if not is_bipartite:
                break
        
        if is_bipartite:
            return False  # Found a Q3
    
    return True

def construct():
    n = 16
    # Start with K_{16} and remove edges to make it Q3-free.
    # Greedy approach: start with complete graph, then iteratively remove edges that create Q3.
    # Or: start with K_{3,13} (39 edges) and try to add edges.
    
    # Let's try a smarter approach: use a known construction.
    # The seed uses K_{3,13} with 39 edges.
    # Can we do better?
    
    # Let's try to build a graph with more edges.
    # One idea: take the complete graph K_16 and remove a minimum edge set to destroy all Q3s.
    # This is a hitting set problem.
    
    # Alternative: use a random restart approach with local search.
    # Start with a random graph, then try to add edges if they don't create a Q3.
    
    import random
    random.seed(42)
    
    best_edges = None
    best_count = 0
    
    for attempt in range(200):
        # Start with K_{3,13} or a random dense graph
        if attempt < 100:
            # Start with K_{3,13}
            adj = np.zeros((n, n), dtype=bool)
            for a in range(3):
                for b in range(3, n):
                    adj[a, b] = True
                    adj[b, a] = True
        else:
            # Start with a random bipartite graph with parts of size 8
            adj = np.zeros((n, n), dtype=bool)
            # Random bipartition
            part1 = list(range(n))
            random.shuffle(part1)
            p1 = set(part1[:8])
            p2 = set(part1[8:])
            # Add random edges between p1 and p2 with high probability
            for u in p1:
                for v in p2:
                    if random.random() < 0.8:
                        adj[u, v] = True
                        adj[v, u] = True
        
        # Try to add edges greedily
        added = True
        while added:
            added = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u, v]:
                        # Try adding edge (u, v)
                        adj[u, v] = True
                        adj[v, u] = True
                        if _q3_free_check(adj, n):
                            added = True
                        else:
                            adj[u, v] = False
                            adj[v, u] = False
        
        edge_count = int(adj.sum() // 2)
        if edge_count > best_count:
            best_count = edge_count
            best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u, v]]
    
    # Try to improve by removing edges and re-adding
    if best_edges is not None:
        adj = np.zeros((n, n), dtype=bool)
        for u, v in best_edges:
            adj[u, v] = True
            adj[v, u] = True
        
        # Try removing one edge and re-optimizing
        improved = True
        while improved:
            improved = False
            for i, (u, v) in enumerate(best_edges[:]):
                # Remove edge (u, v)
                adj[u, v] = False
                adj[v, u] = False
                new_best = best_edges[:]
                new_best.remove([u, v])
                
                # Try to add edges
                added = True
                while added:
                    added = False
                    for a in range(n):
                        for b in range(a + 1, n):
                            if not adj[a, b]:
                                adj[a, b] = True
                                adj[b, a] = True
                                if _q3_free_check(adj, n):
                                    added = True
                                    new_best.append([a, b])
                                else:
                                    adj[a, b] = False
                                    adj[b, a] = False
                
                new_count = len(new_best)
                if new_count > len(best_edges):
                    best_edges = new_best
                    best_count = new_count
                    improved = True
                    break
                else:
                    # Restore edge
                    adj[u, v] = True
                    adj[v, u] = True
    
    return best_edges if best_edges else [[a, b] for a in range(3) for b in range(3, 16)]
# EVOLVE-BLOCK-END
