"""Rewritten program for C6-free graph on 40 vertices using spectral heuristic."""
import numpy as np
import random
import sys

def _has_c6(adj):
    """Check if adjacency matrix (0/1) contains a 6-cycle.
    A 6-cycle exists if there are vertices v1,...,v6 (distinct) with edges
    v1-v2, v2-v3, ..., v6-v1.
    We can check by looking for paths of length 5 that close into a cycle,
    but a simpler necessary/sufficient check for small graphs:
    Count the number of 6-cycles. If > 0, it's not C6-free.
    
    For efficiency on 40 vertices, we can use the trace of A^6 / 6 to count
    6-cycles (each 6-cycle is counted 12 times: 6 starting points * 2 directions).
    Actually, trace(A^6) counts closed walks of length 6. Each 6-cycle contributes
    12 to trace(A^6) (6 vertices * 2 directions). But closed walks of length 6 can
    also include shorter cycles traversed multiple times. However, for a simple graph,
    the number of 6-cycles can be extracted.
    
    A more direct approach: enumerate all 6-tuples of distinct vertices and check
    if they form a cycle. But that's 40*39*38*37*36*35 which is too large.
    
    Better: Use the fact that a 6-cycle is a set of 6 vertices {v1,...,v6} with
    edges v1v2, v2v3, v3v4, v4v5, v5v6, v6v1. We can iterate over all subsets
    of 6 vertices (C(40,6) = 3,838,380) and check if any subset induces a C6.
    That's feasible but might be slow in pure Python.
    
    Alternatively, we can check for C6 by looking at the adjacency matrix powers.
    The (i,j) entry of A^5 gives the number of walks of length 5 from i to j.
    A 6-cycle through vertices i and j (i != j) exists if there's a walk of length 5
    from i to j that uses 6 distinct vertices. But this is tricky.
    
    For a 40-vertex graph, let's use a randomized check or a more efficient method.
    Actually, for the purpose of this heuristic, we can use a probabilistic check:
    sample many random 6-tuples and see if they form a C6. But that's not exact.
    
    Let me use a better approach: The number of 6-cycles in a graph can be computed
    as (1/12) * (trace(A^6) - 6*trace(A^2*A^2) + ... ) but this is complex.
    
    For practical purposes, since we're doing a heuristic, let's use a simple check:
    For each pair of non-adjacent vertices (u,v), check if there's a path of length 5
    between them using distinct intermediate vertices. If so, adding edge (u,v) would
    create a C6. But we need to check if the graph already has a C6.
    
    Let's use a different approach: iterate over all 6-subsets and check if they
    form a cycle. C(40,6) = 3,838,380. For each subset, check if the induced
    subgraph is a C6. This might be too slow in Python.
    
    Instead, let's use the following: a graph is C6-free if and only if it has no
    6-cycle. We can check this by, for each vertex, looking at its neighborhood and
    checking for paths of length 4 in the complement... this is getting complex.
    
    For the heuristic, I'll use a simpler verification: check if trace(A^6) is
    consistent with no 6-cycles. Actually, let me just use a brute-force check on
    the final graph since it's only 40 vertices.
    """
    n = adj.shape[0]
    # Check all 6-cycles by iterating over combinations of 6 vertices
    from itertools import combinations
    for combo in combinations(range(n), 6):
        # Check if these 6 vertices form a C6
        # Try all 5! = 120 permutations (since fixing one vertex and direction)
        # Actually, for a set of 6 vertices to form a C6, they must have exactly
        # 6 edges forming a cycle. Let's check if the induced subgraph is a C6.
        sub_adj = adj[list(combo)][:, list(combo)]
        # A C6 has exactly 6 edges
        num_edges = sum(sub_adj[i,j] for i in range(6) for j in range(i+1,6))
        if num_edges != 6:
            continue
        # Check if it's connected and each vertex has degree 2
        # For a C6, each vertex in the induced subgraph has degree 2
        for i in range(6):
            deg = sum(sub_adj[i,j] for j in range(6) if j != i)
            if deg != 2:
                break
        else:
            # All degrees are 2, and 6 edges, so it's a C6 (or two C3s, but C3 has 3 edges)
            # With 6 edges and all degrees 2, it must be a single C6 or two C3s.
            # Two C3s would have 6 edges and each vertex degree 2, but the graph would be disconnected.
            # Check if connected: do a BFS
            visited = [False] * 6
            visited[0] = True
            queue = [0]
            count = 1
            while queue:
                u = queue.pop(0)
                for v in range(6):
                    if sub_adj[u,v] and not visited[v]:
                        visited[v] = True
                        queue.append(v)
                        count += 1
            if count == 6:
                return True
    return False

def construct():
    """Construct a C6-free graph on 40 vertices maximizing edges."""
    n = 40
    target_degree = 5.25
    target_edges = int(n * target_degree / 2)  # ~105
    
    random.seed(42)
    np.random.seed(42)
    
    # Initialize with a random regular-ish graph
    # Start with the seed: 8 copies of K5
    adj = np.zeros((n, n), dtype=np.int8)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u, v] = 1
                adj[v, u] = 1
    
    # Try to add edges to increase the count while maintaining C6-freeness
    # We'll use a greedy approach: try to add edges one by one, checking for C6
    # But first, let's try a spectral-inspired approach
    
    # Compute initial spectral norm
    eigenvalues = np.linalg.eigvalsh(adj)
    max_eig = eigenvalues[-1]
    
    # Try to add edges in order of "spectral benefit"
    # For each non-edge (u,v), compute how much adding it would increase the
    # spectral norm. We want to add edges that increase edge count but keep
    # spectral norm low.
    
    # Actually, let's just do a greedy edge addition with C6 checking
    # This is more reliable for correctness
    
    # Get all non-edges
    non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 0:
                non_edges.append((u, v))
    
    random.shuffle(non_edges)
    
    # Try to add each non-edge if it doesn't create a C6
    added = 0
    for (u, v) in non_edges:
        # Tentatively add edge
        adj[u, v] = 1
        adj[v, u] = 1
        
        # Check for C6
        if _has_c6(adj):
            # Remove edge
            adj[u, v] = 0
            adj[v, u] = 0
        else:
            added += 1
    
    # Count edges
    num_edges = sum(adj[u, v] for u in range(n) for v in range(u + 1, n))
    
    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 1:
                edges.append([u, v])
    
    return edges
