"""Rewritten program: deterministic local search starting from Schläfli graph,
attempting to grow a (5,5)-free graph beyond 37 vertices."""
import itertools
import numpy as np

def _is_55_free(adj_matrix, n):
    """Check if graph on n vertices has no K5 and no independent set of size 5."""
    # Check for K5: any 5 vertices all pairwise adjacent
    # Check for independent set of size 5: any 5 vertices with no edges between them
    # Efficient check using bitmasks
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j]:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    # Check for K5
    for combo in itertools.combinations(range(n), 5):
        a, b, c, d, e = combo
        if (adj_bits[a] & (1 << b)) and (adj_bits[a] & (1 << c)) and (adj_bits[a] & (1 << d)) and \
           (adj_bits[a] & (1 << e)) and (adj_bits[b] & (1 << c)) and (adj_bits[b] & (1 << d)) and \
           (adj_bits[b] & (1 << e)) and (adj_bits[c] & (1 << d)) and (adj_bits[c] & (1 << e)) and \
           (adj_bits[d] & (1 << e)):
            return False
    
    # Check for independent set of size 5
    non_adj_bits = [0] * n
    for i in range(n):
        non_adj_bits[i] = ((1 << n) - 1) - (1 << i) - adj_bits[i]
    
    for combo in itertools.combinations(range(n), 5):
        a, b, c, d, e = combo
        if (non_adj_bits[a] & (1 << b)) and (non_adj_bits[a] & (1 << c)) and (non_adj_bits[a] & (1 << d)) and \
           (non_adj_bits[a] & (1 << e)) and (non_adj_bits[b] & (1 << c)) and (non_adj_bits[b] & (1 << d)) and \
           (non_adj_bits[b] & (1 << e)) and (non_adj_bits[c] & (1 << d)) and (non_adj_bits[c] & (1 << e)) and \
           (non_adj_bits[d] & (1 << e)):
            return False
    
    return True

def _schlaefi_graph():
    """Construct the Schläfli graph on 27 vertices.
    Vertices are 2-subsets of {0,1,2,3,4,5}. Two vertices are adjacent if they are disjoint."""
    n = 27
    verts = list(itertools.combinations(range(6), 2))
    vert_index = {v: i for i, v in enumerate(verts)}
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if len(set(verts[i]) & set(verts[j])) == 0:
                adj[i][j] = adj[j][i] = True
    return adj

def _try_add_vertex(adj, n, target_adjacency):
    """Try to add a vertex with the given adjacency pattern to existing n vertices.
    Returns True if the new graph is (5,5)-free."""
    new_n = n + 1
    new_adj = [[False] * new_n for _ in range(new_n)]
    for i in range(n):
        for j in range(i + 1, n):
            new_adj[i][j] = new_adj[j][i] = adj[i][j]
    for i in range(n):
        new_adj[i][new_n - 1] = new_adj[new_n - 1][i] = target_adjacency[i]
    return _is_55_free(new_adj, new_n)

def construct():
    # Start with Schläfli graph on 27 vertices
    adj = _schlaefi_graph()
    n = 27
    
    # Verify initial graph is (5,5)-free
    assert _is_55_free(adj, n), "Schläfli graph should be (5,5)-free"
    
    # Try to add vertices one by one using deterministic heuristics
    # Strategy: try all possible adjacency patterns, preferring those that
    # maintain balance and don't create K5 or independent 5-sets
    
    max_attempts_per_vertex = 5000
    added = 0
    
    while n < 42 and added < 15:  # Don't exceed 42 (known bound), try to add up to 15 more
        success = False
        for attempt in range(max_attempts_per_vertex):
            # Generate a candidate adjacency pattern
            # Use a deterministic pseudo-random pattern based on attempt number
            # Prefer patterns with roughly half edges to existing vertices
            if attempt == 0:
                # Try a vertex connected to all existing vertices
                candidate = [True] * n
            elif attempt == 1:
                # Try a vertex connected to no existing vertices
                candidate = [False] * n
            elif attempt == 2:
                # Try connecting to vertices with high degree
                degrees = [sum(adj[i]) for i in range(n)]
                sorted_verts = sorted(range(n), key=lambda x: -degrees[x])
                candidate = [False] * n
                for i in sorted_verts[:n // 2]:
                    candidate[i] = True
            elif attempt == 3:
                # Try connecting to vertices with low degree
                degrees = [sum(adj[i]) for i in range(n)]
                sorted_verts = sorted(range(n), key=lambda x: degrees[x])
                candidate = [False] * n
                for i in sorted_verts[:n // 2]:
                    candidate[i] = True
            else:
                # Deterministic pseudo-random pattern
                # Use a simple LCG for reproducibility
                seed = attempt * 2654435761 % (2**32)
                candidate = []
                for i in range(n):
                    seed = (seed * 1103515245 + 12345) % (2**31)
                    candidate.append((seed & 0x80000000) != 0)
            
            if _try_add_vertex(adj, n, candidate):
                # Successfully added a vertex
                new_n = n + 1
                new_adj = [[False] * new_n for _ in range(new_n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        new_adj[i][j] = new_adj[j][i] = adj[i][j]
                for i in range(n):
                    new_adj[i][new_n - 1] = new_adj[new_n - 1][i] = candidate[i]
                adj = new_adj
                n = new_n
                success = True
                break
        
        if not success:
            # Try a different strategy: duplicate a vertex with modified adjacency
            found = False
            for src in range(n):
                # Try duplicating vertex src with slight modifications
                for mod in range(min(5, n)):
                    candidate = [adj[src][i] for i in range(n)]
                    if mod < n:
                        candidate[mod] = not candidate[mod]
                    if _try_add_vertex(adj, n, candidate):
                        new_n = n + 1
                        new_adj = [[False] * new_n for _ in range(new_n)]
                        for i in range(n):
                            for j in range(i + 1, n):
                                new_adj[i][j] = new_adj[j][i] = adj[i][j]
                        for i in range(n):
                            new_adj[i][new_n - 1] = new_adj[new_n - 1][i] = candidate[i]
                        adj = new_adj
                        n = new_n
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    
    # Convert adjacency matrix to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    
    # Final verification
    assert _is_55_free(adj, n), "Final graph must be (5,5)-free"
    
    return n, edges
