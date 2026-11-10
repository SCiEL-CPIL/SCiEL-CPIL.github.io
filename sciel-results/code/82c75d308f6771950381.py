"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix/list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Bron-Kerbosch with pivot to find maximum clique
    # R: set of vertices in the current clique
    # P: set of candidates to be added to R
    # X: set of vertices already processed and removed from P
    max_clique_size = 0
    max_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique_size, max_clique
        
        if len(P) == 0 and len(X) == 0:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                max_clique = list(R)
            return
        
        if len(P) == 0:
            # R is a maximal clique
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                max_clique = list(R)
            return
            
        # Pivot selection: choose u from P union X that maximizes |P \ N(u)|
        # This minimizes the number of recursive calls
        union_p_x = P.union(X)
        if not union_p_x:
            return
            
        # Pick a pivot
        # For efficiency, we can just pick one with max degree in P
        # But standard is max |P \ N(u)|
        best_pivot = None
        max_uncovered = -1
        for u in union_p_x:
            # P minus neighbors of u
            uncovered = len(P.difference(adj[u]))
            if uncovered > max_uncovered:
                max_uncovered = uncovered
                best_pivot = u
                
        if best_pivot is None:
            return
            
        # Iterate over candidates in P that are not neighbors of the pivot
        # This is the pruning condition
        for v in list(P.difference(adj[best_pivot])):
            bron_kerbosch(R.union([v]), P.intersection(adj[v]), X.intersection(adj[v]))
            P.remove(v)
            X.add(v)
            
    # Initial call
    all_vertices = set(range(n))
    bron_kerbosch(set(), all_vertices, set())
    
    # Fallback if no clique found (should not happen for n>=1)
    if max_clique_size < 2:
        # Use the first edge as a 2-clique
        if edges:
            u, v = edges[0]
            max_clique = [u, v]
            max_clique_size = 2
        else:
            # Graph has no edges? Task implies edges exist.
            # Return trivial invalid? No, must return valid.
            # If n>=2 but no edges, we can't have 2 branches with paths.
            # But the graph provided has edges.
            max_clique = [0, 1] 
            max_clique_size = 2

    # Construct paths for the max clique
    # Since max_clique is a clique, direct edges exist for all pairs.
    branches = [int(v) for v in max_clique]
    paths = []
    for u, v in itertools.combinations(branches, 2):
        paths.append([u, v])
        
    return {'branches': branches, 'paths': paths}

# EVOLVE-BLOCK-END
