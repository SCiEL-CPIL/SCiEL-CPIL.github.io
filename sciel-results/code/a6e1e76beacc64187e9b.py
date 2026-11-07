"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    # Function to find maximal cliques in a subgraph defined by a set of vertices
    def find_maximal_cliques(vertices, adj_subset):
        # adj_subset is a dict or list of sets restricted to vertices
        # We use Bron-Kerbosch
        cliques = []
        V = set(vertices)
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                cliques.append(list(R))
                return
            
            # Choose pivot u from P ∪ X
            u = None
            if P:
                u = next(iter(P))
            elif X:
                u = next(iter(X))
            
            if u is None:
                return
                
            # Iterate over P \ N(u)
            for v in list(P):
                if v == u:
                    continue
                if u in adj_subset.get(v, set()):
                    continue # Skip if v is connected to u
                
                bron_kerbosch(R | {v}, P & adj_subset.get(v, set()), X & adj_subset.get(v, set()))
                P.remove(v)
                X.add(v)
                
        bron_kerbosch(set(), V, set())
        return cliques

    # Generate candidate bicliques
    candidates = []
    
    # 1. Stars: [v, neighbors]
    for v in range(n):
        candidates.append((frozenset([v]), frozenset(adj[v])))
        
    # 2. Bicliques with left side size 1, right side a clique in N(v)
    #    This is covered by stars if we consider the star as the max, but 
    #    a clique in N(v) is a subset of N(v). 
    #    Actually, if C is a clique in N(v), then {v} x C is a biclique.
    #    The star {v} x N(v) is a superset. 
    #    However, {v} x N(v) is a valid biclique. 
    #    Why would we need {v} x C? 
    #    Because {v} x N(v) might contain edges that are already covered, 
    #    while {v} x C might not? 
    #    No, if {v} x N(v) has a covered edge, {v} x C might still have it if C contains that neighbor.
    #    But {v} x C is smaller. 
    #    The greedy algorithm picks the *largest* valid biclique.
    #    So the star is the best candidate for left-side size 1.
    #    We don't need to generate all subsets.
    
    # 3. Bicliques with left side size 2: [u, v] x C where C is a clique in N(u) ∩ N(v)
    #    This can be larger than stars if the common neighborhood is large.
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            common = adj[u] & adj[v]
            if not common:
                continue
            # The biclique is {u, v} x common
            # Is {u, v} x common a valid biclique?
            # Yes, because u and v are connected to all in common.
            # And u-v is an edge (within left side, allowed).
            # Edges between u and common are in graph.
            # Edges between v and common are in graph.
            candidates.append((frozenset([u, v]), frozenset(common)))
            
    # 4. Bicliques with left side size 3?
    #    [u, v, w] x C where C is clique in N(u)∩N(v)∩N(w)
    #    This might be overkill and slow to generate.
    #    Let's stick to size 1 and 2 for now.
    
    # Deduplicate candidates
    unique_candidates = list(set(candidates))
    
    # Greedy Set Cover
    uncovered = set(edge_set)
    bicliques = []
    
    while uncovered:
        best_biclique = None
        best_count = -1
        
        for L, R in unique_candidates:
            # Check if all edges in L x R are uncovered
            # And if L and R are disjoint (they are by construction: 
            # stars: v not in adj[v] (no self loops)
            # pairs: u, v not in common (since u!=v and no self loops, u not in N(v)? 
            # Wait, u is in N(v). So u is in adj[v]. 
            # common = N(u) ∩ N(v). 
            # Is u in common? u is in N(v). Is u in N(u)? No. 
            # So u is NOT in N(u). Thus u is not in common.
            # So L and R are disjoint.
            
            valid = True
            count = 0
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in uncovered:
                        count += 1
                    else:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and count > best_count:
                best_count = count
                best_biclique = (L, R)
                
        if best_biclique is None or best_count <= 0:
            # Fallback: pick any uncovered edge
            e = next(iter(uncovered))
            u, v = e
            best_biclique = (frozenset([u]), frozenset([v]))
            best_count = 1
        
        L, R = best_biclique
        # Remove edges from uncovered
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                uncovered.discard(e)
        
        bicliques.append([list(L), list(R)])
        
    # Convert to proper format
    result = []
    for L, R in bicliques:
        L_ints = [int(x) for x in L]
        R_ints = [int(x) for x in R]
        result.append([L_ints, R_ints])
    
    return result
# EVOLVE-BLOCK-END
