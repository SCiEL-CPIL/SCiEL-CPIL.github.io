import time
import random
import sys
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build initial adjacency
    init_adj = [set() for _ in range(n)]
    init_edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        init_edge_set.add((u, v))
        init_adj[u].add(v)
        init_adj[v].add(u)

    def get_max_biclique(u, v, adj):
        """
        Find a maximal biclique containing edge (u,v) in the graph defined by adj.
        Tries both orientations and returns the larger one.
        """
        # Orientation 1: u in L, v in R
        # L must be neighbors of v (excluding u? no, u is in L, v is in R. 
        # Wait, if u in L and v in R, then every node in L must be connected to v.
        # So L subset of adj[v]. Every node in R must be connected to u.
        # So R subset of adj[u].
        # Also u must be in L, v must be in R.
        # And for all l in L, r in R, (l,r) in adj.
        
        L1 = set(adj[v])
        R1 = set(adj[u])
        
        # Ensure seed vertices are present
        if u not in L1 or v not in R1:
            size1 = 0
        else:
            # Iterative refinement
            changed = True
            while changed:
                changed = False
                # Remove from L1 any node not connected to all in R1
                new_L1 = {x for x in L1 if R1.issubset(adj[x])}
                if len(new_L1) != len(L1):
                    L1 = new_L1
                    changed = True
                # Remove from R1 any node not connected to all in L1
                new_R1 = {y for y in R1 if L1.issubset(adj[y])}
                if len(new_R1) != len(R1):
                    R1 = new_R1
                    changed = True
            
            size1 = len(L1) * len(R1)
            # Verify u in L1, v in R1
            if u not in L1 or v not in R1:
                size1 = 0

        # Orientation 2: u in R, v in L
        L2 = set(adj[u])
        R2 = set(adj[v])
        
        if v not in L2 or u not in R2:
            size2 = 0
        else:
            changed = True
            while changed:
                changed = False
                new_L2 = {x for x in L2 if R2.issubset(adj[x])}
                if len(new_L2) != len(L2):
                    L2 = new_L2
                    changed = True
                new_R2 = {y for y in R2 if L2.issubset(adj[y])}
                if len(new_R2) != len(R2):
                    R2 = new_R2
                    changed = True
            
            size2 = len(L2) * len(R2)
            if v not in L2 or u not in R2:
                size2 = 0

        if size1 >= size2:
            return size1, list(L1), list(R1)
        else:
            return size2, list(L2), list(R2)

    def solve_once(seed_val):
        random.seed(seed_val)
        adj = [set(s) for s in init_adj]
        remaining = set(init_edge_set)
        bicliques = []
        
        while remaining:
            best_size = -1
            best_biclique = None
            best_edges = None
            
            # Sample edges to avoid O(m^2) if m is large, but m is small here (189)
            # So we can check all edges.
            edges_list = list(remaining)
            random.shuffle(edges_list)
            
            for u, v in edges_list:
                size, L, R = get_max_biclique(u, v, adj)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)
                    # Precompute edges covered for speed
                    best_edges = set()
                    for x in L:
                        for y in R:
                            e = (min(x, y), max(x, y))
                            best_edges.add(e)
            
            if best_biclique is None:
                # Fallback: pick any edge
                u, v = next(iter(remaining))
                best_biclique = ([u], [v])
                best_edges = {(u, v)}
                
            L, R = best_biclique
            bicliques.append([L, R])
            
            # Remove covered edges from remaining and adj
            for e in best_edges:
                if e in remaining:
                    remaining.remove(e)
                    x, y = e
                    if y in adj[x]:
                        adj[x].remove(y)
                    if x in adj[y]:
                        adj[y].remove(x)
                        
        return bicliques

    def local_search(bicliques, deadline):
        """
        Try to reduce the number of bicliques by merging or replacing.
        1. Merge: If two bicliques can be merged into one, do it.
        2. Replace: If a set of bicliques can be replaced by fewer bicliques, do it.
        """
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            n_bicliques = len(bicliques)
            
            # Try merging pairs
            for i in range(n_bicliques):
                for j in range(i + 1, n_bicliques):
                    if time.time() > deadline:
                        return bicliques
                    
                    # Get edges of biclique i and j
                    edges_i = set()
                    for x in bicliques[i][0]:
                        for y in bicliques[i][1]:
                            edges_i.add((min(x, y), max(x, y)))
                            
                    edges_j = set()
                    for x in bicliques[j][0]:
                        for y in bicliques[j][1]:
                            edges_j.add((min(x, y), max(x, y)))
                            
                    combined_edges = edges_i.union(edges_j)
                    
                    # Try to cover combined_edges with 1 biclique
                    # We do this by running a mini-solver on these edges
                    # But we need to be careful: the biclique must use only edges from combined_edges?
                    # No, the biclique must be a valid biclique in the original graph, 
                    # and it must cover all edges in combined_edges.
                    # Actually, the task says "cross-edge sets must partition all pinned graph edges".
                    # So if we merge, the new biclique must cover exactly the union of edges.
                    # And it must be a valid biclique (all cross edges exist).
                    
                    # Let's try to find a biclique that covers combined_edges.
                    # We can use get_max_biclique on one of the edges in combined_edges,
                    # and check if the resulting biclique covers ALL of combined_edges.
                    
                    found_merge = False
                    for u, v in combined_edges:
                        size, L, R = get_max_biclique(u, v, init_adj) # Use original adj for validity check?
                        # Wait, get_max_biclique uses adj to find maximal.
                        # If we use init_adj, it finds the maximal biclique in the full graph containing (u,v).
                        # We need to check if this biclique covers combined_edges.
                        
                        # But there might be a smaller biclique that covers combined_edges but is not maximal in the full graph?
                        # If a biclique covers combined_edges, then any superset of that biclique (in terms of vertices) 
                        # that is still a biclique would also cover them.
                        # So checking the maximal biclique is sufficient. If the maximal one doesn't cover them, 
                        # no biclique containing (u,v) can cover them?
                        # Not necessarily. The maximal biclique might have "bad" vertices that prevent other good vertices from being included?
                        # No, the definition of maximal biclique here is "cannot add any more vertices".
                        # If there exists a biclique B covering E, then there exists a maximal biclique M such that B subset M.
                        # So M covers E.
                        # So yes, checking the maximal biclique in the full graph is sufficient.
                        
                        covered = set()
                        for x in L:
                            for y in R:
                                covered.add((min(x, y), max(x, y)))
                                
                        if combined_edges.issubset(covered):
                            # Check if covered is a subset of edges that are currently "free" or just valid?
                            # The new biclique will replace i and j.
                            # The edges covered by the new biclique might include edges that were covered by other bicliques?
                            # No, because the edges are partitioned.
                            # If covered contains edges not in combined_edges, those edges are covered by other bicliques.
                            # This would violate the partition property (double covering).
                            # So we must ensure covered == combined_edges.
                            # Or, we can allow the new biclique to cover a superset, and then remove those edges from other bicliques?
                            # That's complex.
                            # Let's stick to strict partition: covered must be exactly combined_edges.
                            # Or, more loosely, we can try to re-solve the subproblem.
                            
                            # Let's try the strict check first.
                            if covered == combined_edges:
                                new_b = [L, R]
                                # Remove i and j
                                bicliques.pop(j)
                                bicliques.pop(i)
                                bicliques.append(new_b)
                                improved = True
                                found_merge = True
                                break
                    if found_merge:
                        break
                if improved:
                    break
            
            if not improved:
                # Try replacing 2 bicliques with 1 (relaxed)
                # Or 3 with 2?
                # For now, just pair merging.
                pass
                
        return bicliques

    best_bicliques = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 55
    
    restart = 0
    while time.time() < deadline:
        result = solve_once(restart)
        
        if result is None:
            restart += 1
            continue
            
        # Local Search
        result = local_search(result, deadline)
        
        if len(result) < best_count:
            best_count = len(result)
            best_bicliques = result
            
        if best_count == 1:
            break
            
        restart += 1

    # Fallback
    if best_bicliques is None:
        best_bicliques = []
        uncovered = set(init_edge_set)
        while uncovered:
            u, v = next(iter(uncovered))
            best_bicliques.append([[u], [v]])
            uncovered.remove((u, v))

    # Convert to required format
    final_result = []
    for L, R in best_bicliques:
        # Ensure types are list of int
        # L and R are lists from get_max_biclique or local search
        # Make sure they are lists of ints
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        final_result.append([L_int, R_int])
        
    return final_result
