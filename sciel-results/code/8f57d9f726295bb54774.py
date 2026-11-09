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
        L1 = set(adj[v])
        R1 = set(adj[u])
        
        if u not in L1 or v not in R1:
            size1 = 0
        else:
            changed = True
            while changed:
                changed = False
                new_L1 = {x for x in L1 if R1.issubset(adj[x])}
                if len(new_L1) != len(L1):
                    L1 = new_L1
                    changed = True
                new_R1 = {y for y in R1 if L1.issubset(adj[y])}
                if len(new_R1) != len(R1):
                    R1 = new_R1
                    changed = True
            
            size1 = len(L1) * len(R1)
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

    def solve_greedy(edges_to_cover, adj, deadline):
        """
        Greedily cover edges_to_cover with bicliques using adj for validity.
        """
        remaining = set(edges_to_cover)
        bicliques = []
        
        while remaining:
            if time.time() > deadline:
                return None
            
            best_size = -1
            best_biclique = None
            best_edges = None
            
            # Sample edges to avoid O(m^2) if m is large
            edges_list = list(remaining)
            random.shuffle(edges_list)
            
            for u, v in edges_list:
                size, L, R = get_max_biclique(u, v, adj)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)
                    best_edges = set()
                    for x in L:
                        for y in R:
                            e = (min(x, y), max(x, y))
                            # Only count edges that are in remaining
                            if e in remaining:
                                best_edges.add(e)
                    # If the biclique doesn't cover any remaining edges, it's useless
                    if not best_edges:
                        continue
            
            if best_biclique is None:
                # Fallback: pick any edge
                u, v = next(iter(remaining))
                best_biclique = ([u], [v])
                best_edges = {(u, v)}
                
            L, R = best_biclique
            bicliques.append([L, R])
            
            # Remove covered edges from remaining
            for e in best_edges:
                remaining.remove(e)
                
        return bicliques

    def get_edges_from_bicliques(bicliques):
        """
        Returns the set of edges covered by the given bicliques.
        """
        edges = set()
        for L, R in bicliques:
            for x in L:
                for y in R:
                    edges.add((min(x, y), max(x, y)))
        return edges

    def local_search(bicliques, deadline):
        """
        Local search using repacking moves.
        1. Try to repack a subset of bicliques (size 2 or 3) into fewer bicliques.
        """
        improved = True
        # Use a slightly earlier deadline for local search to ensure we return
        ls_deadline = deadline - 2.0
        
        while improved:
            improved = False
            if time.time() > ls_deadline:
                break
                
            n_bicliques = len(bicliques)
            
            # Try repacking pairs
            for i in range(n_bicliques):
                if time.time() > ls_deadline:
                    return bicliques
                for j in range(i + 1, n_bicliques):
                    if time.time() > ls_deadline:
                        return bicliques
                    
                    # Get edges covered by biclique i and j
                    edges_i = get_edges_from_bicliques([bicliques[i]])
                    edges_j = get_edges_from_bicliques([bicliques[j]])
                    combined_edges = edges_i.union(edges_j)
                    
                    # Try to cover combined_edges with 1 biclique
                    # We use the full graph adjacency for validity, but we only want to cover combined_edges
                    # The greedy solver will pick bicliques that cover edges in combined_edges.
                    # Since we are replacing i and j, the new bicliques must cover exactly combined_edges.
                    
                    # Optimization: Check if combined_edges itself is a biclique first (fast check)
                    # If it is, we can just use that one biclique.
                    # To check if a set of edges forms a biclique:
                    # 1. Find all vertices involved.
                    # 2. Check if there is a partition L, R such that all edges are between L and R and no other edges exist between L and R that are not in combined_edges?
                    # Actually, the biclique definition requires that *all* cross pairs in L, R are edges in the *graph*.
                    # It does NOT require that they are in combined_edges.
                    # However, the *partition* constraint says the cross-edge sets must partition the *pinned graph edges*.
                    # So if we use a biclique (L, R), it covers ALL edges between L and R in the graph.
                    # If that set of edges is not exactly combined_edges, we have a problem.
                    # If it's a superset, we are covering edges that belong to other bicliques -> Invalid partition.
                    # If it's a subset, we are not covering all edges in combined_edges -> Invalid partition.
                    # So the new biclique(s) must cover *exactly* combined_edges.
                    
                    # Let's run the greedy solver on combined_edges.
                    # We need to be careful: the greedy solver might pick a biclique that covers edges outside combined_edges.
                    # We need to filter candidates to only those that are subsets of combined_edges?
                    # No, a biclique is defined by vertices. If (L, R) is a biclique in G, it covers E(L, R) in G.
                    # We need E(L, R) == combined_edges.
                    
                    # Strategy:
                    # 1. Try to find a single biclique (L, R) such that E(L, R) == combined_edges.
                    # 2. If not found, try to find 2 bicliques that partition combined_edges. (This is harder, so we might skip for now and just try 1).
                    
                    found_better = False
                    best_new_bicliques = None
                    
                    # Try to find a single biclique covering exactly combined_edges
                    # We can iterate over edges in combined_edges and try to expand them, checking if the result matches combined_edges.
                    for u, v in combined_edges:
                        # Get maximal bicliques containing (u, v) in the FULL graph
                        # We need to check if any maximal biclique contains (u, v) has edge set == combined_edges
                        # Note: If a biclique B has E(B) == combined_edges, then B is a biclique in G.
                        # Any maximal biclique M containing B will have E(M) superset of E(B).
                        # If E(M) == E(B), then M=B.
                        # If E(M) != E(B), then M covers extra edges, which is invalid.
                        # So we only need to check maximal bicliques.
                        
                        size, L, R = get_max_biclique(u, v, init_adj)
                        covered = set()
                        for x in L:
                            for y in R:
                                covered.add((min(x, y), max(x, y)))
                        
                        if covered == combined_edges:
                            best_new_bicliques = [[L, R]]
                            found_better = True
                            break
                    
                    if found_better:
                        # Replace i and j with best_new_bicliques
                        # Remove j, then i
                        bicliques.pop(j)
                        bicliques.pop(i)
                        bicliques.extend(best_new_bicliques)
                        improved = True
                        break
                
                if improved:
                    break
            
            # If no pair merge found, try repacking 3 bicliques into 2?
            # This is O(N^3) and each repack is expensive.
            # Given N is small (around 10-20), N^3 is manageable.
            # But for now, let's stick to pair merges and see if it helps.
            # If we have time, we can add triple merges.
            
        return bicliques

    best_bicliques = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 55
    
    restart = 0
    while time.time() < deadline:
        random.seed(restart)
        
        # Initial greedy construction
        # We use the full graph for adjacency, but we only cover init_edge_set
        result = solve_greedy(init_edge_set, init_adj, deadline)
        
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
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        final_result.append([L_int, R_int])
        
    return final_result
