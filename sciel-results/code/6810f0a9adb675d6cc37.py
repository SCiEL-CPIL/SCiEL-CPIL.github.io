import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix for fast edge check
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
            
    def is_biclique(L, R):
        """Check if all pairs in L x R are edges in the graph."""
        for u in L:
            for v in R:
                if not adj[u][v]:
                    return False
        return True

    def greedy_cover(uncovered_edges):
        """
        Greedily cover the set of edges 'uncovered_edges' with bicliques.
        Returns a list of bicliques [L, R].
        """
        bicliques = []
        uncovered = set(uncovered_edges)
        
        while uncovered:
            if time.time() > deadline:
                return bicliques # Return partial? No, must be valid. 
                # We should ensure we don't time out mid-cover.
                # If we time out, we should probably return the previous best valid solution.
                # So we shouldn't call this if we are close to deadline.
                
            # Pick a random edge
            # Convert to list for random choice? Or iterate.
            # For efficiency, pick first available or random. Random helps variety.
            edge = random.choice(list(uncovered))
            u, v = edge
            
            L = [u]
            R = [v]
            
            # Expand L and R
            changed = True
            while changed:
                changed = False
                # Try to add to L
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # Check if w is connected to all in R via uncovered edges
                    can_add_L = True
                    for r in R:
                        e = (w, r) if w < r else (r, w)
                        if e not in uncovered:
                            can_add_L = False
                            break
                    if can_add_L:
                        L.append(w)
                        changed = True
                        
                # Try to add to R
                for w in range(n):
                    if w in L or w in R:
                        continue
                    can_add_R = True
                    for l in L:
                        e = (l, w) if l < w else (w, l)
                        if e not in uncovered:
                            can_add_R = False
                            break
                    if can_add_R:
                        R.append(w)
                        changed = True
            
            # Remove edges covered by this biclique from uncovered
            for u_l in L:
                for v_r in R:
                    e = (u_l, v_r) if u_l < v_r else (v_r, u_l)
                    uncovered.discard(e)
            
            bicliques.append([L, R])
            
        return bicliques

    def merge_pass(bicliques):
        """
        Try to merge pairs of bicliques if their union of edges forms a biclique.
        Returns new list of bicliques.
        """
        changed = True
        while changed:
            changed = False
            new_bicliques = []
            i = 0
            while i < len(bicliques):
                B1 = bicliques[i]
                merged = False
                for j in range(i + 1, len(bicliques)):
                    B2 = bicliques[j]
                    L1, R1 = B1
                    L2, R2 = B2
                    
                    L_new = list(set(L1) | set(L2))
                    R_new = list(set(R1) | set(R2))
                    
                    # Check if L_new x R_new is a biclique in G
                    if is_biclique(L_new, R_new):
                        # Also need to ensure that the edges covered by B1 and B2
                        # are exactly the edges in L_new x R_new?
                        # No, the union of edges E1 U E2 must be a biclique.
                        # If is_biclique(L_new, R_new) is true, then all edges in L_new x R_new exist in G.
                        # The edges covered by the new biclique are L_new x R_new.
                        # We must ensure that E1 U E2 == L_new x R_new.
                        # If there are "extra" edges in L_new x R_new that were not in E1 or E2,
                        # then the new biclique covers more edges than the union of the old ones.
                        # This would mean we are covering edges that were supposed to be covered by other bicliques?
                        # No, the partition property: E(B_new) must be a subset of the total graph edges.
                        # And the union of all biclique edge sets must be the graph edges.
                        # If we merge B1 and B2 into B_new, we replace E1 U E2 with E(B_new).
                        # If E(B_new) contains edges not in E1 U E2, those edges were covered by other bicliques.
                        # So we would be double-covering them.
                        # Thus, we must check that E(B_new) == E1 U E2.
                        
                        E1 = set()
                        for u in L1:
                            for v in R1:
                                E1.add((u,v) if u<v else (v,u))
                                E1.add((v,u) if u<v else (u,v)) # just canonical
                        # Actually, let's just generate the set of canonical edges.
                        E1_set = set()
                        for u in L1:
                            for v in R1:
                                if u < v: E1_set.add((u,v))
                                else: E1_set.add((v,u))
                                
                        E2_set = set()
                        for u in L2:
                            for v in R2:
                                if u < v: E2_set.add((u,v))
                                else: E2_set.add((v,u))
                                
                        E_union = E1_set | E2_set
                        E_new_set = set()
                        for u in L_new:
                            for v in R_new:
                                if u < v: E_new_set.add((u,v))
                                else: E_new_set.add((v,u))
                                
                        if E_union == E_new_set:
                            # Valid merge
                            bicliques[j] = None # Mark for removal
                            B1 = [L_new, R_new]
                            merged = True
                            changed = True
                            break
                            
                if B1 is not None:
                    new_bicliques.append(B1)
                i += 1
            bicliques = [b for b in new_bicliques if b is not None]
            
        return bicliques

    # Initial Solution
    # Use multiple restarts for initial greedy to find a good starting point
    best_bicliques = None
    best_count = float('inf')
    
    for restart in range(50):
        if time.time() > deadline - 10: break
        
        # Randomize the order of edges to get different greedy outcomes
        # We can shuffle the list of edges before passing to greedy_cover?
        # greedy_cover picks random edge from set. Set order is not random.
        # To make it deterministic-ish but varied, we can just rely on random.choice.
        
        initial = greedy_cover(edge_set)
        if len(initial) < best_count:
            best_count = len(initial)
            best_bicliques = [ [list(b[0]), list(b[1])] for b in initial ]
            
    if best_bicliques is None:
        # Fallback: one biclique per edge
        best_bicliques = [[ [u], [v] ] for u, v in edge_set]
        best_count = len(best_bicliques)

    # Local Search / Refinement
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Merge Pass
        new_bicliques = merge_pass(best_bicliques)
        if len(new_bicliques) < best_count:
            best_bicliques = new_bicliques
            best_count = len(new_bicliques)
            improved = True
            continue # Re-run merge pass on the new set
            
        # 2. Re-optimization
        # Pick a random subset of bicliques to "un-cover" and re-greedy-cover
        # Size of subset: 2 to 5
        k = random.randint(2, 5)
        if len(best_bicliques) < k:
            k = len(best_bicliques)
            
        if k > 0:
            indices = random.sample(range(len(best_bicliques)), k)
            # Collect edges from these bicliques
            edges_to_recover = set()
            for idx in indices:
                L, R = best_bicliques[idx]
                for u in L:
                    for v in R:
                        edges_to_recover.add((u,v) if u<v else (v,u))
                        
            # Remove these bicliques
            # Create a new list without these indices
            remaining_bicliques = [best_bicliques[i] for i in range(len(best_bicliques)) if i not in indices]
            
            # Greedy cover the edges_to_recover
            # Note: These edges are currently "uncovered" in the context of the new solution
            # The rest of the graph is covered by remaining_bicliques
            new_subset = greedy_cover(edges_to_recover)
            
            # Combine
            candidate = remaining_bicliques + new_subset
            
            # Check validity?
            # The greedy_cover ensures edges in new_subset are covered.
            # The remaining_bicliques cover the rest.
            # Since we only touched edges in edges_to_recover, and new_subset covers exactly those?
            # No, greedy_cover covers the input set.
            # So the union of edges in remaining_bicliques and new_subset is the total graph edges.
            # And they are disjoint?
            # The edges in new_subset are a subset of edges_to_recover.
            # The edges in remaining_bicliques are disjoint from edges_to_recover.
            # So yes, they partition the graph.
            
            if len(candidate) < best_count:
                best_bicliques = candidate
                best_count = len(candidate)
                improved = True
                # Run merge pass immediately
                new_bicliques = merge_pass(best_bicliques)
                if len(new_bicliques) < best_count:
                    best_bicliques = new_bicliques
                    best_count = len(new_bicliques)

    # Final Merge Pass
    best_bicliques = merge_pass(best_bicliques)
    
    # Format output
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
        
    return result
