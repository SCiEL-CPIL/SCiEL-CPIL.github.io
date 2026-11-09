"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as sets for fast operations
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    total_edges = len(edge_set)
    
    def greedy_partition(edge_order, deadline):
        """Greedy biclique partition given an edge ordering."""
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            if time.time() > deadline:
                break
                
            # Pick the first edge in the ordering that's still remaining
            seed_edge = None
            for e in edge_order:
                if e in remaining:
                    seed_edge = e
                    break
            
            if seed_edge is None:
                break
            
            u, v = seed_edge
            L = {u}
            R = {v}
            
            # Iteratively expand L and R
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                
                # Try to add vertices to L: must be connected to all in R
                for w in range(n):
                    if w not in L and w not in R:
                        if R.issubset(adj[w]):
                            L.add(w)
                            changed = True
                
                # Try to add vertices to R: must be connected to all in L
                for w in range(n):
                    if w not in L and w not in R:
                        if L.issubset(adj[w]):
                            R.add(w)
                            changed = True
            
            # Extract the biclique edges
            biclique_edges = set()
            for a in L:
                for b in R:
                    if a != b and (min(a,b), max(a,b)) in remaining:
                        biclique_edges.add((min(a,b), max(a,b)))
            
            if not biclique_edges:
                # Fallback: just take the seed edge
                biclique_edges = {seed_edge}
                L = {u}
                R = {v}
            
            bicliques.append([sorted(L), sorted(R)])
            remaining -= biclique_edges
        
        return bicliques
    
    def validate_and_count(bicliques):
        """Validate biclique partition and return count."""
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return None
            for a in L:
                for b in R:
                    if a != b:
                        e = (min(a,b), max(a,b))
                        if e in covered:
                            return None  # Overlap
                        if e not in edge_set:
                            return None  # Non-edge
                        covered.add(e)
        if covered != edge_set:
            return None  # Missing edges
        return len(bicliques)
    
    # Generate multiple edge orderings
    random.seed(42)
    orderings = []
    
    # Ordering 1: by edge weight (degree product)
    deg = [len(adj[i]) for i in range(n)]
    order1 = sorted(edge_set, key=lambda e: -(deg[e[0]] * deg[e[1]]))
    orderings.append(order1)
    
    # Ordering 2: random
    order2 = list(edge_set)
    random.shuffle(order2)
    orderings.append(order2)
    
    # Ordering 3: by sum of degrees
    order3 = sorted(edge_set, key=lambda e: -(deg[e[0]] + deg[e[1]]))
    orderings.append(order3)
    
    # Ordering 4: reverse degree product
    order4 = sorted(edge_set, key=lambda e: deg[e[0]] * deg[e[1]])
    orderings.append(order4)
    
    # Ordering 5: another random
    order5 = list(edge_set)
    random.shuffle(order5)
    orderings.append(order5)
    
    # Try each ordering with time limit
    start_time = time.time()
    overall_deadline = start_time + 85  # Safety margin
    
    best_bicliques = None
    best_count = float('inf')
    
    for i, ordering in enumerate(orderings):
        if time.time() > overall_deadline:
            break
        
        # Give each ordering a fair share of time
        per_ordering_deadline = overall_deadline - (len(orderings) - i - 1) * 5
        
        bicliques = greedy_partition(ordering, per_ordering_deadline)
        count = validate_and_count(bicliques)
        
        if count is not None and count < best_count:
            best_count = count
            best_bicliques = bicliques
    
    # If no valid result, fall back to seed approach
    if best_bicliques is None:
        bicliques = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                bicliques.append([[u], neighbors])
        best_bicliques = bicliques
        best_count = len(bicliques)
    
    # Local improvement: try to merge bicliques
    if best_bicliques and best_count > 1:
        improved = True
        while improved and time.time() < overall_deadline:
            improved = False
            new_bicliques = []
            for i in range(len(best_bicliques)):
                if time.time() > overall_deadline:
                    break
                if i in [b for b in range(len(best_bicliques)) if b not in new_bicliques]:
                    L_i = set(best_bicliques[i][0])
                    R_i = set(best_bicliques[i][1])
                    merged_with = False
                    for j in range(len(best_bicliques)):
                        if i == j or j in [b for b in range(len(best_bicliques)) if b not in new_bicliques]:
                            continue
                        L_j = set(best_bicliques[j][0])
                        R_j = set(best_bicliques[j][1])
                        
                        # Try merging: new L = L_i ∪ L_j, new R = R_i ∪ R_j
                        # Check if all cross edges exist
                        new_L = L_i | L_j
                        new_R = R_i | R_j
                        
                        valid = True
                        for a in new_L:
                            for b in new_R:
                                if a != b and (min(a,b), max(a,b)) not in edge_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            # Check no overlap with existing bicliques in new_bicliques
                            new_biclique_edges = set()
                            for a in new_L:
                                for b in new_R:
                                    if a != b:
                                        new_biclique_edges.add((min(a,b), max(a,b)))
                            
                            overlap = False
                            for k, (L_k, R_k) in enumerate(new_bicliques):
                                for a in L_k:
                                    for b in R_k:
                                        if a != b and (min(a,b), max(a,b)) in new_biclique_edges:
                                            overlap = True
                                            break
                                    if overlap:
                                        break
                                if overlap:
                                    break
                            
                            if not overlap:
                                new_bicliques.append([sorted(new_L), sorted(new_R)])
                                merged_with = True
                                break
                    
                    if not merged_with:
                        new_bicliques.append(best_bicliques[i])
            
            # Validate new partition
            count = validate_and_count(new_bicliques)
            if count is not None and count < best_count:
                best_bicliques = new_bicliques
                best_count = count
                improved = True
    
    # Final validation
    final_count = validate_and_count(best_bicliques)
    if final_count is None:
        # Emergency fallback
        bicliques = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                bicliques.append([[u], neighbors])
        best_bicliques = bicliques
    
    # Convert to required format
    final_result = []
    for bic in best_bicliques:
        l = [int(x) for x in bic[0]]
        r = [int(x) for x in bic[1]]
        final_result.append([l, r])
    
    return final_result
# EVOLVE-BLOCK-END

