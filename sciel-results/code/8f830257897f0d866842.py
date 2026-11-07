import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 85
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    all_edges = list(edge_set)
    total_edges = len(all_edges)
    
    # Seed with a trivial solution: each edge is its own biclique
    # But we can do better by grouping edges that share a vertex
    # Initial solution: for each vertex, one biclique with all its incident edges
    # This gives at most n bicliques (actually fewer, only for vertices with edges)
    
    # Better initial: greedy approach
    # Sort edges by some heuristic
    # For each edge, try to add it to an existing biclique or create a new one
    
    # Let's use a simpler approach: start with vertex-based bicliques
    # For each vertex u, create a biclique [u, neighbors(u)]
    # This covers all edges incident to u, but an edge (u,v) is covered by both u's and v's biclique
    # So we need to be careful: we need a partition
    
    # Actually, the seed solution is: for each u, [u, all neighbors]. 
    # This is valid because:
    # - Each cross pair (u, v) where v in neighbors(u) is an edge
    # - The cross-edge sets partition all edges: edge (u,v) is in u's biclique (as cross edge u-v)
    #   Wait, no. In biclique [u, neighbors(u)], the cross edges are {(u, v) : v in neighbors(u)}
    #   So edge (u,v) appears in u's biclique AND in v's biclique. That's not a partition!
    
    # Let me re-read the seed:
    # return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
    # This creates for each u that appears as first element in edges: [u, [v for (a,v) in edges if a==u]]
    # So it's directed: only edges where u is the first vertex.
    # For edge (u,v), it appears only in u's biclique (if u < v in the edge list representation)
    # Actually, P['edges'] has edges like [0,4], [1,2], etc. The first element is always smaller?
    # Looking at the data: [0,4], [0,7], ... [1,2], [1,4], ... [2,6], ...
    # Yes, it seems edges are listed with smaller vertex first.
    # So each edge (u,v) with u<v is in u's biclique as cross edge (u,v).
    # This is a valid partition!
    
    # Count bicliques in seed:
    seed_bicliques = [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
    seed_count = len(seed_bicliques)
    
    best_bicliques = seed_bicliques
    best_count = seed_count
    
    # Now try to improve by merging bicliques
    # A biclique [L, R] has cross edges L x R (all must be edges)
    # We can try to merge two bicliques if the result is still a valid biclique
    # or restructure
    
    # Alternative: use a local search
    # Represent the partition as a list of bicliques
    # Try moves:
    # 1. Move an edge from one biclique to another
    # 2. Merge two bicliques
    # 3. Split a biclique
    
    # Let's try a simpler approach: greedy edge assignment
    # Sort edges by frequency (edges in denser parts first)
    # For each edge, assign it to the best existing biclique or create a new one
    
    # Actually, let's try to find maximal bicliques and use a set cover approach
    # But that's NP-hard. Let's do local search on the seed solution.
    
    # Current solution: bicliques[i] = [left_i, right_i]
    # Cross edges of biclique i: {(u,v) : u in left_i, v in right_i}
    # These must be disjoint and cover all edges
    
    # Move: take an edge e from biclique A and move to biclique B
    # This requires restructuring both bicliques
    
    # Simpler move: merge two bicliques A and B into one
    # A = [L_A, R_A], B = [L_B, R_B]
    # Merged could be [L_A ∪ L_B, R_A ∪ R_B] but that's not necessarily a biclique
    # Or [L_A, R_A ∪ R_B] if L_A x (R_A ∪ R_B) are all edges
    # Or [L_A ∪ L_B, R_A] if (L_A ∪ L_B) x R_A are all edges
    # Or other combinations
    
    # Let's try: for each pair of bicliques, check if they can be merged
    # Merge condition: there exists L, R such that L x R covers all cross edges of A and B
    # and L x R are all edges in the graph
    
    # This is complex. Let's try a different approach:
    # Use the fact that the graph is relatively small (30 vertices, ~180 edges)
    # Try to find a small number of bicliques via random restarts + local search
    
    def is_valid_biclique(left, right, adj):
        if not left or not right:
            return False
        left_set = set(left)
        right_set = set(right)
        if left_set & right_set:
            return False
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
        return True
    
    def get_cross_edges(left, right):
        edges_set = set()
        for u in left:
            for v in right:
                e = (min(u,v), max(u,v))
                edges_set.add(e)
        return edges_set
    
    def solution_edges(bicliques):
        all_e = set()
        for left, right in bicliques:
            for u in left:
                for v in right:
                    e = (min(u,v), max(u,v))
                    all_e.add(e)
        return all_e
    
    # Local search: try to improve by merging
    def try_merge(bicliques, adj):
        """Try to merge pairs of bicliques. Returns new list if improvement found."""
        n_bic = len(bicliques)
        for i in range(n_bic):
            for j in range(i+1, n_bic):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try merged = [L1 ∪ L2, R1 ∪ R2]
                # Check if it's a valid biclique
                L_new = list(set(L1) | set(L2))
                R_new = list(set(R1) | set(R2))
                if is_valid_biclique(L_new, R_new, adj):
                    new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
                    new_bicliques.append([L_new, R_new])
                    # Check that edges are still partitioned
                    if solution_edges(new_bicliques) == solution_edges(bicliques):
                        return new_bicliques
                
                # Try [L1, R1 ∪ R2]
                R_new = list(set(R1) | set(R2))
                if is_valid_biclique(L1, R_new, adj):
                    # Check edge coverage
                    old_edges = get_cross_edges(L1, R1) | get_cross_edges(L2, R2)
                    new_edges = get_cross_edges(L1, R_new)
                    if new_edges == old_edges:
                        new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
                        new_bicliques.append([L1, R_new])
                        return new_bicliques
                
                # Try [L1 ∪ L2, R1]
                L_new = list(set(L1) | set(L2))
                if is_valid_biclique(L_new, R1, adj):
                    old_edges = get_cross_edges(L1, R1) | get_cross_edges(L2, R2)
                    new_edges = get_cross_edges(L_new, R1)
                    if new_edges == old_edges:
                        new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
                        new_bicliques.append([L_new, R1])
                        return new_bicliques
        
        return None
    
    # Try to merge repeatedly
    bicliques = [list(b) for b in seed_bicliques]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_bicliques = try_merge(bicliques, adj)
        if new_bicliques is not None:
            bicliques = new_bicliques
            improved = True
    
    if len(bicliques) < best_count:
        best_bicliques = bicliques
        best_count = len(bicliques)
    
    # Try random restarts with different initial solutions
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a random initial solution
        # Start with all edges unassigned
        # Greedily assign edges to bicliques
        random_edges = list(all_edges)
        random.shuffle(random_edges)
        
        bicliques = []
        for e in random_edges:
            u, v = e
            assigned = False
            # Try to add to existing biclique
            for bi in range(len(bicliques)):
                L, R = bicliques[bi]
                # Can we add e to this biclique?
                # e = (u, v). We need u in L, v in R or u in R, v in L
                # And all cross pairs must be edges
                
                # Try adding v to R if u in L
                if u in L and v not in R:
                    new_R = list(R) + [v]
                    if is_valid_biclique(L, new_R, adj):
                        bicliques[bi] = [L, new_R]
                        assigned = True
                        break
                # Try adding u to L if v in R
                if v in R and u not in L:
                    new_L = list(L) + [u]
                    if is_valid_biclique(new_L, R, adj):
                        bicliques[bi] = [new_L, R]
                        assigned = True
                        break
                # Try adding u to R if v in L
                if v in L and u not in R:
                    new_R = list(R) + [u]
                    if is_valid_biclique(L, new_R, adj):
                        bicliques[bi] = [L, new_R]
                        assigned = True
                        break
                # Try adding v to L if u in R
                if u in R and v not in L:
                    new_L = list(L) + [v]
                    if is_valid_biclique(new_L, R, adj):
                        bicliques[bi] = [new_L, R]
                        assigned = True
                        break
            
            if not assigned:
                # Create new biclique
                bicliques.append([[u], [v]])
        
        # Local search: merge
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            new_bicliques = try_merge(bicliques, adj)
            if new_bicliques is not None:
                bicliques = new_bicliques
                improved = True
        
        if len(bicliques) < best_count:
            best_bicliques = bicliques
            best_count = len(bicliques)
    
    # Convert to proper format
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result

# EVOLVE-BLOCK-END
