import math

def construct():
    # Step 1: Identify candidate vertices V'
    # V' = {v in [1, 255] | gcd(v, 255) > 1}
    candidates = []
    for v in range(1, 256):
        if math.gcd(v, 255) > 1:
            candidates.append(v)
    
    # We must include 255.
    # The rest of the set must form a clique in the subgraph induced by candidates excluding 255.
    # Let's build the graph on candidates excluding 255.
    # Note: 255 is in candidates.
    if 255 in candidates:
        candidates.remove(255)
    
    n = len(candidates)
    
    # Build adjacency list
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Bron-Kerbosch algorithm to find maximum clique
    # We use a recursive approach with pivoting.
    # To handle recursion depth or just standard implementation, we'll track the max size.
    
    max_clique_size = 0
    
    # Standard Bron-Kerbosch with Pivoting
    # R: current clique
    # P: candidates that can extend the clique
    # X: candidates that have already been processed (to avoid duplicates)
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique_size
        if not P and not X:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
            return
        
        # Choose pivot u from P U X to minimize the number of candidates
        # We want u such that |P \ N(u)| is minimized
        # If P U X is empty, we handled it above.
        pivot_candidates = list(P) + list(X)
        if not pivot_candidates:
            return
            
        # Find pivot u that maximizes |N(u) \cap P| which minimizes |P \ N(u)|
        # Actually, standard pivot choice: u in P U X maximizing |N(u) \cap P|
        best_pivot = -1
        max_neighbors_in_P = -1
        for u in pivot_candidates:
            # Count neighbors of u in P
            # Note: adj[u] contains neighbors in the original graph.
            # We need neighbors of u that are in P.
            count = 0
            for v in P:
                if v in adj[u]:
                    count += 1
            if count > max_neighbors_in_P:
                max_neighbors_in_P = count
                best_pivot = u
        
        if best_pivot == -1:
            return
            
        # Iterate over vertices in P \ N(best_pivot)
        # These are the vertices that are NOT adjacent to the pivot
        # We only need to recurse on these because cliques containing the pivot
        # are covered when the pivot is added to R in a previous step?
        # Actually, the logic is: any maximal clique must contain at least one vertex from P \ N(u) 
        # OR contain u. 
        # The recursive calls on v in P \ N(u) cover all cliques not containing u.
        # The clique containing u is covered when u is part of R in a different branch?
        # Wait, standard BK-P:
        # For each v in P \ N(u):
        #   BronKerbosch(R U {v}, P \ N(v), X \ N(v))
        #   P = P \ {v}
        #   X = X U {v}
        
        # We need to iterate over a copy of P \ N(best_pivot)
        vertices_to_process = [v for v in P if v not in adj[best_pivot]]
        
        for v in vertices_to_process:
            # New P: P \ N(v) \ {v}
            new_P = set()
            for x in P:
                if x != v and x not in adj[v]:
                    new_P.add(x)
            
            # New X: X \ N(v) \ {v}
            new_X = set()
            for x in X:
                if x != v and x not in adj[v]:
                    new_X.add(x)
            
            # Update R
            new_R = R | {v}
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # Remove v from P and add to X for subsequent iterations in this loop
            P.discard(v)
            X.add(v)
            
    # Initial call
    # R is empty
    # P is all vertices
    # X is empty
    P_initial = set(range(n))
    X_initial = set()
    R_initial = set()
    
    # Optimization: If n is large, BK might be slow. n=126 is manageable.
    # However, Python recursion limit might be an issue if depth is high, but clique size is limited.
    # Also, sets in Python are efficient enough for n=126.
    
    import sys
    sys.setrecursionlimit(10000)
    
    bron_kerbosch(R_initial, P_initial, X_initial)
    
    # The maximum clique size in the subgraph is max_clique_size.
    # The total set size is max_clique_size + 1 (for 255).
    # We need to return the actual set.
    # The code above only tracks size. We need to track the set itself.
    
    # Let's modify to track the best set.
    best_set = set()
    
    def bron_kerbosch_set(R, P, X):
        nonlocal best_set, max_clique_size
        if not P and not X:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                best_set = set(R)
            return
        
        pivot_candidates = list(P) + list(X)
        if not pivot_candidates:
            return
            
        best_pivot = -1
        max_neighbors_in_P = -1
        for u in pivot_candidates:
            count = 0
            for v in P:
                if v in adj[u]:
                    count += 1
            if count > max_neighbors_in_P:
                max_neighbors_in_P = count
                best_pivot = u
        
        if best_pivot == -1:
            return
            
        vertices_to_process = [v for v in P if v not in adj[best_pivot]]
        
        for v in vertices_to_process:
            new_P = set()
            for x in P:
                if x != v and x not in adj[v]:
                    new_P.add(x)
            
            new_X = set()
            for x in X:
                if x != v and x not in adj[v]:
                    new_X.add(x)
            
            new_R = R | {v}
            
            bron_kerbosch_set(new_R, new_P, new_X)
            
            P.discard(v)
            X.add(v)
            
    P_initial = set(range(n))
    X_initial = set()
    R_initial = set()
    max_clique_size = 0
    best_set = set()
    
    bron_kerbosch_set(R_initial, P_initial, X_initial)
    
    # Convert indices back to values
    result_values = [candidates[i] for i in best_set]
    result_values.append(255)
    
    return result_values

