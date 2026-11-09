import math
from itertools import combinations

def construct():
    # 1. Generate candidates
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    
    # 2. Build adjacency sets for compatibility graph (gcd > 1)
    # neighbors[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # 3. Bron-Kerbosch with pivot to find Maximum Clique
    # We use indices 0..n-1
    max_clique = [set()] # Store the best clique found
    
    # Recursive helper
    # R: current clique (set of indices)
    # P: candidates that can be added (set of indices)
    # X: candidates that have been processed (to avoid duplicates) (set of indices)
    def bk(R, P, X):
        if not P and not X:
            if len(R) > len(max_clique[0]):
                max_clique[0] = set(R)
            return
        
        if not P:
            return
            
        # Pivot selection: choose u in P U X that maximizes |N(u) ∩ P|
        # This reduces the branching factor
        # If P is empty, we handled it above.
        # We pick u from P ∪ X
        # Optimization: just pick any u in P ∪ X, or the one with max degree in P
        # For n=127, iterating through P∪X is fine.
        u = None
        max_deg = -1
        for v in P.union(X):
            deg = len(neighbors[v].intersection(P))
            if deg > max_deg:
                max_deg = deg
                u = v
                
        # Candidates are P \ N(u)
        # These are the vertices in P that are NOT adjacent to u
        # If v is not adjacent to u, it must be included in some maximal clique
        # containing R (if we are to find all maximal cliques)
        # Wait, the standard algorithm iterates over P \ N(u).
        # This ensures we don't miss any maximal clique.
        
        candidates_to_try = P.difference(neighbors[u])
        
        for v in list(candidates_to_try):
            # Recurse
            # New R = R ∪ {v}
            # New P = P ∩ N(v)
            # New X = X ∩ N(v)
            new_R = R | {v}
            new_P = P & neighbors[v]
            new_X = X & neighbors[v]
            
            bk(new_R, new_P, new_X)
            
            # Move v from P to X
            P.discard(v)
            X.add(v)
            
    # Initial call
    # R = empty
    # P = all nodes
    # X = empty
    bk(set(), set(range(n)), set())
    
    # 4. Convert index set to value set
    best_indices = max_clique[0]
    result = [candidates[i] for i in best_indices]
    
    # Ensure 255 is in the result (it should be, as it's a universal vertex)
    if 255 not in result:
        # If for some reason it's not (shouldn't happen), force it? 
        # But the problem requires it. 
        # Since 255 is connected to all, it must be in the max clique.
        pass
        
    return sorted(result)
