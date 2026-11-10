import math

def construct():
    # 1. Generate candidates
    # V = {v in [1, 255] | gcd(v, 255) > 1}
    # We want max clique containing 255.
    # 255 is universal in V.
    # So MaxClique(V) = MaxClique(V \ {255}) U {255}
    
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1 and i != 255:
            candidates.append(i)
            
    n = len(candidates)
    
    # Map index to value for easier handling in algorithm?
    # Or just use values. Values are unique.
    # Let's use indices 0..n-1 for speed.
    
    # Build adjacency list/matrix
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    best_clique = []
    best_size = 0
    
    # Bron-Kerbosch with pivot
    # R: current clique
    # P: candidates that can be added to R
    # X: candidates that were in P but are now in X (to avoid duplicates)
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        if not P:
            return

        # Select pivot u from P U X with maximum degree in P
        # This reduces the number of recursive calls
        # We want to minimize |P \ N(u)|
        max_deg = -1
        pivot = None
        for u in list(P) + list(X):
            deg = len(adj[u] & P)
            if deg > max_deg:
                max_deg = deg
                pivot = u
                
        # Iterate over P \ N(pivot)
        # If pivot is None (P and X empty? handled above), skip
        if pivot is not None:
            candidates_to_process = P - adj[pivot]
        else:
            candidates_to_process = P
            
        for v in candidates_to_process:
            # Recurse
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            # Remove v from P and add to X
            P.remove(v)
            X.add(v)
            
    # Initial call
    # R = empty
    # P = all vertices
    # X = empty
    P_initial = set(range(n))
    X_initial = set()
    R_initial = set()
    
    bron_kerbosch(R_initial, P_initial, X_initial)
    
    # Convert indices back to values
    result_values = [candidates[i] for i in best_clique]
    # Add 255
    result_values.append(255)
    
    return result_values
