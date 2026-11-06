import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s limit

    # 255 = 3 * 5 * 17
    # Every element must share a factor with 255, so must be divisible by 3, 5, or 17.
    # The multiples of 3 form a clique (all share factor 3) and include 255.
    # Size: floor(255/3) = 85.
    # We attempt to find a larger clique via Bron-Kerbosch with pivot,
    # but always have the multiples-of-3 solution as a guaranteed baseline.
    
    # Guaranteed baseline: all multiples of 3 in [1, 255]
    baseline = list(range(3, 256, 3))  # 3, 6, 9, ..., 255
    best_clique = list(baseline)  # store as list of actual values
    best_size = len(best_clique)  # 85

    # Candidate pool: all numbers in [1, 255] divisible by 3, 5, or 17
    candidates = [i for i in range(1, 256) if (i % 3 == 0 or i % 5 == 0 or i % 17 == 0)]
    n = len(candidates)
    
    # Build adjacency matrix: adj[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True

    # Bron-Kerbosch with pivot to find maximum clique
    # We search for cliques larger than best_size
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        
        if len(R) + len(P) <= best_size:
            # Cannot beat current best
            return
        
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        if not P:
            return
        
        # Pivot selection: choose vertex in P ∪ X with maximum degree in P
        pivot = None
        max_deg = -1
        union_px = list(P) + list(X)
        for v in union_px:
            deg = sum(1 for u in P if adj[v][u])
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        # Candidates: vertices in P not adjacent to pivot
        if pivot is not None:
            candidates_list = [v for v in P if not adj[pivot][v]]
        else:
            candidates_list = list(P)
        
        for v in candidates_list:
            new_R = R + [v]
            new_P = set(u for u in P if adj[v][u])
            new_X = set(u for u in X if adj[v][u])
            bron_kerbosch(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)
    
    # Run the search
    P = set(range(n))
    X = set()
    bron_kerbosch([], P, X)
    
    # best_clique is a list of indices into candidates
    # Convert to actual values
    if best_size > 85:
        result = [candidates[i] for i in best_clique]
    else:
        result = baseline
    
    # Ensure 255 is in the result (it should be, but just in case)
    if 255 not in result:
        result.append(255)
    
    return result
