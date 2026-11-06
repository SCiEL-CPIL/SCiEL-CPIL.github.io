import math
import time

def construct():
    """
    Construct a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # 255 = 3 * 5 * 17
    # Every element must share a factor with 255 to have gcd > 1 with 255.
    # Candidates are numbers in [1, 255] divisible by 3, 5, or 17.
    
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    n = len(candidates)
    
    # Build adjacency list: adj[i] is a set of j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find index of 255
    idx_255 = candidates.index(255)
    
    # We must include 255 in the clique.
    # So we start with R = {idx_255}, P = adj[idx_255], X = empty.
    
    best_clique = {idx_255}
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = R.copy()
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        U = P | X
        if not U:
            if len(R) > len(best_clique):
                best_clique = R.copy()
            return
            
        # Pick the vertex with maximum degree in P (standard BK with pivot)
        pivot = max(U, key=lambda u: len(adj[u] & P))
        
        # For each v in P \ N(pivot)
        candidates_for_recursion = list(P - adj[pivot])
        
        for v in candidates_for_recursion:
            if time.time() > deadline:
                return
            
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)
    
    # Initialize with 255 forced in
    R = {idx_255}
    P = adj[idx_255].copy()
    X = set()
    
    bron_kerbosch(R, P, X)
    
    # Convert back to values
    result_set = set(candidates[i] for i in best_clique)
    
    # Ensure 255 is in the result
    if 255 not in result_set:
        result_set.add(255)
    
    res_list = list(result_set)
    
    # Double-check validity
    if not res_list:
        return [255]
    
    valid = True
    for i in range(len(res_list)):
        for j in range(i + 1, len(res_list)):
            if math.gcd(res_list[i], res_list[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid or 255 not in res_list:
        # Fallback to all multiples of 3 (known valid clique of size 85)
        return list(range(3, 256, 3))
    
    return res_list
