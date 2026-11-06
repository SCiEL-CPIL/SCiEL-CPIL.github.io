import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # Candidates must share a factor with 255: divisible by 3, 5, or 17
    # Also must form a clique: every pair has gcd > 1
    
    # Build candidate list (must include 255)
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Ensure 255 is in candidates
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    
    # Build adjacency: adj[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    # For efficiency, use bitmasks or precomputed gcd
    # n is at most ~110 (multiples of 3: 85, plus some of 5 and 17 not already counted)
    # Actually: |multiples of 3| = 85, |multiples of 5| = 51, |multiples of 17| = 15
    # |union| = 85 + 51 + 15 - |3&5| - |3&17| - |5&17| + |3&5&17|
    # |3&5| = 17, |3&17| = 5, |5&17| = 3, |3&5&17| = 1
    # |union| = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127
    
    # Build adjacency matrix using bitsets for speed
    # adj[i] is a set of indices j != i such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find index of 255
    idx_255 = candidates.index(255)
    
    # We need max clique containing idx_255
    # Use Bron-Kerbosch with pivot and coloring heuristic
    # Since 255 must be in the clique, we can restrict to its neighbors
    
    best_clique = [idx_255]
    
    # Bron-Kerbosch with pivot
    def bron_kerbosch(R, P, X, depth=0):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
        
        # Update best
        if len(R) > len(best_clique):
            best_clique = list(R)
        
        # If P is empty and X is empty, we have a maximal clique
        if not P and not X:
            return
        
        # Choose pivot u from P ∪ X to minimize |P ∩ N(u)|
        # For efficiency, pick u that maximizes |P ∩ N(u)|
        if P or X:
            # Pick pivot from P ∪ X
            pivot_set = P | X
            # Choose u with max |P ∩ adj[u]|
            best_u = -1
            best_count = -1
            for u in pivot_set:
                count = len(P & adj[u])
                if count > best_count:
                    best_count = count
                    best_u = u
            # Only iterate over P \ N(u)
            candidates_iter = P - adj[best_u]
        else:
            candidates_iter = P
        
        for v in list(candidates_iter):
            if time.time() > deadline:
                return
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X, depth + 1)
            P = P - {v}
            X = X | {v}
    
    # Start with R = {255}, P = neighbors of 255, X = empty
    # But we can do better: only consider neighbors of 255 as potential additions
    P = adj[idx_255]
    X = set()
    R = {idx_255}
    
    bron_kerbosch(R, P, X)
    
    # Convert back to values
    result = [candidates[i] for i in best_clique]
    result.sort()
    
    return result
