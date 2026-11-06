import math

# EVOLVE-BLOCK-START
def construct():
    # 1. Identify candidates
    # Candidates must be in [1, 255] and share a factor with 255 (3, 5, 17).
    # Also must contain 255.
    # We want max clique containing 255.
    
    # Generate candidates
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    
    # Ensure 255 is in candidates (it is, gcd(255,255)=255)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    n = len(candidates)
    
    # Index of 255
    idx_255 = val_to_idx[255]
    
    # 2. Build adjacency bitmasks
    # adj[i] is an integer bitmask where bit j is 1 if candidates[i] and candidates[j] are connected (gcd > 1)
    adj = [0] * n
    
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 3. Bron-Kerbosch with pivot
    # We start with R = {255}
    # P = neighbors of 255 in candidates \ {255}
    # X = empty
    
    # P mask: all bits except idx_255 that are neighbors of idx_255
    # Since all candidates share a factor with 255, are they all neighbors of 255?
    # gcd(255, c) > 1 is true for all c in candidates by definition.
    # So P is all candidates except 255.
    
    all_mask = (1 << n) - 1
    P = all_mask & ~(1 << idx_255)
    X = 0
    
    best_size = 1
    best_clique = [255]
    
    # current_clique is a list of values
    # We use recursion with backtracking
    
    def bronk(R_size, P_mask, X_mask, current_clique):
        nonlocal best_size, best_clique
        
        if P_mask == 0 and X_mask == 0:
            if R_size > best_size:
                best_size = R_size
                best_clique = current_clique.copy()
            return
        
        if P_mask == 0:
            return
        
        # Pivot selection
        # Choose u in P U X that maximizes |P intersect N(u)|
        # To speed up, we can just pick any u from P U X? 
        # Max degree is better for pruning.
        
        PX = P_mask | X_mask
        if PX == 0:
            return
            
        # Find best pivot
        best_u = -1
        max_count = -1
        
        # Iterate through bits in PX
        # Optimization: If we just pick the first one, it's faster but less pruning.
        # Let's try to find a good pivot.
        # For n ~ 120, iterating all bits is okay.
        
        temp = PX
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp -= lsb
            
            # Count neighbors of u in P
            # N(u) is adj[u]
            count = (P_mask & adj[u]).bit_count() if hasattr(int, 'bit_count') else bin(P_mask & adj[u]).count('1')
            
            if count > max_count:
                max_count = count
                best_u = u
                
        if best_u == -1:
            return
            
        # Iterate over v in P \ N(best_u)
        # This is P & ~adj[best_u]
        expansion_set = P_mask & ~adj[best_u]
        
        temp = expansion_set
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp -= lsb
            
            # Add v to clique
            current_clique.append(candidates[v])
            
            # Recurse
            new_P = P_mask & adj[v]
            new_X = X_mask & adj[v]
            
            bronk(R_size + 1, new_P, new_X, current_clique)
            
            # Backtrack
            current_clique.pop()
            
            # Update P and X
            P_mask = P_mask & ~lsb
            X_mask = X_mask | lsb
            
    # Initial call
    # R contains 255
    bronk(1, P, X, [255])
    
    return best_clique

# EVOLVE-BLOCK-END
