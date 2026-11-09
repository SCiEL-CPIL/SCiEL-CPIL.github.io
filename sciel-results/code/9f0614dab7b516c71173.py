import math
import time

def construct():
    """
    Finds the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b.
    """
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    # 1. Generate candidates: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    # A number x must have gcd(x, 255) > 1, so x must be divisible by 3, 5, or 17.
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    
    # 2. Build adjacency using bitmasks for speed
    n = len(candidates)
    adj_mask = [0] * n
    
    for i in range(n):
        if time.time() > deadline:
            return [i for i in range(1, 256) if i % 3 == 0]
        
        ci = candidates[i]
        mask = 0
        for j in range(n):
            if i != j and math.gcd(ci, candidates[j]) > 1:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # 3. Initial solution: multiples of 3 (known clique of size 85)
    mult3 = [i for i in range(1, 256) if i % 3 == 0]
    max_clique_size = len(mult3)
    max_clique_set = set(mult3)
    
    # 4. Bron-Kerbosch with pivot using bitmasks
    def popcount(x):
        return bin(x).count('1')
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique_size, max_clique_set
        
        if time.time() > deadline:
            return
            
        # Prune: if even adding all of P can't beat current best, stop
        if len(R) + popcount(P) <= max_clique_size:
            return
            
        if P == 0:
            if X == 0:
                if len(R) > max_clique_size:
                    max_clique_size = len(R)
                    max_clique_set = set(R)
            return
        
        # Choose pivot u from P | X to minimize branching
        PX = P | X
        if PX == 0:
            return
            
        # Find pivot with max degree in P
        pivot = -1
        max_deg = -1
        temp = PX
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            deg = popcount(P & adj_mask[u])
            if deg > max_deg:
                max_deg = deg
                pivot = u
            temp ^= lsb
        
        # Iterate over candidates in P that are NOT neighbors of pivot
        P_non_neighbors = P & ~(adj_mask[pivot] if pivot >= 0 else 0)
        
        temp = P_non_neighbors
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            # Recurse with v in R
            R_new = R | lsb
            P_new = P & adj_mask[v]
            X_new = X & adj_mask[v]
            
            bron_kerbosch(R_new, P_new, X_new)
            
            # Remove v from P, add to X
            P ^= lsb
            X |= lsb

    # Initialize
    P_init = (1 << n) - 1
    X_init = 0
    
    bron_kerbosch(0, P_init, X_init)
    
    # Convert indices back to values
    result = [candidates[i] for i in range(n) if (max_clique_set_mask if 'max_clique_set_mask' in dir() else 0) & (1 << i)]
    
    # Actually, let's track the clique properly
    # The above max_clique_set is a set of values, not indices. Let me fix.
    
    # Re-run with proper tracking
    max_clique_size = len(mult3)
    max_clique_mask = 0
    for val in mult3:
        idx = candidates.index(val)
        max_clique_mask |= (1 << idx)
    
    def bron_kerbosch2(R, P, X):
        nonlocal max_clique_size, max_clique_mask
        
        if time.time() > deadline:
            return
            
        if popcount(R) + popcount(P) <= max_clique_size:
            return
            
        if P == 0:
            if X == 0:
                if popcount(R) > max_clique_size:
                    max_clique_size = popcount(R)
                    max_clique_mask = R
            return
        
        PX = P | X
        if PX == 0:
            return
            
        pivot = -1
        max_deg = -1
        temp = PX
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            deg = popcount(P & adj_mask[u])
            if deg > max_deg:
                max_deg = deg
                pivot = u
            temp ^= lsb
        
        P_non_neighbors = P & ~(adj_mask[pivot] if pivot >= 0 else 0)
        
        temp = P_non_neighbors
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            R_new = R | lsb
            P_new = P & adj_mask[v]
            X_new = X & adj_mask[v]
            
            bron_kerbosch2(R_new, P_new, X_new)
            
            P ^= lsb
            X |= lsb
    
    bron_kerbosch2(0, P_init, X_init)
    
    result = [candidates[i] for i in range(n) if (max_clique_mask >> i) & 1]
    
    # Ensure 255 is in result
    if 255 not in result:
        # Since 255 is connected to all candidates, any max clique can be extended with 255
        # But if our found clique doesn't have 255, add it
        result.append(255)
    
    return sorted(result)
