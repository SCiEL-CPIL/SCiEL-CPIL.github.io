"""Program to find a large pairwise non-coprime subset of [1,255] containing 255."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    Strategy: Use Bron-Kerbosch algorithm to find the maximum clique in the 
    graph of candidates sharing a factor with 255.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin for 90s limit
    
    n = 255
    # Candidates must share a factor with 255 (3, 5, 17)
    candidates = [x for x in range(1, n+1) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    
    # We must include 255.
    # Any other element x in the set must satisfy gcd(x, 255) > 1, which is true for all candidates.
    # Also, for any x, y in the set (x != y), gcd(x, y) > 1.
    # Since 255 is in the set, all other elements must be connected to 255.
    # All candidates are connected to 255.
    # So we need to find a maximum clique in the subgraph induced by candidates \ {255},
    # and then add 255 to it.
    
    # Remove 255 from candidates for the clique search on the rest
    if 255 in candidates:
        candidates.remove(255)
        
    N = len(candidates)
    if N == 0:
        return [255]
        
    # Map candidates to indices 0..N-1
    # Precompute adjacency bitmasks
    # adj[i] is a bitmask of j such that gcd(candidates[i], candidates[j]) > 1
    
    adj = [0] * N
    for i in range(N):
        ci = candidates[i]
        mask = 0
        for j in range(N):
            if i != j:
                cj = candidates[j]
                if math.gcd(ci, cj) > 1:
                    mask |= (1 << j)
        adj[i] = mask
        
    # Bron-Kerbosch with pivot
    # We want to find the maximum clique.
    # R: current clique
    # P: candidates to add to R (bitmask)
    # X: candidates already processed (bitmask)
    
    max_size = 0
    best_clique_mask = 0
    
    # Optimization: If time runs out, we might not find the exact max.
    # But for N=126, it should be fast.
    
    def bron_kerbosch(R_size, P, X):
        nonlocal max_size, best_clique_mask, deadline
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            # Found a maximal clique
            if R_size > max_size:
                max_size = R_size
                best_clique_mask = P # Wait, P is 0 here. We need to track R.
                # Actually, standard BK doesn't track R as a bitmask in this signature easily 
                # unless we pass it or reconstruct.
                # Let's modify to pass R_mask or just store size and reconstruct later?
                # Reconstructing is hard. Let's pass R_mask.
            return
            
        # Pivot selection
        # Choose u from P union X to maximize |P & adj[u]|
        union = P | X
        if union == 0:
            return
            
        # Find pivot
        pivot = -1
        max_neighbors = -1
        # Iterate through bits in union
        temp = union
        while temp:
            # Get least significant bit
            lsb = temp & -temp
            idx = (lsb.bit_length() - 1)
            # Count neighbors in P
            neigh_count = bin(P & adj[idx]).count('1')
            if neigh_count > max_neighbors:
                max_neighbors = neigh_count
                pivot = idx
            temp ^= lsb
            
        # Candidates to try: P minus neighbors of pivot
        # If pivot is -1, try all P
        if pivot == -1:
            candidates_to_try = P
        else:
            candidates_to_try = P & ~adj[pivot]
            
        # Iterate through candidates_to_try
        temp = candidates_to_try
        while temp:
            lsb = temp & -temp
            v = (lsb.bit_length() - 1)
            
            # Recurse
            bron_kerbosch(R_size + 1, P & adj[v], X & adj[v])
            
            # Remove v from P, add to X
            P ^= lsb
            X |= lsb
            temp ^= lsb
            
            if time.time() > deadline:
                return

    # Standard BK needs R to be tracked for extraction.
    # Let's use a slightly different approach:
    # Store the best clique mask found.
    
    # Re-implementing with R_mask to allow extraction
    max_size = 0
    best_R_mask = 0
    
    def bk(R_size, R_mask, P, X):
        nonlocal max_size, best_R_mask, deadline
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if R_size > max_size:
                max_size = R_size
                best_R_mask = R_mask
            return
            
        # Upper bound pruning
        if R_size + bin(P).count('1') <= max_size:
            return
            
        union = P | X
        if union == 0:
            return
            
        # Pivot
        pivot = -1
        max_neigh = -1
        temp = union
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cnt = bin(P & adj[idx]).count('1')
            if cnt > max_neigh:
                max_neigh = cnt
                pivot = idx
            temp ^= lsb
            
        if pivot == -1:
            to_try = P
        else:
            to_try = P & ~adj[pivot]
            
        temp = to_try
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            
            bk(R_size + 1, R_mask | lsb, P & adj[v], X & adj[v])
            
            P ^= lsb
            X |= lsb
            temp ^= lsb
            
            if time.time() > deadline:
                return

    # Initial call
    # P = all nodes, X = 0, R = 0
    bk(0, 0, (1 << N) - 1, 0)
    
    # Extract best clique
    result = [255]
    if max_size > 0:
        for i in range(N):
            if best_R_mask & (1 << i):
                result.append(candidates[i])
                
    return result
# EVOLVE-BLOCK-END
