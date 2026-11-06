"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    N = 512
    
    # Precompute LCM compatibility
    # Two numbers a, b are compatible if lcm(a,b) <= 512
    # lcm(a,b) = a*b // gcd(a,b)
    
    # We'll build a graph where vertices are 1..512
    # Edge between i and j if lcm(i,j) <= 512
    # Note: every vertex is compatible with itself (lcm(a,a)=a <= 512)
    
    # For maximum clique, we need a set where ALL pairs are compatible.
    # This is the maximum clique problem on the compatibility graph.
    
    # Key insight: if a | b (a divides b), then lcm(a,b) = b.
    # So if b <= 512, then a and b are compatible.
    # More generally, lcm(a,b) <= 512 means a*b/gcd(a,b) <= 512.
    
    # Let's think about structure. If we pick numbers that all divide some common
    # number M <= 512, then lcm of any pair divides M, so lcm <= M <= 512.
    # So the set of all divisors of any M <= 512 is a valid clique.
    
    # To maximize size, we want M with many divisors.
    # But we can also mix: numbers don't all need to divide a single M.
    
    # Let's use Bron-Kerbosch with bitsets for maximum clique.
    
    # Build adjacency as bitmasks
    # Vertex i (1-indexed) corresponds to bit (i-1)
    n_vertices = N
    adj = [0] * (n_vertices + 1)  # 1-indexed
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            lcm_val = i * j // g
            if lcm_val <= N:
                mask |= (1 << (j - 1))
        adj[i] = mask
    
    # Greedy baseline: try multiple orderings
    def greedy_clique(order_key):
        current = []
        current_mask = 0
        remaining = list(range(1, N + 1))
        remaining.sort(key=order_key, reverse=True)
        for v in remaining:
            if (adj[v] & current_mask) == current_mask:
                current.append(v)
                current_mask |= (1 << (v - 1))
        return current, current_mask
    
    # Try ordering by degree (compatibility count)
    best_clique, best_mask = greedy_clique(lambda x: adj[x].bit_count())
    best_size = len(best_clique)
    
    # Try ordering by value (small numbers more compatible)
    c2, m2 = greedy_clique(lambda x: -x)
    if len(c2) > best_size:
        best_clique, best_mask, best_size = c2, m2, len(c2)
    
    # Bron-Kerbosch with pivot and bitsets
    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_clique, best_mask, best_size
        
        if time.time() > deadline - 2:
            return
        
        if P_mask == 0 and X_mask == 0:
            size = R_mask.bit_count()
            if size > best_size:
                best_size = size
                best_mask = R_mask
                best_clique = [i for i in range(1, N + 1) if (R_mask >> (i - 1)) & 1]
            return
        
        # Choose pivot u from P_mask | X_mask with maximum |P_mask & adj[u]|
        union_mask = P_mask | X_mask
        best_u = 0
        best_count = -1
        temp = union_mask
        while temp:
            low = temp & (-temp)
            u = low.bit_length()  # vertex index
            count = (P_mask & adj[u]).bit_count()
            if count > best_count:
                best_count = count
                best_u = u
            temp ^= low
        
        if best_u == 0:
            return
        
        # P_mask minus neighbors of best_u
        candidates = P_mask & ~adj[best_u]
        
        while candidates:
            low = candidates & (-candidates)
            v = low.bit_length()  # vertex index
            candidates ^= low
            P_mask_next = P_mask & adj[v] & ~low  # Remove v from P
            bron_kerbosch(R_mask | low, P_mask_next, X_mask & adj[v])
            P_mask = P_mask & ~low
            X_mask |= low
    
    # Run Bron-Kerbosch
    # Start with R=0, P=all vertices, X=0
    all_mask = (1 << N) - 1
    bron_kerbosch(0, all_mask, 0)
    
    # Local improvement: try to add vertices to the best clique
    if best_mask:
        changed = True
        while changed:
            changed = False
            for v in range(1, N + 1):
                if not (best_mask >> (v - 1)) & 1:
                    if (adj[v] & best_mask) == best_mask:
                        best_mask |= (1 << (v - 1))
                        best_size += 1
                        changed = True
    
    # Extract result
    result = [i for i in range(1, N + 1) if (best_mask >> (i - 1)) & 1]
    
    if not result:
        # Fallback
        result = list(range(1, int(N**0.5) + 1))
    
    return result
# EVOLVE-BLOCK-END
