"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 512] such that lcm(a, b) <= 512 for every pair (a, b).
    
    Approach:
    1. Model the problem as finding a Maximum Independent Set (MIS) in a conflict graph.
       Node i (representing number i+1) conflicts with node j if lcm(i+1, j+1) > 512.
    2. The MIS in the conflict graph is equivalent to a Maximum Clique in the complement graph.
       The complement graph has an edge between i and j if lcm(i+1, j+1) <= 512.
    3. Use the Bron-Kerbosch algorithm with pivoting to find the Maximum Clique.
       We use Python integers as bitsets for efficient set operations.
    """
    from math import gcd
    import sys
    
    # Increase recursion limit for safety
    sys.setrecursionlimit(10000)
    
    N = 512
    # compat_mask[i] is a bitmask where bit j is 1 if number (i+1) and (j+1) are compatible (lcm <= 512)
    # We exclude self-loops (i != j)
    
    compat_mask = [0] * N
    
    # Precompute compatibility
    for a in range(1, N + 1):
        i = a - 1
        mask = 0
        for b in range(1, N + 1):
            if a == b:
                continue
            g = gcd(a, b)
            if a * b // g <= N:
                j = b - 1
                mask |= (1 << j)
        compat_mask[i] = mask
        
    # Global variables for the solver
    best_size = 0
    best_set = 0
    
    # Bron-Kerbosch with Pivoting
    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_set
        
        # Pruning
        if R.bit_count() + P.bit_count() <= best_size:
            return
            
        if P == 0:
            if X == 0 and R.bit_count() > best_size:
                best_size = R.bit_count()
                best_set = R
            return
            
        # Choose pivot u from P | X that maximizes |P & adj[u]|
        PX = P | X
        if PX == 0:
            return
            
        max_inter = -1
        pivot = -1
        px = PX
        while px:
            lsb = px & -px
            u = lsb.bit_length() - 1
            px ^= lsb
            
            inter = P & compat_mask[u]
            cnt = inter.bit_count()
            if cnt > max_inter:
                max_inter = cnt
                pivot = u
                if cnt == P.bit_count():
                    break
                    
        if pivot == -1:
            pivot = (PX & -PX).bit_length() - 1

        # The loop runs over v in P \ adj[pivot]
        candidates = P & ~compat_mask[pivot]
        
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            bron_kerbosch(R | lsb, P & compat_mask[v], X & compat_mask[v])
            
            P ^= lsb
            X |= lsb
            
    # Initialize
    initial_P = (1 << N) - 1
    bron_kerbosch(0, initial_P, 0)
    
    # Extract result
    result = []
    s = best_set
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        s ^= lsb
        
    return result

# EVOLVE-BLOCK-END
