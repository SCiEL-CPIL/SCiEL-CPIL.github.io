import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 512
    # adj[i] is a bitmask of neighbors of i (0-indexed)
    adj = [0] * N
    
    # Precompute compatibility
    for i in range(1, N + 1):
        ui = i - 1
        for j in range(i + 1, N + 1):
            uj = j - 1
            # Check lcm(i, j) <= N
            if (i * j) // gcd(i, j) <= N:
                adj[ui] |= (1 << uj)
                adj[uj] |= (1 << ui)
    
    # Maximum Clique via Bron-Kerbosch with Pivoting
    # P: set of candidates (bitmask)
    # R: current clique (list of indices)
    best_clique = []
    
    def bron_kerbosch(P, R):
        nonlocal best_clique
        if time.time() > deadline:
            return
            
        # If R is larger than best, update
        if len(R) > len(best_clique):
            best_clique = R[:]
            
        # If P is empty, return
        if P == 0:
            return
            
        # Pivoting: choose u in P maximizing |P & adj[u]|
        # Iterate through bits in P
        p_temp = P
        max_count = -1
        pivot = -1
        while p_temp:
            # Extract lowest set bit
            lsb = p_temp & (-p_temp)
            idx = lsb.bit_length() - 1
            # Count neighbors in P
            count = (P & adj[idx]).bit_count()
            if count > max_count:
                max_count = count
                pivot = idx
            p_temp ^= lsb
            
        # Candidates to iterate: P \ N(pivot)
        # N(pivot) is adj[pivot]
        candidates = P & ~adj[pivot]
        
        while candidates:
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            # Recurse
            bron_kerbosch(P & adj[v], R + [v])
            
            # Remove v from P
            P &= ~(1 << v)
            
            # Early termination check
            if time.time() > deadline:
                return

    # Initial P is all vertices
    initial_P = (1 << N) - 1
    bron_kerbosch(initial_P, [])
    
    # Convert indices back to values (1-based)
    # best_clique contains indices 0..511
    # Return sorted list of values
    result = sorted([x + 1 for x in best_clique])
    
    # Ensure non-empty
    if not result:
        return [1]
        
    return result
