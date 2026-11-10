import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0  # Safety margin
    
    # 1. Generate candidates U = {x in [1, 254] | gcd(x, 255) > 1}
    candidates = []
    for x in range(1, N):
        if math.gcd(x, N) > 1:
            candidates.append(x)
            
    n_cands = len(candidates)
    
    if n_cands == 0:
        return [N]

    # 2. Build adjacency bitsets
    # adj[i] is a bitmask of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n_cands
    for i in range(n_cands):
        vi = candidates[i]
        mask = 0
        for j in range(i + 1, n_cands):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                mask |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] = mask

    # 3. Find Maximum Clique using Bron-Kerbosch with Pivoting
    max_clique_size = 0
    max_clique_set = []
    
    def popcount(x):
        return bin(x).count('1')
    
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb
            
    def bron_kerbosch(R, P, X):
        nonlocal max_clique_size, max_clique_set
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                max_clique_set = R[:]
            return
        
        # Pivot selection
        PX = P | X
        if PX == 0:
            return
        
        best_u = -1
        max_deg = -1
        for u in iter_bits(PX):
            deg = popcount(P & adj[u])
            if deg > max_deg:
                max_deg = deg
                best_u = u
        
        if best_u == -1:
            return

        P_not_N_u = P & ~adj[best_u]
        
        for v in iter_bits(P_not_N_u):
            R.append(candidates[v])
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            P &= ~(1 << v)
            X |= (1 << v)

    initial_P = (1 << n_cands) - 1
    bron_kerbosch([], initial_P, 0)
    
    # Construct the final set
    if max_clique_set:
        result = max_clique_set + [N]
    else:
        # Fallback: Multiples of 3
        result = [x for x in range(1, N + 1) if x % 3 == 0]
        
    # Ensure N is in result
    if N not in result:
        result.append(N)
        
    return result
