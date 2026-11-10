import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Vertices: numbers in [1, 255] that are divisible by at least one of {3, 5, 17}
    # (i.e., gcd(x, 255) > 1)
    vertices = [x for x in range(1, N + 1) if math.gcd(x, N) > 1]
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    n_v = len(vertices)
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask where bit j is set if gcd(vertices[i], vertices[j]) > 1
    adj = [0] * n_v
    for i in range(n_v):
        vi = vertices[i]
        mask = 0
        for j in range(n_v):
            if i != j and math.gcd(vi, vertices[j]) > 1:
                mask |= (1 << j)
        adj[i] = mask
    
    # 255 must be in the set
    idx_255 = v_to_idx[255]
    
    # Helper: iter bits of a mask
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb
    
    # Helper: count bits
    def popcount(x):
        return bin(x).count('1')
    
    # Bron-Kerbosch with Pivoting for Maximum Clique
    # We want a clique that includes idx_255.
    # We can run BK on the neighborhood of idx_255 (plus idx_255 itself).
    # Let P be the set of vertices compatible with idx_255 (excluding idx_255 itself).
    # Any clique in P plus {idx_255} is a valid clique containing idx_255.
    # Note: idx_255 is connected to all vertices in vertices by construction?
    # No, vertices are defined as gcd(x, 255) > 1. So gcd(255, x) > 1 for all x in vertices.
    # So idx_255 is connected to all other vertices in `vertices`.
    # Therefore, the problem reduces to finding the maximum clique in the subgraph induced by `vertices`.
    # Since idx_255 is universal in this graph (connected to everyone), it will always be part of any maximum clique?
    # Not necessarily the *only* maximum clique, but any maximum clique C can be extended to C U {255} if 255 is not in C.
    # Since we *require* 255, and 255 is connected to everyone, the max clique containing 255 is exactly 1 + (max clique in V \ {255}).
    # Actually, if C is a max clique in V, and 255 notin C, then C U {255} is a clique.
    # So max clique size containing 255 is max_clique_size(V).
    
    max_clique_size = 0
    max_clique_set = []
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique_size, max_clique_set
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                max_clique_set = R[:]
            return
        
        # Pivot selection: choose u in P union X with max degree in P
        # Standard pivot: u in P \cup X that maximizes |P \cap N(u)|
        
        PX = P | X
        if not PX:
            # Should be handled by P==0 and X==0 check above, but just in case
            if not P:
                 if len(R) > max_clique_size:
                     max_clique_size = len(R)
                     max_clique_set = R[:]
            return

        best_u = -1
        max_deg = -1
        
        # Iterate through PX to find best pivot
        for u in iter_bits(PX):
            # Degree of u in P
            deg = popcount(P & adj[u])
            if deg > max_deg:
                max_deg = deg
                best_u = u
        
        if best_u == -1:
            return

        # Recurse on P \ N(u)
        # The elements in P that are NOT neighbors of u
        # P_not_N_u = P & ~adj[best_u]
        
        P_not_N_u = P & ~adj[best_u]
        
        for v in iter_bits(P_not_N_u):
            # R' = R U {v}
            R.append(vertices[v])
            # P' = P ∩ N(v)
            # X' = X ∩ N(v)
            
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)
            
            if time.time() > deadline:
                return

    # Initial call
    # R = []
    # P = all vertices (mask with n_v bits set)
    # X = []
    
    # Optimization: We can start with a greedy solution to set a lower bound,
    # and potentially prune branches that can't beat it.
    # But standard BK with pivoting is usually fast enough for n~127.
    
    # Let's add a simple greedy initialization to set max_clique_size
    # This helps if we add pruning later, but BK doesn't strictly need it.
    # However, having a good initial bound might help if we implement "if len(R) + popcount(P) <= max_clique_size: return"
    
    # Greedy heuristic:
    # Start with empty set.
    # Repeatedly pick vertex with highest degree in current candidate set P.
    
    P_greedy = (1 << n_v) - 1
    R_greedy = []
    
    while P_greedy:
        # Pick v in P_greedy with max degree in P_greedy
        best_v = -1
        max_deg = -1
        for v in iter_bits(P_greedy):
            deg = popcount(P_greedy & adj[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        if best_v == -1:
            break
            
        R_greedy.append(vertices[best_v])
        # Update P_greedy to neighbors of best_v
        P_greedy &= adj[best_v]
        
    if len(R_greedy) > max_clique_size:
        max_clique_size = len(R_greedy)
        max_clique_set = R_greedy[:]
        
    # Now run BK to find the true maximum
    # We can add a pruning condition: if len(R) + popcount(P) <= max_clique_size, return
    # Let's wrap the call to include this logic inside the recursion or just rely on BK.
    # Standard BK doesn't have this pruning unless we modify it.
    # Let's modify bron_kerbosch to include this pruning.
    
    def bron_kerbosch_pruned(R, P, X):
        nonlocal max_clique_size, max_clique_set
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                max_clique_set = R[:]
            return
        
        # Pruning: if we can't beat current best, return
        if len(R) + popcount(P) <= max_clique_size:
            return
            
        # Pivot selection
        PX = P | X
        if not PX:
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
            R.append(vertices[v])
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bron_kerbosch_pruned(R, P_new, X_new)
            
            R.pop()
            
            P &= ~(1 << v)
            X |= (1 << v)
            
            if time.time() > deadline:
                return

    initial_P = (1 << n_v) - 1
    bron_kerbosch_pruned([], initial_P, 0)
    
    # Ensure 255 is in the result
    # The max_clique_set found is a set of values.
    # Since 255 is connected to everyone, it should be possible to include it.
    # If 255 is not in max_clique_set, we can add it (as long as it's not already there).
    # Wait, if max_clique_set is a max clique in the whole graph, and 255 is universal,
    # then max_clique_set U {255} is a larger clique unless 255 is already in max_clique_set.
    # So the true max clique size is max(max_clique_size, max_clique_size + 1 if 255 notin set).
    # Actually, if 255 is universal, any clique C not containing 255 can be extended to C U {255}.
    # So the maximum clique MUST contain 255 (unless the graph is empty, which it isn't).
    # So max_clique_set should contain 255?
    # Not necessarily, BK finds *a* maximum clique. It might find one without 255.
    # But since we can always add 255, the size we found is correct, but the set might not contain 255.
    # We need to return a set containing 255.
    # So if 255 not in max_clique_set, we add it. The size increases by 1.
    # Wait, if max_clique_set is a max clique of size K, and 255 notin it, then max_clique_set U {255} is a clique of size K+1.
    # This contradicts K being the maximum size.
    # Therefore, if 255 is universal, any maximum clique MUST contain 255?
    # Yes. If C is a max clique and 255 notin C, then C U {255} is a larger clique, contradiction.
    # So max_clique_set MUST contain 255.
    
    # However, due to implementation details or if the graph isn't perfectly universal (it is, by construction),
    # let's just check.
    
    result = max_clique_set[:]
    if 255 not in result:
        result.append(255)
        
    # Fallback if something went wrong (e.g. timeout before finding anything, though greedy should find something)
    if not result:
        result = [255]
        
    return result
