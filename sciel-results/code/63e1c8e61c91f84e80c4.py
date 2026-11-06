"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [1, 512] with lcm(a,b) <= 512 for every pair.
    Maximize cardinality.
    
    Strategy:
    1. Build compatibility matrix using bitsets.
    2. Find best divisor set as initial seed.
    3. Run Bron-Kerbosch with pivot on the compatibility graph.
    4. Return the maximum clique found.
    """
    import time
    from math import gcd
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    N = 512
    
    # 1. Build compatibility matrix as bitsets
    # compat[i] is a bitset (Python int) where bit j is set if lcm(i,j) <= N
    compat = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            g = gcd(a, b)
            if a * b // g <= N:
                mask |= (1 << b)
        compat[a] = mask
    
    # 2. Find best divisor set as seed
    best_seed = set()
    best_seed_size = 0
    
    for M in range(1, N + 1):
        divs = []
        for i in range(1, int(M**0.5) + 1):
            if M % i == 0:
                divs.append(i)
                if i != M // i:
                    divs.append(M // i)
        
        if len(divs) > best_seed_size:
            best_seed_size = len(divs)
            best_seed = set(divs)
    
    # 3. Bron-Kerbosch with pivot for maximum clique
    # We'll work with the induced subgraph on candidates that could be in a large clique.
    # First, let's identify promising candidates: those with high degree.
    
    # Degree of each vertex in the compatibility graph
    degrees = [bin(compat[i]).count('1') for i in range(N + 1)]
    
    # Sort candidates by degree (descending) for better branching
    candidates = list(range(1, N + 1))
    candidates.sort(key=lambda x: -degrees[x])
    
    # Map from original index to position in sorted order
    pos_map = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency as sets for the Bron-Kerbosch algorithm
    # adj[v] = set of candidates u such that u is compatible with v and u comes after v in our ordering
    # Actually, for Bron-Kerbosch, we need the full neighborhood.
    # Let's use a different approach: represent the graph with bitsets over the candidate list.
    
    num_cands = len(candidates)
    # adj_bits[i] = bitset of neighbors of candidates[i] among candidates[0..num_cands-1]
    adj_bits = [0] * num_cands
    for i, v in enumerate(candidates):
        mask = 0
        for j, u in enumerate(candidates):
            if i != j and (compat[v] >> u) & 1:
                mask |= (1 << j)
        adj_bits[i] = mask
    
    best_clique = []
    best_clique_size = 0
    
    def bron_kerbosch(R, P, X):
        """
        R: current clique (list of indices into candidates)
        P: set of candidates that can still be added (bitset)
        X: set of candidates already processed (bitset)
        """
        nonlocal best_clique, best_clique_size, deadline
        
        if time.time() > deadline:
            return
        
        if P == 0 and X == 0:
            # Found a maximal clique
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique = list(R)
            return
        
        if P == 0:
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        PX = P | X
        # Pick u with maximum degree in PX
        best_u = -1
        best_deg = -1
        temp = PX
        while temp:
            # Extract lowest set bit
            lsb = temp & (-temp)
            u_idx = lsb.bit_length() - 1
            deg = bin(adj_bits[u_idx] & PX).count('1')
            if deg > best_deg:
                best_deg = deg
                best_u = u_idx
            temp ^= lsb
        
        if best_u == -1:
            return
        
        # Branch on vertices in P \ N(u)
        candidates_to_branch = P & ~(adj_bits[best_u] | (1 << best_u))
        
        temp = candidates_to_branch
        while temp:
            lsb = temp & (-temp)
            v_idx = lsb.bit_length() - 1
            temp ^= lsb
            
            R.append(v_idx)
            bron_kerbosch(R, P & adj_bits[v_idx], X & adj_bits[v_idx])
            R.pop()
            
            P ^= (1 << v_idx)
            X |= (1 << v_idx)
            
            if time.time() > deadline:
                return
    
    # Initial P: all candidates
    initial_P = (1 << num_cands) - 1
    
    # Use the seed to warm-start: we know the seed is a valid clique.
    # Let's first verify and use it as a lower bound.
    seed_indices = [pos_map[v] for v in best_seed if v in pos_map]
    best_clique = seed_indices
    best_clique_size = len(seed_indices)
    
    # Run Bron-Kerbosch
    bron_kerbosch([], initial_P, 0)
    
    # 4. Convert back to original values
    result = [candidates[i] for i in best_clique]
    
    # Verify the result
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            g = gcd(a, b)
            if a * b // g > N:
                # Invalid, fall back to seed
                result = list(best_seed)
                break
        else:
            continue
        break
    
    return result
# EVOLVE-BLOCK-END
