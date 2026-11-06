"""Combined program: structural insight from inspirations + Bron-Kerbosch refinement."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    N = 512
    
    # Precompute LCM compatibility using bitsets
    # adj[i] is a bitmask where bit j is set if lcm(i+1, j+1) <= N
    # Using 0-indexed: index i represents number i+1
    adj = [0] * N
    
    # Precompute for all pairs
    for i in range(1, N + 1):
        ui = i - 1
        for j in range(i, N + 1):
            uj = j - 1
            g = gcd(i, j)
            lcm_val = i * j // g
            if lcm_val <= N:
                adj[ui] |= (1 << uj)
                adj[uj] |= (1 << ui)
    
    # Track best clique found
    best_clique = []
    
    def update_best(R_list):
        nonlocal best_clique
        if len(R_list) > len(best_clique):
            best_clique = R_list[:]
    
    # Bron-Kerbosch with Pivoting
    # P: bitmask of candidate vertices
    # R: list of vertices in current clique
    # X: bitmask of excluded vertices (for correctness, but we can omit for speed)
    
    def bron_kerbosch(P, R, X=0):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
        
        # Update best if R is larger
        if len(R) > len(best_clique):
            best_clique = R[:]
        
        # If P is empty, return
        if P == 0:
            return
        
        # Choose pivot u from P | X that maximizes |P & adj[u]|
        union = P | X
        best_u = -1
        best_count = -1
        temp = union
        while temp:
            low = temp & (-temp)
            u = low.bit_length() - 1
            count = (P & adj[u]).bit_count()
            if count > best_count:
                best_count = count
                best_u = u
            temp ^= low
        
        if best_u == -1:
            return
        
        # Candidates: P \ N(best_u)
        candidates = P & ~adj[best_u]
        
        while candidates:
            low = candidates & (-candidates)
            v = low.bit_length() - 1
            candidates ^= low
            
            # Recurse: R + v, P ∩ N(v), X ∩ N(v)
            bron_kerbosch(P & adj[v], R + [v], X & adj[v])
            
            # Remove v from P, add to X
            P &= ~low
            X |= low
            
            if time.time() > deadline:
                return
    
    # Run Bron-Kerbosch starting with all vertices
    all_mask = (1 << N) - 1
    bron_kerbosch(all_mask, [])
    
    # If Bron-Kerbosch didn't find anything (shouldn't happen), use greedy fallback
    if not best_clique:
        # Greedy: add vertices in order of decreasing degree
        order = list(range(N))
        order.sort(key=lambda x: adj[x].bit_count(), reverse=True)
        clique_mask = 0
        for v in order:
            if (adj[v] & clique_mask) == clique_mask:
                clique_mask |= (1 << v)
        best_clique = [i for i in range(N) if (clique_mask >> i) & 1]
    
    # Local improvement: try to add any vertex to the best clique
    if best_clique:
        clique_mask = 0
        for v in best_clique:
            clique_mask |= (1 << v)
        
        changed = True
        while changed:
            changed = False
            for v in range(N):
                if not (clique_mask >> v) & 1:
                    if (adj[v] & clique_mask) == clique_mask:
                        clique_mask |= (1 << v)
                        changed = True
        
        if clique_mask.bit_count() > len(best_clique):
            best_clique = [i for i in range(N) if (clique_mask >> i) & 1]
    
    # Convert 0-indexed to 1-indexed
    result = sorted([x + 1 for x in best_clique])
    
    # Ensure non-empty
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
