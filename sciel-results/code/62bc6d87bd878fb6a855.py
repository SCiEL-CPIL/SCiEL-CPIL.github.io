import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 512
    
    # Precompute compatibility bitmask for each vertex
    # compat_mask[i] for vertex i (1-indexed), bit (j-1) set if lcm(i,j) <= N
    compat_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            lcm_val = i * j // g
            if lcm_val <= N:
                mask |= (1 << (j - 1))
        compat_mask[i] = mask
    
    def time_up():
        return time.time() > deadline
    
    # Greedy construction: build a clique by adding vertices that are compatible with all current members
    # Strategy: sort vertices by degree (descending), then greedily add if compatible with all in clique
    
    # Compute degrees
    degrees = [bin(compat_mask[i]).count('1') for i in range(1, N + 1)]
    
    # Sort vertices by degree descending
    sorted_vertices = sorted(range(1, N + 1), key=lambda v: -degrees[v])
    
    # Greedy: add vertex if compatible with all current clique members
    greedy_clique = []
    for v in sorted_vertices:
        if time_up():
            break
        # Check if v is compatible with all in greedy_clique
        compatible = True
        for u in greedy_clique:
            if not (compat_mask[v] & (1 << (u - 1))):
                compatible = False
                break
        if compatible:
            greedy_clique.append(v)
    
    best_clique = list(greedy_clique)
    
    # Try local search: try to swap out one vertex for one or more others
    # Or: try multiple greedy orders and keep the best
    
    # Try different greedy strategies
    strategies = [
        sorted_vertices,
        sorted(range(1, N + 1), key=lambda v: -degrees[v] / v),  # degree/value ratio
        list(range(1, N + 1)),  # natural order
        list(range(N, 0, -1)),  # reverse natural
    ]
    
    for strategy in strategies:
        if time_up():
            break
        clique = []
        for v in strategy:
            if time_up():
                break
            compatible = True
            for u in clique:
                if not (compat_mask[v] & (1 << (u - 1))):
                    compatible = False
                    break
            if compatible:
                clique.append(v)
        if len(clique) > len(best_clique):
            best_clique = list(clique)
    
    # Try Bron-Kerbosch on a reduced candidate set if time permits
    if not time_up():
        # Try to find a larger clique using Bron-Kerbosch
        # Start from the best greedy clique and try to extend/improve
        
        # Alternative: run Bron-Kerbosch on the full graph with a good pivot
        # But this might be slow. Let's try a limited version.
        
        best_clique_mask = 0
        for v in best_clique:
            best_clique_mask |= (1 << (v - 1))
        
        # Try to find a maximal clique containing or near best_clique
        # Actually, let's just run a simple Bron-Kerbosch with a time limit
        
        bk_best = list(best_clique)
        
        def bron_kerbosch(R_mask, P_mask, X_mask):
            nonlocal bk_best
            
            if time_up():
                return
            
            if P_mask == 0 and X_mask == 0:
                if R_mask > 0:
                    clique = []
                    m = R_mask
                    while m:
                        lb = m & (-m)
                        idx = lb.bit_length() - 1
                        clique.append(idx + 1)
                        m ^= lb
                    if len(clique) > len(bk_best):
                        bk_best = clique
                return
            
            U_mask = P_mask | X_mask
            if U_mask == 0:
                return
            
            # Pick pivot: vertex in U_mask with max neighbors in P_mask
            # For efficiency, sample a few or use a heuristic
            # Let's pick the vertex in U_mask with the highest degree in P_mask
            # To keep it fast, just pick the first few and choose best
            pivot_v = None
            max_deg = -1
            # Sample up to 20 vertices from U_mask
            sample_mask = U_mask
            count = 0
            m = sample_mask
            while m and count < 20:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                v = idx + 1
                deg = bin(compat_mask[v] & P_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    pivot_v = v
                m ^= lb
                count += 1
            
            if pivot_v is None:
                return
            
            candidates = P_mask & ~compat_mask[pivot_v]
            C_mask = candidates if candidates != 0 else P_mask
            
            m = C_mask
            while m:
                if time_up():
                    return
                lb = m & (-m)
                idx = lb.bit_length() - 1
                v = idx + 1
                m ^= lb
                
                new_R = R_mask | (1 << idx)
                new_P = P_mask & compat_mask[v]
                new_X = X_mask & compat_mask[v]
                
                bron_kerbosch(new_R, new_P, new_X)
                
                P_mask ^= (1 << idx)
                X_mask |= (1 << idx)
        
        all_mask = (1 << N) - 1
        bron_kerbosch(0, all_mask, 0)
        
        if len(bk_best) > len(best_clique):
            best_clique = bk_best
    
    if not best_clique:
        from math import isqrt
        return list(range(1, isqrt(N) + 1))
    
    return sorted(best_clique)
