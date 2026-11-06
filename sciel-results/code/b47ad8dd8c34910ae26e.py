import math

def construct():
    N = 512
    # Numbers are 1..512, indices 0..511
    
    # 1. Build adjacency bitmasks
    # adj[i] is a bitmask of j such that lcm(i+1, j+1) <= 512
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        # Optimization: only check j > i? No, bitmask needs full.
        # But we can fill symmetrically.
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if math.gcd(a, b) * (a // math.gcd(a, b)) * (b // math.gcd(a, b)) <= N:
                # Wait, a*b//gcd
                # To avoid overflow or slowness, check b // gcd(a,b) * a <= N ?
                # a * (b // gcd(a,b)) <= N
                if a * (b // math.gcd(a, b)) <= N:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i) # Fill symmetric part immediately? 
                    # Actually, doing it inside the loop for all j is O(N^2) which is 262k. Fine.
                    
    # Let's re-build cleanly to be safe and fast
    adj = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if a * (b // math.gcd(a, b)) <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Find initial lower bound
    # Divisors of k for k in 1..512
    max_divs = 0
    best_init_mask = 0
    for k in range(1, N + 1):
        count = 0
        mask = 0
        for d in range(1, int(math.isqrt(k)) + 1):
            if k % d == 0:
                count += 1
                mask |= (1 << (d - 1)) # d-1 index
                if d * d != k:
                    count += 1
                    mask |= (1 << (k // d - 1))
        if count > max_divs:
            max_divs = count
            best_init_mask = mask
            
    best_size = max_divs
    best_mask = best_init_mask
    
    # 3. Bron-Kerbosch with bitmasks
    # R, P, X are bitmasks
    
    def greedy_color_bound(P_mask):
        """
        Returns an upper bound on clique size in P_mask.
        Uses greedy coloring on the subgraph induced by P_mask.
        """
        if P_mask == 0:
            return 0
            
        # Extract vertices
        verts = []
        m = P_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        # Greedy coloring
        color = [0] * len(verts)
        max_c = 0
        # Order by degree in P? Or just sequential.
        # For better bound, sort by degree descending
        verts.sort(key=lambda v: (adj[v] & P_mask).bit_count(), reverse=True)
        
        # Map original index to color
        # We just need the number of colors used
        # A vertex v can have color c if no neighbor has color c
        # Simpler: assign smallest available color
        colors_used = [0] * len(verts) # color of verts[i]
        # Actually, we just need the max color number assigned
        
        # This is O(V^2) per call. With V<=512, it's okay.
        # But we can optimize: 
        # Standard greedy coloring for upper bound:
        # We assign colors 1, 2, ...
        # If we can color with k colors, clique size <= k.
        
        # Implementation:
        # For each vertex in order, assign the smallest color not used by neighbors.
        # This might be slow if done naively.
        
        # Alternative: Just return popcount(P) as a weak bound?
        # No, coloring is much stronger.
        
        # Let's implement a fast greedy coloring.
        # colors[v] = 0 initially.
        # For each v in verts:
        #   forbidden = set of colors of neighbors in verts that are already colored
        #   color[v] = smallest positive int not in forbidden
        #   max_c = max(max_c, color[v])
        
        # To make it faster, we can skip coloring if popcount(P) + current_R_size <= best_size
        # But we need the bound to decide pruning.
        
        # Optimization: If len(verts) <= best_size - current_R_size, we might not need tight bound?
        # No, we need to know if we can beat best.
        
        # Let's just do it.
        colors = {}
        for v in verts:
            used_colors = set()
            # Check neighbors in verts
            for u in verts:
                if u == v: continue
                if (adj[v] >> u) & 1:
                    if u in colors:
                        used_colors.add(colors[u])
            # Find smallest available
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_c:
                max_c = c
                
        return max_c

    # Recursive Bron-Kerbosch
    import sys
    sys.setrecursionlimit(10000)
    
    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_mask
        
        # Pruning: if R size + max possible from P <= best_size
        # Use coloring bound
        if P:
            bound = greedy_color_bound(P)
            if R.bit_count() + bound <= best_size:
                return
        else:
            if R.bit_count() <= best_size:
                return
            # Update best
            best_size = R.bit_count()
            best_mask = R
            return
            
        # Pivot selection
        # Choose u in P | X with max degree in P
        U = P | X
        if U:
            # Find u with max (adj[u] & P).bit_count()
            max_deg = -1
            pivot = 0
            m = U
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                deg = (adj[u] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
                m ^= lsb
            candidates = P & ~(1 << pivot)
        else:
            candidates = P
            
        # Iterate candidates
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            # R' = R U {v}, P' = P N(v), X' = X N(v)
            R_new = R | (1 << v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bron_kerbosch(R_new, P_new, X_new)
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)
            
            # Pruning
            if P == 0 and X == 0:
                break # Can't continue if no vertices left? 
            # Actually, if P is empty, the loop ends naturally.
            # Check if we can beat best with remaining P
            if P:
                 # Weak bound: popcount(P)
                 if R.bit_count() + P.bit_count() <= best_size:
                     # We can stop if even adding all P doesn't beat best
                     # But we need to be careful. 
                     # Coloring bound is better, but expensive.
                     # This weak bound is cheap.
                     # If it fails, we might still need to check?
                     # No, if |R| + |P| <= best, we can't find a larger clique in this branch.
                     # So we can break the loop if we know we can't beat best.
                     # However, the loop structure iterates through candidates.
                     # If we break here, we stop exploring other candidates.
                     # Is it safe?
                     # Yes, because all remaining candidates are in P.
                     # Any clique found will be R U {subset of P}.
                     # Size <= |R| + |P|.
                     if R.bit_count() + P.bit_count() <= best_size:
                         break

    # Start search
    # Initial P is all vertices
    initial_P = (1 << N) - 1
    bron_kerbosch(0, initial_P, 0)
    
    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return sorted(result)

# Test locally
# print(construct())
