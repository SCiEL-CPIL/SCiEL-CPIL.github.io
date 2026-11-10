import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            lcm_table[i][j] = i * j // math.gcd(i, j)

    # Precompute prime factorization or just use a helper to find M_u
    # M_u(L, u) = product of p^k for p^k || L if v_p(u) < k
    # We can compute this by iterating through divisors or prime factors.
    # Given small N, we can just check divisors of L.
    
    # For each pair (u, v), check if they are incompatible.
    # Incompatible if exists w in [1, N], w != u, w != v such that:
    # 1. w | L
    # 2. lcm(u, w) == L
    # 3. lcm(v, w) == L
    
    # Optimization:
    # lcm(u, w) == L is equivalent to (L // gcd(L, u)) divides w.
    # Let req_u = L // gcd(L, u). Then req_u must divide w.
    # Similarly req_v = L // gcd(L, v). req_v must divide w.
    # So lcm(req_u, req_v) must divide w.
    # And w must divide L.
    # So we need a divisor w of L such that lcm(req_u, req_v) | w.
    # Let R = lcm(req_u, req_v).
    # We need a multiple of R that divides L.
    # Such a w exists if and only if R | L.
    # If R | L, then w = R is a candidate.
    # We must check if w == u or w == v.
    # If w == u, then req_u | u. req_u = L/gcd(L,u).
    # If req_u | u, then L/gcd(L,u) | u => L | u * gcd(L,u). Since u|L, this implies L|L.
    # Actually, if w=u, then lcm(u,u)=u. We need u=L.
    # If w=v, we need v=L.
    # So if R | L, let w_candidate = R.
    # If w_candidate != u and w_candidate != v, then u,v are incompatible.
    # What if w_candidate == u? Then we need another multiple of R dividing L.
    # Multiples of R dividing L are of the form R * k where k | (L/R).
    # So we need to check if there is a divisor w of L such that R | w and w not in {u, v}.
    
    # Precompute divisors for each number up to 128? Or just iterate.
    # LCM can be up to 128*127 ~ 16k.
    # Let's precompute divisors for all possible LCMs?
    # Max LCM is lcm(128, 127) = 16256.
    # Precomputing divisors for 1..16256 is fast.
    
    max_lcm = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if lcm_table[i][j] > max_lcm:
                max_lcm = lcm_table[i][j]
                
    # divisors[x] = list of divisors of x
    divisors = [[] for _ in range(max_lcm + 1)]
    for i in range(1, max_lcm + 1):
        for j in range(i, max_lcm + 1, i):
            divisors[j].append(i)

    # Build incompatibility graph
    # incompat[u][v] = True if incompatible
    # Use bitmasks for speed in MIS
    # incompat_mask[u] is an int with bit v set if u and v are incompatible
    
    incompat_mask = [0] * (n + 1)
    
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            L = lcm_table[u][v]
            req_u = L // math.gcd(L, u)
            req_v = L // math.gcd(L, v)
            R = req_u * req_v // math.gcd(req_u, req_v)
            
            # Check if there exists w | L such that R | w and w not in {u, v}
            # This is equivalent to:
            # 1. R | L
            # 2. There is a multiple of R in divisors[L] that is not u and not v.
            
            if L % R == 0:
                # Find if any divisor of L is a multiple of R and not u, v
                found = False
                # Iterate divisors of L
                for w in divisors[L]:
                    if w % R == 0:
                        if w != u and w != v:
                            found = True
                            break
                if found:
                    incompat_mask[u] |= (1 << v)
                    incompat_mask[v] |= (1 << u)
            
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break

    # Now solve MIS on the graph defined by incompat_mask
    # Vertices 1..n
    # We want max independent set.
    # Independent set in G is Clique in Complement G'.
    # Let's work directly with Independent Set using B&B.
    # Or convert to Clique in Complement.
    # Complement adjacency: comp_mask[u] has bit v set if NOT incompatible.
    # Note: u is always compatible with itself in the graph sense for clique?
    # For MIS, we pick vertices.
    
    # Let's use a standard MIS B&B.
    # State: available set P (bitmask), current set C (size).
    # Upper bound: Greedy coloring of P.
    
    # Precompute complement masks for faster intersection?
    # comp_mask[u] = ~incompat_mask[u] & ((1 << (n+1)) - 1)
    # But we only care about vertices 1..n.
    # Let's define ALL_MASK = sum(1 << i for i in range(1, n+1))
    
    ALL_MASK = 0
    for i in range(1, n + 1):
        ALL_MASK |= (1 << i)
        
    # comp_mask[u] contains bits for all v != u such that u and v are compatible.
    comp_mask = [0] * (n + 1)
    for u in range(1, n + 1):
        # Start with all, remove u, remove incompatible
        comp_mask[u] = (ALL_MASK ^ (1 << u)) & ~incompat_mask[u]

    best_size = 0
    best_set = 0  # bitmask
    
    # Greedy Coloring for Upper Bound
    # Input: P (bitmask of candidates)
    # Output: number of colors
    def greedy_color_bound(P):
        if P == 0:
            return 0
        colors = 0
        remaining = P
        while remaining:
            colors += 1
            # Pick a vertex with max degree in remaining? Or just first one?
            # For upper bound, any valid coloring works.
            # Greedy: pick a vertex, remove it and its neighbors from 'available for this color'
            # Wait, standard greedy coloring:
            # Assign color 1 to a maximal independent set of remaining?
            # No, greedy coloring assigns colors sequentially.
            # Vertex v gets color k if it is not adjacent to any vertex with color < k.
            # This is O(V^2).
            
            # Faster heuristic:
            # Just count how many vertices we can pack.
            # A simple bound is just popcount(P), which is weak.
            # Let's do a simple greedy coloring.
            uncolored = remaining
            color_set = 0
            while uncolored:
                # Pick vertex with lowest index?
                # To make it fast, pick the first set bit.
                v_bit = uncolored & (-uncolored)
                v = v_bit.bit_length() - 1
                
                # Remove v from uncolored
                uncolored ^= v_bit
                
                # Remove neighbors of v from uncolored (they can't have same color)
                # Neighbors in the graph G (incompatibility) are those that CANNOT share color?
                # Wait. In Coloring for MIS Bound:
                # We color the vertices of the induced subgraph on P.
                # The graph is the INCOMPATIBILITY graph.
                # If u and v are incompatible, they must have different colors.
                # So we color the incompatibility graph.
                # The number of colors is an upper bound on the size of the Independent Set.
                
                # So, if we assign color `colors` to v, we must remove all neighbors of v (incompatibilities) from the pool for this color?
                # No, standard greedy coloring:
                # We try to assign the current color to as many vertices as possible.
                # A vertex can take the current color if it is not adjacent to any vertex already assigned this color.
                
                # Let's re-implement:
                # available_for_color = remaining
                # While available_for_color:
                #   Pick v
                #   Assign color
                #   Remove v and its neighbors from available_for_color?
                #   No, remove v. Its neighbors cannot take this color, so remove them from available_for_color.
                
                # My loop above was slightly off.
                pass
                
            # Let's do a cleaner implementation
            pass
        return colors

    # Correct Greedy Coloring for Upper Bound
    def get_color_bound(P):
        if P == 0:
            return 0
        
        # Order vertices by degree in P? Or just iterate.
        # For speed, we'll use a simple heuristic:
        # 1. Initialize color_count = 0
        # 2. While P not empty:
        #    color_count++
        #    Let C be the set of vertices in P that can take this color.
        #    Initially C = P.
        #    While C not empty:
        #       Pick v from C (e.g., lowest index or highest degree)
        #       Assign color to v.
        #       Remove v from C.
        #       Remove all neighbors of v (incompatibilities) from C.
        #       Remove v from P.
        #    (Actually, we remove v from P immediately when assigned).
        #    Wait, the standard algorithm:
        #    while P:
        #       color++
        #       candidates = P
        #       while candidates:
        #          v = pick(candidates)
        #          candidates = candidates \ {v} \ N(v)
        #          P = P \ {v}
        #       # End while candidates
        #    # End while P
        #    Return color
        
        P_copy = P
        colors = 0
        while P_copy:
            colors += 1
            candidates = P_copy
            while candidates:
                # Pick vertex. Heuristic: pick one with max degree in candidates?
                # For speed, pick lowest bit.
                v_bit = candidates & (-candidates)
                v = v_bit.bit_length() - 1
                
                # Remove v from candidates
                candidates ^= v_bit
                
                # Remove neighbors of v from candidates
                # Neighbors are in incompat_mask[v]
                candidates &= ~incompat_mask[v]
                
                # Remove v from P_copy
                P_copy ^= v_bit
            # End while candidates
        # End while P_copy
        return colors

    # B&B Search
    # P: bitmask of available vertices
    # C_size: current size of independent set
    # C_mask: bitmask of current independent set (for reconstruction, optional)
    
    # To speed up, we can sort vertices or use heuristics.
    # We will use a recursive function.
    
    # Optimization: If P is empty, update best.
    # Upper bound: C_size + get_color_bound(P)
    
    # We need to avoid deep recursion limits.
    # sys.setrecursionlimit(10000)
    
    def expand(P, C_size, C_mask):
        nonlocal best_size, best_set
        
        if P == 0:
            if C_size > best_size:
                best_size = C_size
                best_set = C_mask
            return
        
        # Upper bound
        # Simple bound: C_size + popcount(P)
        # If C_size + popcount(P) <= best_size: return
        if C_size + P.bit_count() <= best_size:
            return
            
        # Stronger bound: Coloring
        # Note: Coloring is expensive. Use it selectively or with a faster heuristic.
        # Let's try a fast greedy coloring bound.
        # If time is tight, skip coloring and use degree-based heuristic?
        # For N=128, coloring is probably fast enough if implemented with bit ops.
        
        bound = get_color_bound(P)
        if C_size + bound <= best_size:
            return
        
        # Heuristic: Order vertices in P
        # Pick vertices with high degree in P first?
        # Or low degree?
        # For MIS, picking a vertex with low degree might leave more options?
        # Standard: Pick vertex with max degree in the subgraph.
        
        # Let's just iterate through bits.
        # To improve, we can reorder P.
        # For now, simple iteration.
        
        # Pick a vertex v from P
        # Heuristic: Pick v that maximizes something?
        # Let's pick the vertex with the highest index? Or lowest?
        # Let's try picking the one with the most neighbors in P (highest degree) to prune early.
        
        # Calculate degrees in P for all v in P?
        # This is O(|P|^2).
        # Let's just pick the first one for now to keep it simple, or use a static order.
        
        # Static order: 1 to n.
        # Or dynamic:
        # Let's pick v such that it has max number of neighbors in P.
        # This helps prune.
        
        best_v = -1
        max_deg = -1
        temp_P = P
        while temp_P:
            v_bit = temp_P & (-temp_P)
            v = v_bit.bit_length() - 1
            temp_P ^= v_bit
            # Degree of v in P: popcount(P & incompat_mask[v])
            deg = (P & incompat_mask[v]).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        if best_v == -1:
            return

        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New P: P without best_v and without its neighbors
        new_P = P & ~v_bit & ~incompat_mask[best_v]
        expand(new_P, C_size + 1, C_mask | v_bit)
        
        # Branch 2: Exclude best_v
        # New P: P without best_v
        new_P = P & ~v_bit
        expand(new_P, C_size, C_mask)
        
        # Time check
        if time.time() > deadline - 2:
            # We could raise an exception or just return
            # Since we can't easily abort recursion cleanly without exception,
            # we check at the top.
            pass

    # Initial call
    # Start with all vertices available
    P_init = ALL_MASK
    expand(P_init, 0, 0)
    
    # Convert bitmask to list
    result = []
    for i in range(1, n + 1):
        if best_set & (1 << i):
            result.append(i)
            
    return result

if __name__ == "__main__":
    # For testing
    res = construct()
    print(len(res))
