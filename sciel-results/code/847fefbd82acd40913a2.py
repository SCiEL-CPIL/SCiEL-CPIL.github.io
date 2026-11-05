"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from math import gcd
    from collections import defaultdict
    import random

    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute bad triples and conflict structure
    # bad_triples is a list of tuples (a, b, c) with a < b < c
    bad_triples = []
    # For efficient conflict checking, we can store for each pair (x, y) the list of z
    # such that (x, y, z) is a bad triple.
    # Since order doesn't matter for the set, we can just check all pairs in S.
    # Let's store a set of frozensets or just a set of tuples for O(1) lookup?
    # Number of triples is not huge.
    
    # Precompute gcd matrix for speed
    gcd_matrix = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcd_matrix[i][j] = gcd(i, j)
            gcd_matrix[j][i] = gcd_matrix[i][j]
            
    # Generate bad triples
    # a < b < c
    # gcd(a,b) == gcd(a,c) == gcd(b,c)
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            d_ab = gcd_matrix[a][b]
            for c in range(b+1, N+1):
                if gcd_matrix[a][c] == d_ab and gcd_matrix[b][c] == d_ab:
                    bad_triples.append((a, b, c))
                    
    # Convert to a set for O(1) lookup if needed, or use for degree calc
    bad_triple_set = set(bad_triples)
    
    # Calculate static degree for each vertex
    degree = [0] * (N+1)
    for a, b, c in bad_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
        
    # 2. Weighted Greedy: Sort by degree ascending
    # Tie-breaker: lower number? or random? 
    # Let's use (degree, vertex) for determinism first, or maybe random tie-breaking to explore different solutions?
    # The directive says "prioritizes vertices with the lowest hypergraph degree".
    # We will sort by degree.
    sorted_vertices = sorted(range(1, N+1), key=lambda x: (degree[x], x))
    
    def is_valid_addition(candidate, current_set):
        """Check if candidate can be added to current_set without forming a bad triple."""
        # For each pair (u, v) in current_set, check if (candidate, u, v) is a bad triple.
        # We need to check if there exist u, v in current_set such that 
        # gcd(candidate, u) == gcd(candidate, v) == gcd(u, v).
        # Let d1 = gcd(candidate, u), d2 = gcd(candidate, v), d3 = gcd(u, v).
        # We need d1 == d2 == d3.
        
        # Optimization: 
        # If we add candidate, it forms a triple with any pair (u, v) in current_set 
        # such that gcd(candidate, u) == gcd(candidate, v) == gcd(u, v).
        
        # Iterate through pairs in current_set? 
        # Size of current_set is up to ~80. Pairs ~3000.
        # 128 candidates * 3000 pairs * 3 gcd lookups = ~1M ops. Acceptable.
        
        cs = list(current_set)
        n_cs = len(cs)
        for i in range(n_cs):
            u = cs[i]
            g_cu = gcd_matrix[candidate][u]
            for j in range(i+1, n_cs):
                v = cs[j]
                g_cv = gcd_matrix[candidate][v]
                if g_cu == g_cv:
                    g_uv = gcd_matrix[u][v]
                    if g_uv == g_cu:
                        return False
        return True

    # Greedy Construction
    S = set()
    for v in sorted_vertices:
        if is_valid_addition(v, S):
            S.add(v)
            
    # 3. Local Search for Swaps
    # Goal: Improve the set by swapping elements to increase the "potential" 
    # (number of vertices that can be added) or simply try to add more.
    
    def get_potential(S):
        """Count how many vertices not in S can be added to S."""
        count = 0
        for v in range(1, N+1):
            if v not in S:
                if is_valid_addition(v, S):
                    count += 1
        return count

    current_potential = get_potential(S)
    
    improved = True
    while improved:
        improved = False
        
        # Step 1: Try to add any vertex that is currently valid
        # Since S was built greedily, it should be maximal, but after swaps, it might not be.
        # We check all outside vertices.
        added_any = False
        for v in range(1, N+1):
            if v not in S:
                if is_valid_addition(v, S):
                    S.add(v)
                    added_any = True
                    # Update potential? We just added one, so potential decreases by at least 1.
                    # But other vertices might become invalid.
                    # Let's just restart the loop to be safe.
                    improved = True
                    break # Restart to re-evaluate all
        
        if added_any:
            continue
            
        # Step 2: If no additions, try swaps to increase potential
        # Calculate current potential again to be sure
        current_potential = get_potential(S)
        
        best_swap = None
        best_potential = current_potential
        best_u = None
        best_v = None
        
        # Try swapping u in S with v not in S
        # To keep it fast, we might limit swaps or use heuristics.
        # N=128, |S|~60, |outside|~60. 3600 pairs.
        # For each pair, check if swap is valid.
        # If valid, calculate new potential.
        
        for u in list(S):
            for v in range(1, N+1):
                if v in S:
                    continue
                
                # Check if S - {u} + {v} is independent
                # This requires checking if v is valid with S - {u}
                # We can reuse is_valid_addition logic but excluding u.
                
                # Let's create a temporary set or just check carefully.
                # is_valid_addition(v, S - {u})
                
                # Optimization: 
                # v is invalid in S if there is a pair (x, y) in S such that (v, x, y) is bad.
                # If we remove u, v is valid if NO pair (x, y) in S\{u} is bad with v.
                # Pairs in S\{u} are pairs in S that don't involve u.
                
                # Let's just call a helper that checks validity against a set excluding one element.
                
                def is_valid_excluding(candidate, current_set, excluded):
                    cs = [x for x in current_set if x != excluded]
                    n_cs = len(cs)
                    for i in range(n_cs):
                        u_val = cs[i]
                        g_cu = gcd_matrix[candidate][u_val]
                        for j in range(i+1, n_cs):
                            v_val = cs[j]
                            g_cv = gcd_matrix[candidate][v_val]
                            if g_cu == g_cv:
                                g_uv = gcd_matrix[u_val][v_val]
                                if g_uv == g_cu:
                                    return False
                    return True

                if is_valid_excluding(v, S, u):
                    # Swap is valid
                    # Calculate potential of new set
                    new_S = S.copy()
                    new_S.remove(u)
                    new_S.add(v)
                    
                    # Potential calculation is expensive (O(N * |S|^2)).
                    # 3600 swaps * 128 * 60^2 is too slow.
                    # We need a faster potential estimate or fewer swaps.
                    
                    # Heuristic: Only swap if v has lower degree than u?
                    # Or if v is "more central"?
                    # Let's try a simpler heuristic: 
                    # If we swap, does it unblock any specific vertex?
                    # Let's just try a limited number of swaps per iteration or use a faster check.
                    
                    # Actually, let's just try to swap if it reduces the "conflict load".
                    # Let's skip the full potential calculation for every swap.
                    # Instead, let's just try to find a swap that allows an immediate addition.
                    # i.e., if S' = S - {u} + {v}, can we add w to S'?
                    
                    # This is getting complex. Let's stick to a simpler local search:
                    # Randomized restarts of the greedy algorithm.
                    # Given N=128, we can run the greedy 100 times with random tie-breaking.
                    
                    pass # Skip complex swap logic for now, rely on restarts?
                    
        # If we don't do complex swaps, we should do Randomized Greedy.
        # Let's change strategy: Run the greedy multiple times with different orderings.
        
        break # Exit local search if we don't implement complex swaps
        
    # Let's implement Randomized Greedy instead of complex local search if the above is too slow/complex.
    # The directive says "followed by a final local search for swaps".
    # I will implement a simple swap: 
    # For each u in S, for each v not in S:
    #   If swap is valid, and degree[v] < degree[u], swap.
    # This might not increase size, but might lead to a state where size increases?
    # Actually, let's just do the Randomized Greedy. It's a "local search" in the space of orderings.
    # But the prompt specifically says "swaps".
    
    # Let's try a simple swap heuristic:
    # If no vertex can be added, try to swap u in S with v not in S 
    # if v has lower degree than u. 
    # This might not help immediately, but we can repeat.
    
    # However, a more effective "swap" for Max Independent Set is:
    # If S is maximal, find u in S such that removing u allows adding at least 2 vertices?
    # This is a 1-for-2 exchange.
    
    # Let's implement 1-for-2 exchange:
    # For each u in S:
    #   Let S_u = S - {u}
    #   Find all v not in S_u such that v is valid for S_u.
    #   If len(valids) >= 2, remove u and add all valids? 
    #   Wait, adding multiple vertices might create conflicts among them.
    #   So we need to find a maximal independent set in the "available" vertices given S_u.
    
    # This is getting too heavy. 
    # Let's stick to the simplest interpretation that might yield a jump:
    # 1. Greedy by degree.
    # 2. Local Search: Try to add any valid vertex.
    # 3. If stuck, try to swap u (high degree) with v (low degree) if valid.
    #    This keeps size same but might change structure.
    #    Then try to add again.
    
    # Let's implement this loop.
    
    max_iterations = 100
    for _ in range(max_iterations):
        # Try to add
        added = False
        for v in range(1, N+1):
            if v not in S:
                if is_valid_addition(v, S):
                    S.add(v)
                    added = True
                    break
        if added:
            continue
            
        # Try to swap
        swapped = False
        # Sort S by degree descending to remove "bad" vertices first?
        # Sort outside by degree ascending to add "good" vertices?
        # We want to replace high-degree u with low-degree v.
        
        # To speed up, we can just iterate.
        for u in list(S):
            for v in range(1, N+1):
                if v in S:
                    continue
                # Only consider swap if v is "better" (lower degree)
                if degree[v] >= degree[u]:
                    continue
                    
                if is_valid_excluding(v, S, u):
                    S.remove(u)
                    S.add(v)
                    swapped = True
                    break
            if swapped:
                break
                
        if not swapped:
            break
            
    result = sorted(S)
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
