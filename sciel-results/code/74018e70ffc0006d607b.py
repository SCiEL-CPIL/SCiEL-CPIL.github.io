def construct():
    import math
    from math import gcd

    n = 128
    
    # Precompute gcd table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            gcd_table[i][j] = gcd(i, j)

    # Find all bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                    bad_triples.append((a, b, c))

    # Initial set is all numbers
    S = set(range(1, n + 1))
    
    # Count how many bad triples each number is part of
    # We will iteratively remove numbers to break all bad triples
    
    # To make it faster, we can maintain a list of active bad triples
    # But with N=128, total triples is ~ 350k. 
    # Number of bad triples is likely much smaller.
    
    # Let's filter bad_triples to only those with all elements in S
    # We'll use a loop:
    # 1. Find all bad triples in current S
    # 2. If none, stop
    # 3. For each number, count how many bad triples it is in
    # 4. Remove the number with the max count
    # 5. Repeat
    
    # Optimization: Pre-calculate for each number, the list of bad triples it is in?
    # Or just re-scan?
    # Re-scanning 350k triples is fast in Python if optimized.
    # But let's use a smarter way.
    
    # Let's use the "remove max degree" heuristic on the hypergraph.
    
    # First, let's identify the bad triples once.
    # Then, we maintain a set of "active" bad triples.
    
    # Actually, a simpler heuristic:
    # Sort numbers by their "badness" (number of bad triples they are in).
    # Greedily keep numbers with low badness, remove those with high badness.
    
    # Let's calculate initial badness for all nodes.
    badness = [0] * (n + 1)
    for a, b, c in bad_triples:
        badness[a] += 1
        badness[b] += 1
        badness[c] += 1
    
    # Create a list of numbers sorted by badness (ascending)
    # We want to keep numbers with low badness.
    # But removing one number breaks all triples containing it.
    
    # Greedy Set Cover / Hitting Set approach:
    # We want to remove a set of vertices $R$ such that every bad triple intersects $R$.
    # We want to minimize $|R|$, which maximizes $|S| = 128 - |R|$.
    
    # Greedy Hitting Set:
    # While there are uncovered bad triples:
    #   Pick the vertex that covers the most uncovered bad triples.
    #   Add it to R.
    #   Mark all bad triples containing it as covered.
    
    covered = [False] * len(bad_triples)
    R = set()
    
    # Precompute for each vertex, the list of indices of bad triples it is in
    vertex_triples = [[] for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(bad_triples):
        vertex_triples[a].append(idx)
        vertex_triples[b].append(idx)
        vertex_triples[c].append(idx)
    
    # Greedy loop
    # To be efficient, we can just iterate.
    # Since N=128, and number of bad triples is manageable, this is fast.
    
    # We need to pick the vertex covering the most UNCOVERED triples.
    # We can just scan all vertices in each iteration?
    # Number of iterations <= number of vertices = 128.
    # Scanning 128 vertices * 128 iterations = 16k steps.
    # Each step checks list of triples for that vertex.
    # This is very fast.
    
    while True:
        # Check if all covered
        all_covered = True
        for is_cov in covered:
            if not is_cov:
                all_covered = False
                break
        
        if all_covered:
            break
        
        # Find vertex with max uncovered coverage
        best_vertex = -1
        max_covered = -1
        
        for v in range(1, n + 1):
            if v in R:
                continue
            
            count = 0
            for idx in vertex_triples[v]:
                if not covered[idx]:
                    count += 1
            
            if count > max_covered:
                max_covered = count
                best_vertex = v
        
        if best_vertex == -1:
            break # Should not happen if there are uncovered triples
        
        R.add(best_vertex)
        
        # Mark all triples containing best_vertex as covered
        for idx in vertex_triples[best_vertex]:
            covered[idx] = True
            
    S = set(range(1, n + 1)) - R
    
    # Now try to add back removed vertices if possible
    # Iterate over R in reverse order of addition (or sorted by badness ascending)
    # Try to add v back if it doesn't create a new bad triple with existing S
    
    # We need a fast check: "Does adding v to S create a bad triple?"
    # A bad triple containing v must be composed of v and two other elements in S.
    # So we check all pairs (x, y) in S such that (v, x, y) is a bad triple.
    
    # Precompute for each v, the set of pairs (x,y) such that (v,x,y) is bad.
    # Or just check on the fly.
    
    # Let's sort R by badness (ascending) so we try to add back "least bad" vertices first.
    # Actually, we should try to add back vertices that don't conflict.
    
    # Let's just iterate through all v in R (maybe sorted by original badness)
    # and try to add them.
    
    # To check if v can be added:
    # We need to ensure no triple (v, x, y) with x,y in S is bad.
    
    # We can precompute for each v, the list of "bad partners" pairs?
    # Or just:
    # For each v in R:
    #   potential_conflict = False
    #   For each x in S:
    #     For each y in S where y > x:
    #       if gcd(v,x) == gcd(v,y) == gcd(x,y):
    #         potential_conflict = True
    #         break
    #   if not potential_conflict:
    #     S.add(v)
    
    # This is O(|R| * |S|^2). |R| ~ 50, |S| ~ 78. 50 * 6000 = 300k gcd checks.
    # This is fine.
    
    # Let's sort R by badness (ascending) to add back "safer" vertices first.
    # Vertices with low badness are less likely to conflict.
    r_list = sorted(list(R), key=lambda x: badness[x])
    
    for v in r_list:
        can_add = True
        s_list = list(S)
        for i in range(len(s_list)):
            x = s_list[i]
            for j in range(i + 1, len(s_list)):
                y = s_list[j]
                if gcd_table[v][x] == gcd_table[v][y] == gcd_table[x][y]:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            S.add(v)
            
    return list(S)
