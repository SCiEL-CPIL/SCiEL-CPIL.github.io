"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from math import gcd
    from collections import defaultdict

    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute all bad triples (hyperedges)
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    # We need to find all such triples among [1, 128]
    
    # First, let's identify which triples are "bad"
    # For efficiency, we'll build a conflict graph / hypergraph
    
    # Let's think about the structure:
    # gcd(a,b) == gcd(a,c) == gcd(b,c) = d
    # This means a, b, c are all multiples of d, and when divided by d,
    # the three resulting numbers are pairwise coprime.
    
    # So for each d from 1 to 128, consider numbers that are multiples of d in [1,128].
    # For each such d, any three numbers a=d*x, b=d*y, c=d*z where x,y,z are pairwise coprime
    # form a bad triple with common gcd d.
    
    # Wait, that's not quite right. Let me reconsider.
    # If gcd(a,b) = gcd(a,c) = gcd(b,c) = d, then:
    # - d divides a, b, c
    # - gcd(a/d, b/d) = gcd(a/d, c/d) = gcd(b/d, c/d) = 1
    # So a/d, b/d, c/d are pairwise coprime.
    
    # So the bad triples are: for each d, pick three multiples of d (say d*x, d*y, d*z)
    # such that x, y, z are pairwise coprime.
    
    # Strategy: Build a hypergraph where vertices are 1..128, and hyperedges are bad triples.
    # Then find a maximum independent set (no vertex in any hyperedge... wait, no).
    # Actually, we want a subset with NO bad triple entirely contained in it.
    # So we want a set S such that no hyperedge is fully contained in S.
    # This is the "3-uniform hypergraph independent set" or more precisely,
    # a set that does not contain any hyperedge.
    
    # This is equivalent to: S is an independent set in the 3-uniform hypergraph
    # where hyperedges are the bad triples. An independent set in a hypergraph
    # is a set of vertices that contains no hyperedge.
    
    # Heuristic: Greedy approach.
    # 1. Compute degree of each vertex (number of bad triples it's part of).
    # 2. Start with all vertices.
    # 3. While there exists a bad triple in the current set, remove one vertex from it.
    #    To maximize the final set size, we want to remove vertices that are in many bad triples.
    #    So when a bad triple is found, remove the vertex with the highest degree (among the three).
    
    # Alternatively, a better heuristic:
    # - Build the list of all bad triples.
    # - Use a weighted greedy: repeatedly pick the vertex with the fewest conflicts
    #   (i.e., involved in the fewest bad triples with already-selected vertices) and add it.
    
    # Let me try the "remove from bad triples" approach first, as it might be more effective.
    
    # Step 1: Enumerate all bad triples
    bad_triples = []
    # For each pair (a, b), compute d = gcd(a,b). Then for each c > b, check if gcd(a,c)==d and gcd(b,c)==d.
    # This is O(n^3) but n=128, so ~2M operations, which is fine.
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            d = gcd(a, b)
            for c in range(b+1, N+1):
                if gcd(a, c) == d and gcd(b, c) == d:
                    bad_triples.append((a, b, c))
    
    # Step 2: Build degree count for each vertex
    degree = [0] * (N+1)
    for (a, b, c) in bad_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
    # Step 3: Greedy removal
    # Start with all vertices selected.
    selected = set(range(1, N+1))
    
    # We'll maintain a list of active bad triples (those where all three vertices are in selected)
    # When we remove a vertex, we need to update which triples are still "active".
    
    # For efficiency, let's use a different approach:
    # Repeatedly find any bad triple in the current selected set, and remove the vertex
    # with the highest degree (in the current induced hypergraph) from that triple.
    
    # To do this efficiently, let's precompute for each vertex the list of bad triples it's in.
    triples_by_vertex = defaultdict(list)
    for idx, (a, b, c) in enumerate(bad_triples):
        triples_by_vertex[a].append(idx)
        triples_by_vertex[b].append(idx)
        triples_by_vertex[c].append(idx)
    
    # Current active triples: a triple is active if all three vertices are in selected.
    # We'll maintain a count of how many vertices of each triple are still in selected.
    triple_active_count = [3] * len(bad_triples)  # Initially all 3 vertices are in selected
    
    # Remove vertices until no active triple remains
    # Priority: remove the vertex that is in the most active triples
    
    while True:
        # Find if there's any active triple
        active_triple_idx = -1
        for i in range(len(bad_triples)):
            if triple_active_count[i] == 3:
                active_triple_idx = i
                break
        
        if active_triple_idx == -1:
            break
        
        a, b, c = bad_triples[active_triple_idx]
        
        # Among a, b, c, find the one with the highest number of active triples it's in
        # For each vertex, count how many active triples (count==3) it's part of
        def count_active_triples(v):
            count = 0
            for tidx in triples_by_vertex[v]:
                if triple_active_count[tidx] == 3:
                    count += 1
            return count
        
        ca = count_active_triples(a)
        cb = count_active_triples(b)
        cc = count_active_triples(c)
        
        # Pick the vertex with the highest count
        max_count = max(ca, cb, cc)
        if max_count == ca:
            to_remove = a
        elif max_count == cb:
            to_remove = b
        else:
            to_remove = c
        
        # Remove to_remove from selected
        selected.remove(to_remove)
        
        # Update triple_active_count for all triples involving to_remove
        for tidx in triples_by_vertex[to_remove]:
            triple_active_count[tidx] -= 1
    
    # Now selected has no bad triple. Let's convert to a list.
    result = sorted(selected)
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
