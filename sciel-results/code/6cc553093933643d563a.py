"""Program for R(K2,2,2) task using CSP-style local search with NumPy."""
import numpy as np

def construct():
    # We want to maximize n such that there exists a 2-coloring of K_n
    # with no monochromatic K_{2,2,2}.
    # 
    # K_{2,2,2} is the complete tripartite graph with parts of size 2.
    # It has 6 vertices. A monochromatic K_{2,2,2} in color c means
    # there exist two disjoint pairs {a,b}, {c,d}, {e,f} such that all
    # edges between the parts are colored c.
    #
    # Equivalently, for each color c, the graph G_c (edges of color c)
    # must not contain K_{2,2,2} as a subgraph.
    #
    # Strategy: Start with a known valid coloring (Paley 17), then try to
    # extend it to larger n by adding vertices and using local search to
    # fix violations. Alternatively, use local search on a fixed n to find
    # a valid coloring.
    #
    # Let's try n=29 first (known record), then attempt n=30.
    #
    # For efficiency, we'll use a vectorized approach to count monochromatic
    # K_{2,2,2} subgraphs.
    
    def count_mono_k222(adj):
        """Count monochromatic K_{2,2,2} subgraphs.
        adj: n x n binary matrix where adj[i][j] is the color (0 or 1).
        Returns total count of mono K222.
        """
        n = adj.shape[0]
        # For each color c, count K_{2,2,2} in G_c
        total = 0
        for c in [0, 1]:
            # G_c has edge (i,j) if adj[i,j] == c
            # K_{2,2,2} with parts {a,b}, {c,d}, {e,f} requires:
            # all 4 edges between {a,b} and {c,d} are in G_c
            # all 4 edges between {a,b} and {e,f} are in G_c
            # all 4 edges between {c,d} and {e,f} are in G_c
            # So we need three pairs of vertices, each pair internally
            # not necessarily connected, but all cross edges present.
            #
            # Let's use a different approach: for each unordered pair of
            # disjoint vertex pairs (u1,u2) and (v1,v2), count the number
            # of vertices w such that all edges from {u1,u2} to {v1,v2}
            # are in G_c... wait, that's not right either.
            #
            # Actually, K_{2,2,2} has 6 vertices partitioned into 3 pairs.
            # Let's iterate over all ways to choose 3 disjoint pairs.
            # But that's O(n^6) which is too slow.
            #
            # Better approach: For a fixed color c, define for each pair
            # (i,j) the "common neighborhood" in G_c. Then K_{2,2,2} with
            # parts {a,b}, {c,d}, {e,f} exists iff:
            # - All 4 edges between {a,b} and {c,d} are in G_c
            # - All 4 edges between {a,b} and {e,f} are in G_c  
            # - All 4 edges between {c,d} and {e,f} are in G_c
            #
            # This is equivalent to: the 6x6 bipartite structure...
            #
            # Let me think differently. K_{2,2,2} is the complement of a
            # perfect matching in K_6. So a monochromatic K_{2,2,2} in
            # color c means a 6-set where all but 3 edges (the matching)
            # are color c... no, that's not right. K_{2,2,2} has 12 edges.
            #
            # Let's just do a direct check for small n. For n=29, there are
            # C(29,6) = 475020 six-sets, which is manageable.
            
            # For each 6-set, check if it induces a K_{2,2,2} in G_c.
            # K_{2,2,2} on vertices {0,1,2,3,4,5} has edges between
            # {0,1}-{2,3}, {0,1}-{4,5}, {2,3}-{4,5}.
            # There are 15 ways to partition 6 vertices into 3 pairs.
            # For each partition, check if all 12 cross edges are in G_c.
            
            from itertools import combinations
            # Precompute all partitions of 6 elements into 3 unordered pairs
            # where the pairs are ordered (part 0, part 1, part 2)
            partitions = []
            for a in range(6):
                for b in range(a+1, 6):
                    for c2 in range(b+1, 6):
                        for d in range(c2+1, 6):
                            remaining = [x for x in range(6) if x not in (a,b,c2,d)]
                            e, f = remaining
                            partitions.append(((a,b), (c2,d), (e,f)))
            
            for combo in combinations(range(n), 6):
                # Check if this 6-set forms K_{2,2,2} in G_c
                for (p1, p2, p3) in partitions:
                    ok = True
                    # Check all edges between p1 and p2
                    for u in p1:
                        for v in p2:
                            if adj[combo[u], combo[v]] != c:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    # Check all edges between p1 and p3
                    for u in p1:
                        for v in p3:
                            if adj[combo[u], combo[v]] != c:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    # Check all edges between p2 and p3
                    for u in p2:
                        for v in p3:
                            if adj[combo[u], combo[v]] != c:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        total += 1
                        break  # Count each 6-set at most once per color
        
        return total

    # Start with Paley 17
    def paley_coloring(n):
        """Paley coloring for prime n where n ≡ 1 mod 4."""
        if n == 17:
            squares = {x * x % 17 for x in range(1, 17)}
            return [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
        return None
    
    # Try to find a valid coloring for n=29 using local search
    # Start from random initializations and flip edges to reduce mono K222 count
    
    import random
    import time
    
    best_n = 17
    best_colors = paley_coloring(17)
    
    # Verify the 17-vertex solution
    def colors_to_adj(colors, n):
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                adj[i, j] = colors[idx]
                adj[j, i] = colors[idx]
                idx += 1
        return adj
    
    adj17 = colors_to_adj(best_colors, 17)
    # This should have 0 mono K222
    
    # Try n=29 with local search
    target_n = 29
    max_time = 110  # seconds
    
    # We'll use a faster counting method
    def fast_count_mono_k222(adj):
        """Faster count using matrix operations."""
        n = adj.shape[0]
        total = 0
        for c in [0, 1]:
            # Build adjacency matrix for color c
            A = (adj == c).astype(np.int8)
            np.fill_diagonal(A, 0)
            
            # For each pair of vertices (i,j), compute the number of common
            # neighbors in G_c. But we need K_{2,2,2} which is more complex.
            #
            # Alternative: iterate over all pairs of disjoint pairs.
            # For pairs (i,j) and (k,l) with all distinct, count if all
            # 4 cross edges are in G_c. Then for each such "rectangle",
            # count how many pairs (m,n) also form rectangles with both.
            
            # Let's use a simpler approach for n=29:
            # Precompute for each pair (i,j) the set of vertices connected
            # to both i and j in G_c.
            
            # Actually, let's just do the O(n^6) approach but optimized.
            # For n=29, C(29,6) ≈ 475k, and for each we check 15 partitions
            # with up to 12 edge checks each. That's ~85M operations, which
            # might be slow in Python but let's try with NumPy vectorization.
            
            from itertools import combinations
            # Precompute all partitions of 6 elements into 3 ordered pairs
            partitions = []
            for a in range(6):
                for b in range(a+1, 6):
                    for c2 in range(b+1, 6):
                        for d in range(c2+1, 6):
                            remaining = [x for x in range(6) if x not in (a,b,c2,d)]
                            e, f = remaining
                            partitions.append(((a,b), (c2,d), (e,f)))
            
            for combo in combinations(range(n), 6):
                # Get the 6x6 submatrix
                sub = A[np.ix_(combo, combo)]
                for (p1, p2, p3) in partitions:
                    # Check if all edges between p1 and p2 are 1
                    # p1, p2, p3 are tuples of positions in 0..5
                    ok = True
                    for u in p1:
                        for v in p2:
                            if sub[u, v] != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    for u in p1:
                        for v in p3:
                            if sub[u, v] != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    for u in p2:
                        for v in p3:
                            if sub[u, v] != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        total += 1
                        break
        
        return total

    # Local search for n=29
    start_time = time.time()
    found_valid_29 = False
    
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        # Initialize: start with random coloring, or extend Paley 17
        if attempt == 0:
            # Try to extend Paley 17 to 29
            colors = [0] * (29 * 28 // 2)
            # Fill in the 17x17 part with Paley
            idx = 0
            for i in range(17):
                for j in range(i+1, 17):
                    colors[idx] = best_colors[idx]
                    idx += 1
            # Random for the rest
            for i in range(17, 29):
                for j in range(i+1, 29):
                    colors[idx] = random.randint(0, 1)
                    idx += 1
                for j in range(i):
                    # Need to find the right index
                    pass
            # This indexing is tricky. Let me redo.
            
            colors = [random.randint(0, 1) for _ in range(29 * 28 // 2)]
        
        adj = colors_to_adj(colors, 29)
        
        # Local search: flip edges that reduce mono K222 count
        for iteration in range(500):
            if time.time() - start_time > max_time:
                break
            
            count = fast_count_mono_k222(adj)
            if count == 0:
                found_valid_29 = True
                best_n = 29
                best_colors = colors[:]
                break
            
            # Try flipping a random edge
            i = random.randint(0, 28)
            j = random.randint(0, 28)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Compute new count after flip
            old_color = adj[i, j]
            adj[i, j] = 1 - old_color
            adj[j, i] = 1 - old_color
            new_count = fast_count_mono_k222(adj)
            
            if new_count < count:
                # Accept
                idx = 0
                for a in range(29):
                    for b in range(a+1, 29):
                        if a == i and b == j:
                            colors[idx] = 1 - old_color
                        idx += 1
            else:
                # Revert
                adj[i, j] = old_color
                adj[j, i] = old_color
        
        if found_valid_29:
            break
    
    if found_valid_29:
        return {'n': 29, 'colors': best_colors}
    
    # If we couldn't find 29, try to at least return the best we have
    # Let's also try n=18, 19, ... up to 28 with simpler search
    
    for target_n in range(18, 29):
        if time.time() - start_time > max_time:
            break
        
        for attempt in range(10):
            if time.time() - start_time > max_time:
                break
            
            colors = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
            adj = colors_to_adj(colors, target_n)
            
            for iteration in range(100):
                if time.time() - start_time > max_time:
                    break
                
                count = fast_count_mono_k222(adj)
                if count == 0:
                    best_n = target_n
                    best_colors = colors[:]
                    break
                
                i = random.randint(0, target_n - 1)
                j = random.randint(0, target_n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                old_color = adj[i, j]
                adj[i, j] = 1 - old_color
                adj[j, i] = 1 - old_color
                new_count = fast_count_mono_k222(adj)
                
                if new_count < count:
                    idx = 0
                    for a in range(target_n):
                        for b in range(a+1, target_n):
                            if a == i and b == j:
                                colors[idx] = 1 - old_color
                            idx += 1
                else:
                    adj[i, j] = old_color
                    adj[j, i] = old_color
            
            if best_n == target_n:
                break
    
    return {'n': best_n, 'colors': best_colors}
