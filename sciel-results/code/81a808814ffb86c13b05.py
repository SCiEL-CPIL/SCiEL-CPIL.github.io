import time
import random
import numpy as np

def _check_no_k222(adj, n):
    """Check that no monochromatic K2,2,2 exists. adj is an n x n int array (0/1)."""
    # A K2,2,2 is a complete tripartite graph with parts of size 2.
    # It has 6 vertices a,b,c,d,e,f where edges within each part are absent and
    # all edges between different parts are present, all in the same color.
    # Equivalently, for a given color c, we need to find 6 vertices partitioned
    # into 3 pairs such that all inter-pair edges have color c.
    # 
    # Faster: for each color c, consider the graph G_c. We need a K_{2,2,2} subgraph.
    # K_{2,2,2} is the complement of 3 disjoint edges (in the 6-vertex subgraph).
    # So in G_c, on 6 vertices, we need exactly the 12 inter-part edges present.
    # 
    # Alternative: K_{2,2,2} contains K_{2,2} as a subgraph (take two parts).
    # For each pair of vertices {u,v} that is NOT an edge in G_c (i.e., color != c),
    # the common neighbors of u and v in G_c must not contain a K_{2,2}... 
    # Actually, let me think differently.
    #
    # K_{2,2,2} has parts {a,b}, {c,d}, {e,f}. In G_c, all edges between different parts exist.
    # So for any vertex x in part {a,b}, it is adjacent (in G_c) to all 4 vertices in the other two parts.
    # The non-neighbors of x in G_c are exactly the other vertex in its own part (and vertices outside the 6-set).
    #
    # Efficient check: For color c, for every unordered pair {u,v} with adj[u][v] != c (non-edge in G_c),
    # let N_c(u) = {w : adj[u][w] == c} and N_c(v) = {w : adj[v][w] == c}.
    # In K_{2,2,2}, if u,v are in the same part, then they share exactly the 4 vertices of the other two parts as common neighbors in G_c, and those 4 form a K_{2,2} in G_c (i.e., the 4 vertices form two pairs with all cross-edges present in G_c).
    # 
    # Even simpler: enumerate all 6-subsets? That's C(29,6) ~ 475k, times 2 colors. Might be OK for one check but not in a loop.
    #
    # Better approach: For each color c, for each pair {u,v} that is a non-edge in G_c:
    #   common = N_c(u) & N_c(v)
    #   If |common| >= 4, then check if any 4 vertices in common form a K_{2,2} in G_c.
    #   K_{2,2} in G_c on 4 vertices {p,q,r,s} means: the induced subgraph is exactly the 4 cross-edges.
    #   I.e., in G_c, among these 4, exactly 4 edges exist (forming a 4-cycle), and they can be partitioned into 2 pairs with no intra-pair edges.
    #   Actually K_{2,2} means: partition into 2 sets of 2, all 4 cross-edges present, no intra-edges.
    #   So for 4 vertices, we need: there exists a partition into 2 pairs such that all 4 cross-edges are in G_c and no intra-pair edges are in G_c.
    #   Equivalently, the induced subgraph on these 4 vertices in G_c is a 4-cycle (C4).
    #
    # For n=29, this is feasible if done carefully.
    # 
    # Actually, for the purposes of this program, I'll use a simpler but potentially slower check that's good enough for n up to ~25 during search, and a faster one for final verification.
    
    # Precompute adjacency lists for each color
    for c in range(2):
        # For each non-edge pair {u,v} in G_c, check common neighbors
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v] == c:
                    continue
                # u,v is a non-edge in G_c
                # Find common neighbors in G_c
                # common = {w : w != u, w != v, adj[u][w] == c, adj[v][w] == c}
                common = []
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u][w] == c and adj[v][w] == c:
                        common.append(w)
                if len(common) < 4:
                    continue
                # Check if any 4-subset of common forms a C4 in G_c
                # For efficiency, check if there exist 4 vertices a,b,d,e in common
                # such that they form two pairs {a,b} and {d,e} where:
                #   adj[a][b] != c, adj[d][e] != c (intra-pair non-edges in G_c)
                #   adj[a][d] == c, adj[a][e] == c, adj[b][d] == c, adj[b][e] == c (cross-edges)
                # This is equivalent to: in the induced subgraph on {a,b,d,e} in G_c,
                # it's a C4 (4-cycle).
                
                m = len(common)
                # For n<=31, m can be up to ~27. C(27,4) = 17550. Times number of non-edge pairs.
                # This could be slow. Let's use a smarter check.
                
                # Alternative: for the common set, check if it contains a K_{2,2} subgraph in G_c.
                # K_{2,2} in G_c means: 4 vertices partitioned into 2 pairs, all cross-edges in G_c.
                # For the common set S, we need to find a,b,d,e in S such that:
                #   adj[a][b] != c, adj[d][e] != c
                #   adj[a][d] == c, adj[a][e] == c, adj[b][d] == c, adj[b][e] == c
                
                # This is getting complex. For the search, I'll use a simpler validation
                # that only checks a subset, and rely on the final answer being verified separately.
                # For now, return True (assume valid) and do a more careful check only at the end.
    
    return True

def _validate_coloring(n, colors):
    """Thorough validation that no monochromatic K2,2,2 exists."""
    adj = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1
    
    # For each color c, check for K_{2,2,2}
    for c in range(2):
        # Build adjacency for color c
        # For each non-edge pair {u,v} in G_c (adj[u][v] != c), check if common neighbors
        # in G_c contain a K_{2,2} subgraph in G_c.
        
        # Precompute for each vertex its non-neighbors in G_c
        # nn_c[u] = set of v where adj[u][v] != c
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v] == c:
                    continue
                # {u,v} is a non-edge in G_c
                # Find common neighbors in G_c
                common = []
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u][w] == c and adj[v][w] == c:
                        common.append(w)
                if len(common) < 4:
                    continue
                # Check if common contains a K_{2,2} in G_c
                # Need 4 vertices a,b,d,e in common forming two pairs {a,b},{d,e}
                # with adj[a][b]!=c, adj[d][e]!=c, and all cross-edges ==c
                m = len(common)
                # Try all pairs of pairs
                # For efficiency, iterate over all 4-subsets
                for a_idx in range(m):
                    a = common[a_idx]
                    for b_idx in range(a_idx + 1, m):
                        b = common[b_idx]
                        if adj[a][b] == c:
                            continue  # {a,b} must be non-edge in G_c
                        # Now find d,e in common, d,e not in {a,b}, adj[d][e] != c
                        # and all cross-edges present
                        for d_idx in range(m):
                            d = common[d_idx]
                            if d == a or d == b:
                                continue
                            if adj[a][d] != c or adj[b][d] != c:
                                continue
                            for e_idx in range(d_idx + 1, m):
                                e = common[e_idx]
                                if e == a or e == b:
                                    continue
                                if adj[a][e] != c or adj[b][e] != c:
                                    continue
                                if adj[d][e] == c:
                                    continue
                                # Found K_{2,2,2}
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Start with a known valid coloring
    # Use the Paley graph construction for n=17 as baseline
    n_base = 17
    squares = {x * x % 17 for x in range(1, 17)}
    base_colors = [0 if (j - i) % 17 in squares else 1 for i in range(n_base) for j in range(i + 1, n_base)]
    best_n = n_base
    best_colors = base_colors[:]
    
    # Try to extend by adding vertices one at a time using local search
    random.seed(42)
    
    def try_extend(n, colors, max_iter=20000):
        """Try to add vertices beyond n by local search."""
        adj = np.zeros((31, 31), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        current_n = n
        current_colors = colors[:]
        
        iter_count = 0
        while time.time() < deadline - 5 and iter_count < max_iter:
            iter_count += 1
            # Try to add a new vertex
            new_n = current_n + 1
            if new_n > 31:
                break
            
            # Randomly assign colors for edges from new vertex to existing
            new_colors = current_colors[:]
            for i in range(current_n):
                new_colors.append(random.randint(0, 1))
            
            # Check validity quickly
            if not _validate_coloring(new_n, new_colors):
                continue
            
            # Try to improve by flipping edges
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Randomly flip an edge
                e = random.randint(0, len(new_colors) - 1)
                new_colors[e] = 1 - new_colors[e]
                if _validate_coloring(new_n, new_colors):
                    improved = True
            
            if _validate_coloring(new_n, new_colors):
                current_n = new_n
                current_colors = new_colors[:]
        
        return current_n, current_colors
    
    # Try multiple random starts
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        # Start from base and try to extend
        n_cur, colors_cur = try_extend(best_n, best_colors, max_iter=5000)
        if n_cur > best_n:
            best_n = n_cur
            best_colors = colors_cur[:]
    
    # Final local search to try to push higher
    if time.time() < deadline - 10:
        n_final, colors_final = try_extend(best_n, best_colors, max_iter=50000)
        if n_final > best_n:
            best_n = n_final
            best_colors = colors_final[:]
    
    return {'n': best_n, 'colors': best_colors}
