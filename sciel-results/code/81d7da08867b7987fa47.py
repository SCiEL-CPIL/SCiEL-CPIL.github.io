"""Two-colourings without a monochromatic octahedron K2,2,2 (R(K2,2,2))."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _has_mono_k222(color_matrix, n):
    """Check if either color class contains K2,2,2 as a subgraph.
    color_matrix[i][j] is 0 or 1 for i<j, symmetric.
    Returns True if a monochromatic K2,2,2 exists."""
    # For each pair of vertices (a,b), look at common neighbors in color 0 and color 1
    # K2,2,2 in color c exists iff there are two disjoint pairs (a,b) and (c,d)
    # such that all 4 edges between {a,b} and {c,d} have color c.
    # Equivalently: there exist a,b,c,d (distinct) such that for some color col,
    # color(a,c)==color(a,d)==color(b,c)==color(b,d)==col.
    # 
    # Efficient check: for each color col, build the col-graph adjacency.
    # K2,2,2 in col-graph iff there exist two vertices a,b whose common neighborhood
    # in the col-graph has at least 4 vertices that form a set where we can pick
    # two more vertices c,d from common_neigh(a,b) such that c and d are also
    # connected... wait, K2,2,2 means complete bipartite K_{2,2} between two pairs?
    # No: K_{2,2,2} is the complete tripartite graph with parts of size 2.
    # It has 6 vertices: partition into 3 groups of 2, all edges between different groups.
    # So it's not bipartite. Let me reconsider.
    
    # K_{2,2,2} = complete tripartite graph K_{2,2,2} with 6 vertices.
    # Vertices partitioned into A={a1,a2}, B={b1,b2}, C={c1,c2}.
    # Edges: all pairs from different parts. 12 edges total.
    # So in the color graph, we need 6 vertices that can be partitioned into 3 pairs
    # such that all cross-edges have the same color.
    
    # A monochromatic K_{2,2,2} in color col exists iff there exist 6 distinct vertices
    # v1..v6 that can be grouped into 3 pairs (p1,p2,p3) such that for any two vertices
    # in different pairs, the edge color is col.
    
    # Equivalent: there exist 3 pairs of vertices such that the bipartite graph between
    # each pair and the union of the other two pairs is complete in color col.
    
    # More concretely: pick 6 distinct vertices. They form a K_{2,2,2} in color col
    # if and only if there's a partition into 3 pairs where all 12 cross-pair edges
    # have color col.
    
    # For a fixed set of 6 vertices, there are 15 ways to partition into 3 unordered pairs.
    # We check if any partition works.
    
    # But iterating over all C(n,6) * 15 is too slow for n=29.
    
    # Better approach: K_{2,2,2} is the complement of a perfect matching on 6 vertices
    # within the 6-vertex subgraph. So in the col-graph, the 6 vertices induce a subgraph
    # whose complement (within those 6 vertices) is a perfect matching (3 disjoint edges).
    
    # Alternative: K_{2,2,2} contains K_{2,2} as a subgraph. In fact, K_{2,2,2} can be
    # characterized as: there exist two disjoint edges (a,b) and (c,d) in the col-graph
    # such that a,b,c,d form a K_{2,2} in the col-graph AND there exist two more vertices
    # e,f that are connected to all of a,b,c,d in the col-graph, and e-f is NOT an edge
    # in the col-graph (or rather, e and f are in the same part).
    
    # Let me think differently. K_{2,2,2} has 6 vertices. Let's label them 0,1,2,3,4,5
    # with parts {0,1}, {2,3}, {4,5}. The non-edges are (0,1), (2,3), (4,5).
    # So in the col-graph on these 6 vertices, exactly 3 non-edges exist, and they form
    # a perfect matching.
    
    # So: a monochromatic K_{2,2,2} in color col exists iff there exist 6 distinct vertices
    # such that the induced subgraph in the col-graph has exactly 3 missing edges, and
    # those 3 missing edges form a perfect matching.
    
    # For n up to 31, C(31,6) = 736281, times checking 3 missing edges... this might be
    # feasible with bit operations.
    
    # Let's use bitmasks. For each color col, represent adjacency as a list of bitmasks.
    # adj_col[i] is a bitmask of vertices j where edge (i,j) has color col.
    
    # For 6 vertices v0..v5, the induced subgraph in col-graph has edge (vi,vj) present
    # iff adj_col[vi] has bit vj set.
    # We need exactly 3 missing edges forming a perfect matching.
    # The 3 missing edges must be a perfect matching: (v0,v1), (v2,v3), (v4,v5) or some
    # permutation. There are 15 matchings on 6 vertices.
    
    # For a given matching M = {(a,b),(c,d),(e,f)}, the condition is:
    # - Edges (a,b), (c,d), (e,f) are NOT in col-graph (i.e., they have the other color)
    # - All other 9 edges among the 6 vertices ARE in col-graph.
    
    # So for each color col, for each 6-set, check if there's a perfect matching where
    # the 3 matched pairs are non-edges in col-graph and all cross-pairs are edges.
    
    # This is equivalent to: in the col-graph, the 6 vertices induce a subgraph that is
    # K_6 minus a perfect matching.
    
    # Let me just do a direct check with early termination.
    
    for col in [0, 1]:
        # Build adjacency bitmask for color col
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if color_matrix[i][j] == col:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # For each 6-tuple, check if it forms K_{2,2,2} in color col
        # K_{2,2,2} in col means: 6 vertices, partitioned into 3 pairs, all cross-edges in col.
        # Equivalently: the non-edges in the col-induced subgraph are exactly a perfect matching.
        
        # Iterate over all 6-combinations
        for combo in itertools.combinations(range(n), 6):
            v0, v1, v2, v3, v4, v5 = combo
            # Check all 15 perfect matchings
            # Matching 1: (v0,v1),(v2,v3),(v4,v5)
            # We need: edges (v0,v1),(v2,v3),(v4,v5) NOT in col
            # and all other 9 edges IN col.
            
            # Define helper: edge_in_col(a, b)
            # For speed, precompute for this combo
            e = {}
            verts = [v0, v1, v2, v3, v4, v5]
            for a in range(6):
                for b in range(a+1, 6):
                    va, vb = verts[a], verts[b]
                    e[(a,b)] = (adj[va] >> vb) & 1
            
            # 15 perfect matchings on 6 elements
            # Represented as list of 3 pairs
            matchings = [
                [(0,1),(2,3),(4,5)],
                [(0,1),(2,4),(3,5)],
                [(0,1),(2,5),(3,4)],
                [(0,2),(1,3),(4,5)],
                [(0,2),(1,4),(3,5)],
                [(0,2),(1,5),(3,4)],
                [(0,3),(1,2),(4,5)],
                [(0,3),(1,4),(2,5)],
                [(0,3),(1,5),(2,4)],
                [(0,4),(1,2),(3,5)],
                [(0,4),(1,3),(2,5)],
                [(0,4),(1,5),(2,3)],
                [(0,5),(1,2),(3,4)],
                [(0,5),(1,3),(2,4)],
                [(0,5),(1,4),(2,3)],
            ]
            
            found = False
            for m in matchings:
                # Check: the 3 matched pairs are non-edges, all other 9 are edges
                ok = True
                matched_pairs = set(m)
                for a, b in m:
                    if e[(a,b)] == 1:
                        ok = False
                        break
                if ok:
                    for a in range(6):
                        for b in range(a+1, 6):
                            if (a,b) not in matched_pairs and (b,a) not in matched_pairs:
                                if e[(a,b)] == 0:
                                    ok = False
                                    break
                        if not ok:
                            break
                if ok:
                    found = True
                    break
            if found:
                return True
    
    return False


def _validate(colors, n):
    """Validate that neither color class contains K_{2,2,2}."""
    color_matrix = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            color_matrix[i][j] = colors[idx]
            color_matrix[j][i] = colors[idx]
            idx += 1
    return not _has_mono_k222(color_matrix, n)


def _paley_29():
    """Construct a known valid 29-vertex coloring using Paley-type construction.
    
    We use the Paley graph on 29 vertices (29 is a prime ≡ 1 mod 4).
    Color 0: edges where (j-i) mod 29 is a quadratic residue (nonzero).
    Color 1: edges where (j-i) mod 29 is a quadratic non-residue.
    
    The Paley graph on 29 vertices is strongly regular with parameters (29, 14, 6, 7).
    We need to verify it has no monochromatic K_{2,2,2}.
    """
    p = 29
    squares = set()
    for x in range(1, p):
        squares.add((x*x) % p)
    
    colors = []
    for i in range(p):
        for j in range(i+1, p):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
    
    return p, colors


def _try_extend(base_n, base_colors, max_n=31, deadline=None):
    """Try to extend a valid coloring from base_n to larger n by adding vertices one at a time.
    
    For each new vertex, try all 2^(base_n) color assignments (or use greedy/heuristic)
    and check if the result is still valid.
    """
    n = base_n
    colors = list(base_colors)
    
    # Build color matrix for current n
    def build_matrix(cols, nn):
        mat = [[0]*nn for _ in range(nn)]
        idx = 0
        for i in range(nn):
            for j in range(i+1, nn):
                mat[i][j] = cols[idx]
                mat[j][i] = cols[idx]
                idx += 1
        return mat
    
    mat = build_matrix(colors, n)
    
    # Precompute: for each pair (i,j) and color col, the set of common neighbors in col
    # This helps with the K_{2,2,2} check when adding a new vertex.
    
    # When adding vertex n with color assignments c[i] for i in 0..n-1:
    # A monochromatic K_{2,2,2} involving the new vertex n must have n in one of the 3 parts.
    # Case 1: n is in part A = {n, a} where a is some existing vertex.
    #   Then we need parts B={b1,b2} and C={c1,c2} from existing vertices such that:
    #   - edge(n,a) has color col
    #   - all edges from {n,a} to {b1,b2} have color col
    #   - all edges from {n,a} to {c1,c2} have color col
    #   - all edges from {b1,b2} to {c1,c2} have color col
    #   - edge(b1,b2) has NOT color col, edge(c1,c2) has NOT color col
    #   (Wait, in K_{2,2,2}, edges within a part are NOT present. So edge(a,n) IS present,
    #    edge(b1,b2) is NOT present, edge(c1,c2) is NOT present.)
    
    # This is complex. Let's just do the full check for now, but optimize.
    
    # For greedy extension, try each new vertex with all 2^n colorings (too many for n=29).
    # Instead, use a heuristic: try to find a coloring for the new vertex that avoids
    # creating K_{2,2,2}.
    
    # For now, let's try a simpler approach: random search for extension.
    import random
    
    best_n = n
    best_colors = list(colors)
    
    # Try to add vertices one by one
    for target_n in range(n+1, max_n+1):
        if deadline and time.time() > deadline:
            break
        
        found = False
        # Try random assignments for the new vertex
        new_vertex_idx = n  # 0-indexed, the new vertex is at position n
        
        # Number of edges from new vertex to existing vertices
        num_edges = target_n  # edges from new vertex to 0..target_n-1... wait
        
        # Actually, when we go from n to n+1, the new vertex is n (0-indexed).
        # It has edges to vertices 0..n-1, so n edges.
        # Total edges in K_{n+1} is (n+1)*n/2.
        # We have n edges from new vertex, and the rest are already in colors.
        
        # Try up to some number of random assignments
        max_trials = 5000
        for trial in range(max_trials):
            if deadline and time.time() > deadline:
                break
            
            # Random coloring for new vertex's edges
            new_edge_colors = [random.randint(0, 1) for _ in range(n)]
            
            # Build full color list
            # Edges in lexicographic order: (0,1),(0,2),...,(0,n),(1,2),...,(n-1,n)
            # The new edges are (i, n) for i in 0..n-1, which come after (0,n-1) etc.
            # Actually in lex order: (0,1),(0,2),...,(0,n-1),(0,n),(1,2),...,(1,n),...,(n-1,n)
            # So the new edges are interspersed.
            
            # Let me rebuild the color list properly.
            # For K_{n+1}, edges are (i,j) for 0<=i<j<=n.
            # Existing edges: (i,j) for 0<=i<j<=n-1, which are the first n*(n-1)/2 entries.
            # New edges: (i,n) for 0<=i<=n-1, which are the remaining n entries.
            # But in lex order, (i,n) for i=0 comes right after (0,n-1), before (1,2).
            # So the ordering is NOT simply "existing then new".
            
            # Let me just build the full list.
            full_colors = []
            idx_old = 0
            for i in range(n+1):
                for j in range(i+1, n+1):
                    if j == n:
                        full_colors.append(new_edge_colors[i])
                    else:
                        full_colors.append(colors[idx_old])
                        idx_old += 1
            
            # Validate
            if _validate(full_colors, n+1):
                found = True
                best_n = n+1
                best_colors = full_colors
                colors = full_colors
                n = n+1
                break
        
        if not found:
            # Try a few more with different strategies
            # Greedy: for each edge, pick the color that minimizes the risk
            # For now, break if we can't extend
            break
    
    return best_n, best_colors


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with Paley 29
    n, colors = _paley_29()
    
    # Verify it's valid
    if not _validate(colors, n):
        # Fallback to Paley 17
        p = 17
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        colors = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        n = p
    
    # Try to extend to 30
    if n < 30 and time.time() < deadline:
        n2, colors2 = _try_extend(n, colors, 30, deadline)
        if n2 > n:
            n = n2
            colors = colors2
    
    # Try to extend to 31
    if n < 31 and time.time() < deadline:
        n3, colors3 = _try_extend(n, colors, 31, deadline)
        if n3 > n:
            n = n3
            colors = colors3
    
    # Final validation
    if not _validate(colors, n):
        # Should not happen, but fallback
        p = 17
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        colors = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        n = p
    
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
