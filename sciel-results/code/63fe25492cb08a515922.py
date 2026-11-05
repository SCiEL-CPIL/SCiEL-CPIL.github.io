"""Improved program: use the known Paley(29) construction for R(K2,2,2)."""

# EVOLVE-BLOCK-START
import time
import random

def _paley_colors(p=29):
    """Return the 29-vertex Paley coloring as a list of 0/1 in lexicographic order."""
    squares = {x * x % p for x in range(1, p)}
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            d = (j - i) % p
            colors.append(0 if d in squares else 1)
    return colors

def _edge_index(n, i, j):
    """Index in the lexicographic edge list for pair (i,j) with i<j."""
    if i > j:
        i, j = j, i
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _check_k222_with_new_vertex(colors29, n_old, new_colors, n_new):
    """
    Check whether adding vertex n_old with the given colors to the 29-vertex Paley graph
    creates a K2,2,2 in either color class.
    
    colors29: list of length C(n_old,2) for the 29-vertex graph
    new_colors: list of length n_old (colors of edges from n_old to 0..n_old-1)
    n_new: n_old + 1
    
    We only need to check for K2,2,2 that involve the new vertex, since the old graph is valid.
    """
    n = n_old
    # Build adjacency bitmasks for the old graph for both colors
    # adj_c[v] is a bitmask of neighbors of v in color c among vertices 0..n-1
    adj = [[0] * n for _ in range(2)]
    for v in range(n):
        adj[0][v] = 0
        adj[1][v] = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx = _edge_index(n, i, j)
            c = colors29[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    
    # For the new vertex (index n), its neighbors in each color among 0..n-1
    new_adj = [0, 0]
    for v in range(n):
        c = new_colors[v]
        new_adj[c] |= (1 << v)
    
    # Check for K2,2,2 involving the new vertex
    # A K2,2,2 has 6 vertices partitioned into 3 pairs: (a1,a2), (b1,b2), (c1,c2)
    # All edges between different pairs are the same color.
    # If the new vertex is in the K2,2,2, it belongs to one of the pairs.
    
    # Case 1: New vertex is in pair 1: (new, a2), and (b1,b2), (c1,c2) are among old vertices
    # Edges: new-b1, new-b2, new-c1, new-c2 must be color C
    #        a2-b1, a2-b2, a2-c1, a2-c2 must be color C
    #        b1-c1, b1-c2, b2-c1, b2-c2 must be color C
    
    for C in range(2):
        # new's neighbors in color C
        N_new = new_adj[C]
        if N_new == 0:
            continue
        # a2 must be an old vertex such that a2's neighbors in color C include all of {b1,b2,c1,c2}
        # But we need to find b1,b2,c1,c2 in N_new such that:
        #   - b1,b2,c1,c2 are distinct
        #   - all edges between {b1,b2} and {c1,c2} are color C
        #   - there exists a2 (old vertex) such that a2 is connected to b1,b2,c1,c2 in color C
        #     and a2 is connected to new in color C (i.e., a2 is in N_new)
        
        # Let S = N_new (vertices connected to new in color C)
        # We need b1,b2,c1,c2 in S, distinct, such that the 4x4 bipartite check passes,
        # and there exists a2 in S such that a2 is connected to b1,b2,c1,c2 in color C.
        
        S = N_new
        # Get list of vertices in S
        verts_in_S = [v for v in range(n) if (S >> v) & 1]
        if len(verts_in_S) < 4:
            continue
        
        # For each candidate a2 in S, check if there exist b1,b2,c1,c2 in S\{a2}
        # such that all cross edges between {b1,b2} and {c1,c2} are color C,
        # and a2 is connected to all of b1,b2,c1,c2 in color C.
        
        for a2 in verts_in_S:
            # a2 must be connected to new in color C (already in S)
            # Check a2's neighbors in color C among S\{a2}
            N_a2 = adj[C][a2]
            # Candidates for b1,b2,c1,c2 must be in S \ {a2} and in N_a2
            cand_mask = (S & ~(1 << a2)) & N_a2
            cand_verts = [v for v in range(n) if (cand_mask >> v) & 1]
            if len(cand_verts) < 4:
                continue
            # Now check if there exist b1,b2,c1,c2 in cand_verts such that
            # all 4 edges between {b1,b2} and {c1,c2} are color C
            # This is a small subproblem. We can iterate over pairs.
            # For efficiency, precompute for each pair (u,v) in cand_verts whether
            # they are connected in color C.
            # Actually, we just need to find two disjoint pairs {b1,b2} and {c1,c2}
            # such that all 4 cross edges exist.
            # Equivalent: find a 4-cycle in the graph induced by cand_verts with edges of color C.
            # More precisely, a K2,2 subgraph in the color-C graph on cand_verts.
            
            # Build adjacency within cand_verts for color C
            # For each u in cand_verts, adj_C_within[u] = bitmask of v in cand_verts with color C edge
            adj_within = {}
            for u in cand_verts:
                mask = 0
                for v in cand_verts:
                    if u != v and ((adj[C][u] >> v) & 1):
                        mask |= (1 << v)
                adj_within[u] = mask
            
            # Find K2,2: two disjoint pairs {b1,b2} and {c1,c2} such that
            # b1-c1, b1-c2, b2-c1, b2-c2 all exist.
            # Iterate over pairs (b1,b2) and (c1,c2)
            # This is O(m^4) in worst case but cand_verts is small.
            found = False
            m = len(cand_verts)
            for i in range(m):
                b1 = cand_verts[i]
                for j in range(i + 1, m):
                    b2 = cand_verts[j]
                    # b1 and b2 must be connected in color C? No, they don't need to be.
                    # We need c1,c2 such that all 4 cross edges exist.
                    # c1,c2 must be in cand_verts \ {b1,b2}
                    # And c1-c2 edge doesn't matter, but b1-c1, b1-c2, b2-c1, b2-c2 must exist.
                    # So c1 must be in adj_within[b1] & adj_within[b2] & cand_mask & ~(1<<b1) & ~(1<<b2)
                    # Similarly c2.
                    common = adj_within[b1] & adj_within[b2] & cand_mask & ~(1 << b1) & ~(1 << b2)
                    common_verts = [v for v in range(n) if (common >> v) & 1]
                    if len(common_verts) >= 2:
                        found = True
                        break
                if found:
                    break
            if found:
                return False  # K2,2,2 found
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Start with the known valid 29-vertex Paley coloring
    colors29 = _paley_colors(29)
    n_old = 29
    
    # Try to extend to 30 vertices
    # We need to assign colors to the 29 edges from vertex 29 to vertices 0..28
    # Use local search: start with a random assignment, then flip edges to fix violations
    
    # Precompute some data for faster checking
    # adj_c[v] for the 29-vertex graph
    adj = [[0] * n_old for _ in range(2)]
    for v in range(n_old):
        adj[0][v] = 0
        adj[1][v] = 0
    for i in range(n_old):
        for j in range(i + 1, n_old):
            idx = _edge_index(n_old, i, j)
            c = colors29[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    
    def check_valid(new_colors):
        """Check if adding vertex 29 with given colors creates K2,2,2."""
        new_adj = [0, 0]
        for v in range(n_old):
            c = new_colors[v]
            new_adj[c] |= (1 << v)
        
        for C in range(2):
            N_new = new_adj[C]
            if N_new == 0:
                continue
            verts_in_S = [v for v in range(n_old) if (N_new >> v) & 1]
            if len(verts_in_S) < 4:
                continue
            
            for a2 in verts_in_S:
                N_a2 = adj[C][a2]
                cand_mask = (N_new & ~(1 << a2)) & N_a2
                cand_verts = [v for v in range(n_old) if (cand_mask >> v) & 1]
                if len(cand_verts) < 4:
                    continue
                
                adj_within = {}
                for u in cand_verts:
                    mask = 0
                    for v in cand_verts:
                        if u != v and ((adj[C][u] >> v) & 1):
                            mask |= (1 << v)
                    adj_within[u] = mask
                
                m = len(cand_verts)
                found = False
                for i in range(m):
                    b1 = cand_verts[i]
                    for j in range(i + 1, m):
                        b2 = cand_verts[j]
                        common = adj_within[b1] & adj_within[b2] & cand_mask & ~(1 << b1) & ~(1 << b2)
                        common_verts = [v for v in range(n_old) if (common >> v) & 1]
                        if len(common_verts) >= 2:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False
        return True
    
    # Try multiple random initial assignments
    best_result = None
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Random initial assignment
        new_colors = [random.randint(0, 1) for _ in range(n_old)]
        
        if check_valid(new_colors):
            # Found a valid 30-vertex coloring
            all_colors = colors29 + new_colors
            return {'n': 30, 'colors': all_colors}
        
        # Local search: try flipping one edge at a time
        for step in range(50):
            if time.time() > deadline:
                break
            # Try flipping each edge
            improved = False
            for v in range(n_old):
                if time.time() > deadline:
                    break
                new_colors[v] ^= 1
                if check_valid(new_colors):
                    # Found valid
                    all_colors = colors29 + new_colors
                    return {'n': 30, 'colors': all_colors}
                new_colors[v] ^= 1  # revert
                improved = True
            if not improved:
                break
    
    # If we couldn't find n=30, return the known valid n=29 coloring
    return {'n': 29, 'colors': colors29}
# EVOLVE-BLOCK-END
