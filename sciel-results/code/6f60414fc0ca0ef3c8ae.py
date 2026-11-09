import time
import sys
import numpy as np
from itertools import combinations

def _paley_coloring(n):
    """Return colors for Paley coloring of K_n if n is a prime (or appropriate)."""
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        # For Paley graph, we need n to be a prime p ≡ 1 (mod 4)
        # Actually, for Ramsey, Paley graphs work for p ≡ 1 mod 4
        pass
    
    # Let's implement a general Paley-like construction for prime p
    # where we use quadratic residues
    p = n
    # Check if p is prime (for small numbers we can hardcode or use simple check)
    def is_prime(num):
        if num < 2:
            return False
        if num == 2:
            return True
        if num % 2 == 0:
            return False
        for i in range(3, int(num**0.5) + 1, 2):
            if num % i == 0:
                return False
        return True
    
    if not is_prime(p):
        return None
    
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            # Color 0 if diff is a nonzero quadratic residue
            if diff in squares and diff != 0:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _has_k222(adj, n):
    """
    Check if graph with adjacency matrix adj (n x n, 0/1) contains K_{2,2,2} as a subgraph.
    K_{2,2,2} is the complete 3-partite graph with parts of size 2.
    It has 6 vertices: {a1, a2, b1, b2, c1, c2} where all edges between different parts exist.
    Equivalently, it's a 6-vertex graph with degree 4 for each vertex (each vertex connects to all except its "twin" in the same part).
    
    Actually, let's think about this differently. K_{2,2,2} is the complement of 3 disjoint edges (K2 + K2 + K2).
    So a set of 6 vertices induces K_{2,2,2} if and only if the induced subgraph is the complement of a perfect matching on those 6 vertices.
    
    But we need it as a subgraph, not necessarily induced.
    So we need 6 vertices that can be partitioned into 3 pairs (A, B, C) such that:
    - All edges between A and B exist
    - All edges between A and C exist
    - All edges between B and C exist
    
    This is equivalent to: there exist 6 distinct vertices v1,v2,v3,v4,v5,v6 and a partition into 3 pairs such that the bipartite graph between each pair of parts is complete.
    
    A simpler characterization: K_{2,2,2} is isomorphic to the 6-vertex graph where each vertex has degree 4 (connected to all 4 vertices not in its pair).
    
    For subgraph detection: we need to find 6 vertices that form a K_{2,2,2} subgraph.
    
    Efficient approach: For each pair of vertices (u,v) that are NOT adjacent (or rather, for each potential "part"), we need to find 2 more pairs that are both completely connected to (u,v) and to each other.
    
    Actually, let's use the following: K_{2,2,2} contains a 4-cycle with chords... no.
    
    Let me think again. The 6 vertices are partitioned into 3 pairs: {a1,a2}, {b1,b2}, {c1,c2}.
    Edges: a1-b1, a1-b2, a2-b1, a2-b2 (complete bipartite between first two pairs)
           a1-c1, a1-c2, a2-c1, a2-c2 (complete bipartite between first and third)
           b1-c1, b1-c2, b2-c1, b2-c2 (complete bipartite between second and third)
    
    So for a fixed pair {a1,a2}, we need to find two other disjoint pairs {b1,b2} and {c1,c2} such that:
    - {b1,b2} is completely connected to {a1,a2} (i.e., both b1 and b2 are adjacent to both a1 and a2)
    - {c1,c2} is completely connected to {a1,a2}
    - {b1,b2} is completely connected to {c1,c2}
    
    Let N(a1,a2) = {w : w is adjacent to both a1 and a2, w != a1, w != a2}
    We need to find in N(a1,a2) two disjoint pairs that are also completely connected to each other.
    
    For a fixed pair (a1,a2), let S = N(a1,a2). We need to find b1,b2,c1,c2 in S, all distinct, with b1!=b2, c1!=c2, {b1,b2} != {c1,c2}, such that all 4 edges between {b1,b2} and {c1,c2} exist.
    
    This means: the induced subgraph on S must contain a K_{2,2} (complete bipartite with parts of size 2).
    
    So: for each pair (a1,a2), compute S = common neighbors. Check if S contains K_{2,2}.
    
    K_{2,2} in a graph: two disjoint pairs with all 4 cross-edges.
    For a set S, K_{2,2} exists iff there exist 4 distinct vertices x1,x2,y1,y2 in S such that x1-y1, x1-y2, x2-y1, x2-y2 are all edges.
    This is equivalent to: there exist two vertices x1,x2 in S such that they have at least 2 common neighbors in S that are not x1 or x2... wait, no.
    
    Actually, for K_{2,2}: we need two disjoint pairs. Let's say we pick two vertices u,v in S. If they have at least 2 common neighbors w1,w2 in S, then {u,v} and {w1,w2} form a K_{2,2}? No, that's not right either because u and v need to be on the same side.
    
    Let me re-examine. K_{2,2} has parts {x1,x2} and {y1,y2}. Edges: x1-y1, x1-y2, x2-y1, x2-y2.
    So we need to find 4 vertices that can be split into two pairs of 2 such that every vertex in the first pair is connected to every vertex in the second pair.
    
    For a given set S, this exists iff there exist two vertices a,b in S such that |N(a) ∩ N(b) ∩ S| >= 2.
    Because if a and b are on one side, then we need at least 2 common neighbors to form the other side.
    
    Wait, but a and b are on the same side, so they don't need to be connected to each other.
    Yes! So for S, K_{2,2} exists iff there exist a,b in S with at least 2 common neighbors in S.
    
    So the algorithm for checking K_{2,2,2} in graph G=(V,E):
    For each pair (a1,a2) in V:
        S = {w in V : w != a1, w != a2, w adjacent to a1, w adjacent to a2}
        If |S| >= 4:
            Check if there exist a,b in S with at least 2 common neighbors in S.
            If yes, return True (K_{2,2,2} found with a1,a2 as one part)
    
    This is O(n^2 * (|S|^2 * |S|)) in the worst case, which is fine for n=30.
    
    Actually, we can optimize: for each pair (a1,a2), we compute S. Then for each pair (u,v) in S, we check if |N(u) ∩ N(v) ∩ S| >= 2.
    """
    n = adj.shape[0]
    for a1 in range(n):
        for a2 in range(a1+1, n):
            # Compute S = common neighbors of a1 and a2, excluding a1 and a2
            S = []
            for w in range(n):
                if w == a1 or w == a2:
                    continue
                if adj[a1][w] == 1 and adj[a2][w] == 1:
                    S.append(w)
            
            if len(S) < 4:
                continue
            
            # Check if S contains K_{2,2}
            # i.e., exist u,v in S with at least 2 common neighbors in S
            S_set = set(S)
            found = False
            for idx, u in enumerate(S):
                for v in S[idx+1:]:
                    common = 0
                    for w in S:
                        if w != u and w != v:
                            if adj[u][w] == 1 and adj[v][w] == 1:
                                common += 1
                    if common >= 2:
                        found = True
                        break
                if found:
                    break
            if found:
                return True
    return False

def _check_validity(n, colors):
    """Check if a coloring is valid (no monochromatic K_{2,2,2})."""
    # Build adjacency matrices for both colors
    adj0 = np.zeros((n, n), dtype=np.int8)
    adj1 = np.zeros((n, n), dtype=np.int8)
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            if c == 0:
                adj0[i][j] = 1
                adj0[j][i] = 1
            else:
                adj1[i][j] = 1
                adj1[j][i] = 1
            idx += 1
    
    if _has_k222(adj0, n):
        return False
    if _has_k222(adj1, n):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin
    
    # Try to find a valid coloring for n=30 first, then n=29, etc.
    # Start with known constructions
    
    # First, try Paley coloring for n=31 and see if removing a vertex gives n=30
    # Or just try n=30 with some heuristic
    
    # Let's start with the known best: n=29 Paley
    p29_colors = _paley_coloring(29)
    if p29_colors is not None and _check_validity(29, p29_colors):
        best_n = 29
        best_colors = p29_colors
    else:
        best_n = 17
        best_colors = _paley_coloring(17)
    
    # Now try to improve to n=30
    # Strategy 1: Try Paley(31) and remove one vertex
    p31_colors = _paley_coloring(31)
    if p31_colors is not None:
        # For each vertex to remove, create a new coloring
        for remove_v in range(31):
            if time.time() > deadline - 10:
                break
            
            new_colors = []
            new_to_old = []
            for v in range(31):
                if v != remove_v:
                    new_to_old.append(v)
            
            n_new = 30
            # Build the new coloring
            new_adj = {}
            for i_new in range(n_new):
                for j_new in range(i_new+1, n_new):
                    i_old = new_to_old[i_new]
                    j_old = new_to_old[j_new]
                    # Find the index in original colors
                    # Original: pair (a,b) with a<b has index sum_{k=0}^{a-1} (p-1-k) + (b-1-a)
                    # For p=31: index = (31-1-a) + (31-1-a-1) + ... + (31-1-(a-1)) ... 
                    # Simpler: precompute a map
                    pass
            
            # Let me build a direct map
            def pair_index(n, a, b):
                """Get index of pair (a,b) with a<b in the lexicographic list."""
                idx = 0
                for x in range(n):
                    if x == a:
                        break
                    idx += (n - 1 - x)
                idx += (b - a - 1)
                return idx
            
            new_colors_list = []
            for i_new in range(n_new):
                for j_new in range(i_new+1, n_new):
                    i_old = new_to_old[i_new]
                    j_old = new_to_old[j_new]
                    a, b = min(i_old, j_old), max(i_old, j_old)
                    idx = pair_index(31, a, b)
                    new_colors_list.append(p31_colors[idx])
            
            if _check_validity(30, new_colors_list):
                return {'n': 30, 'colors': new_colors_list}
    
    # If we didn't find n=30, try other heuristics
    # Try random local search starting from n=29 and adding vertices
    
    # For now, return the best we found
    if time.time() > deadline - 5:
        return {'n': best_n, 'colors': best_colors}
    
    # Try to extend from n=29 to n=30
    if best_n == 29:
        # Start with Paley(29) and try to add vertex 29
        base_colors = best_colors[:]
        
        # Try all 2^29 possible colorings for the new vertex (29 edges)
        # That's 536 million, too many.
        # Instead, do a greedy/backtracking approach
        
        # For each assignment of the 29 edges from vertex 29 to vertices 0..28,
        # check if the resulting graph is valid.
        # Use backtracking with pruning.
        
        def try_add_vertex(base_n, base_colors, deadline):
            """Try to add one vertex to a valid coloring."""
            new_n = base_n + 1
            # The new vertex is index new_n - 1
            # It connects to vertices 0..base_n-1
            
            # We need to assign colors to these base_n edges
            # such that no monochromatic K_{2,2,2} is formed
            
            # Build adjacency for base graph
            adj0 = np.zeros((base_n, base_n), dtype=np.int8)
            adj1 = np.zeros((base_n, base_n), dtype=np.int8)
            idx = 0
            for i in range(base_n):
                for j in range(i+1, base_n):
                    c = base_colors[idx]
                    if c == 0:
                        adj0[i][j] = 1
                        adj0[j][i] = 1
                    else:
                        adj1[i][j] = 1
                        adj1[j][i] = 1
                    idx += 1
            
            # Backtracking
            assignment = [-1] * base_n  # color for edge (new_vertex, i)
            
            def backtrack(pos):
                if time.time() > deadline - 5:
                    return None
                
                if pos == base_n:
                    # Check validity
                    new_adj0 = np.zeros((new_n, new_n), dtype=np.int8)
                    new_adj1 = np.zeros((new_n, new_n), dtype=np.int8)
                    
                    # Copy base
                    for i in range(base_n):
                        for j in range(i+1, base_n):
                            if adj0[i][j]:
                                new_adj0[i][j] = 1
                                new_adj0[j][i] = 1
                            if adj1[i][j]:
                                new_adj1[i][j] = 1
                                new_adj1[j][i] = 1
                    
                    # Add new edges
                    for i in range(base_n):
                        c = assignment[i]
                        if c == 0:
                            new_adj0[base_n][i] = 1
                            new_adj0[i][base_n] = 1
                        else:
                            new_adj1[base_n][i] = 1
                            new_adj1[i][base_n] = 1
                    
                    if _has_k222(new_adj0, new_n):
                        return None
                    if _has_k222(new_adj1, new_n):
                        return None
                    
                    return assignment[:]
                
                # Try color 0
                assignment[pos] = 0
                result = backtrack(pos + 1)
                if result is not None:
                    return result
                
                # Try color 1
                assignment[pos] = 1
                result = backtrack(pos + 1)
                if result is not None:
                    return result
                
                assignment[pos] = -1
                return None
            
            return backtrack(0)
        
        result = try_add_vertex(29, best_colors, deadline)
        if result is not None:
            new_colors = best_colors + result
            return {'n': 30, 'colors': new_colors}
    
    return {'n': best_n, 'colors': best_colors}
