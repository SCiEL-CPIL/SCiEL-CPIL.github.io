"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _contains_k222_bitmask(neigh):
    """Check if graph (given as list of neighbor bitmasks) contains K2,2,2.
    
    K2,2,2 = complete tripartite K_{2,2,2}: three disjoint pairs with all 12 cross-edges.
    For each pair (a,b), S = N(a)∩N(b). If |S|>=4, check if subgraph on S contains K2,2.
    """
    n = len(neigh)
    for a in range(n):
        for b in range(a + 1, n):
            S = neigh[a] & neigh[b]
            if S.bit_count() < 4:
                continue
            # Extract vertices in S
            verts = []
            mask = S
            while mask:
                v = (mask & (-mask)).bit_length() - 1
                verts.append(v)
                mask &= mask - 1
            L = len(verts)
            # Check if induced subgraph on verts contains K2,2
            for i in range(L):
                c = verts[i]
                for j in range(i + 1, L):
                    d = verts[j]
                    T = neigh[c] & neigh[d] & S
                    if T.bit_count() >= 2:
                        return True
    return False

def _build_neigh_from_adj(adj):
    """Build neighbor bitmasks from adjacency matrix."""
    n = adj.shape[0]
    neigh = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                neigh[i] |= (1 << j)
                neigh[j] |= (1 << i)
    return neigh

def _check_valid(adj):
    """Check if neither color class contains K2,2,2."""
    n = adj.shape[0]
    neigh1 = _build_neigh_from_adj(adj)
    if _contains_k222_bitmask(neigh1):
        return False
    # Complement graph for color 0
    neigh0 = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                neigh0[i] |= (1 << j)
                neigh0[j] |= (1 << i)
    if _contains_k222_bitmask(neigh0):
        return False
    return True

def _paley_coloring(p):
    """Generate Paley coloring for prime p ≡ 1 (mod 4)."""
    squares = {x * x % p for x in range(1, p)}
    colors = []
    adj = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(i + 1, p):
            c = 1 if (j - i) % p in squares else 0
            colors.append(c)
            if c == 1:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj, colors

def _try_extend(adj_base, n_base, deadline):
    """Try to extend a valid coloring of K_n_base to K_{n_base+1} by greedy assignment.
    
    Returns (success, new_adj, new_colors) or (False, None, None).
    """
    n_new = n_base + 1
    new_adj = np.zeros((n_new, n_new), dtype=np.int8)
    # Copy base
    new_adj[:n_base, :n_base] = adj_base
    
    # For the new vertex (index n_base), assign edges to vertices 0..n_base-1
    # We need to choose color for each edge (n_base, v) such that no K2,2,2 is created.
    # Strategy: try color 0 first, then color 1, checking validity after each assignment.
    # But full validity check after each edge is expensive. Instead, use incremental check.
    
    # Simpler approach: try all 2^(n_base) assignments? Too many for n_base=29.
    # Greedy: for each vertex v in order, try color 0, check if it creates a K2,2,2
    # involving the new vertex. If yes, try color 1. If both fail, return failure.
    
    # A K2,2,2 involving new vertex w: w is in one of the three parts.
    # Case 1: w is alone in its part? No, parts have size 2.
    # So w must be paired with some vertex u. Then we need two other pairs (a,b) and (c,d)
    # such that all cross-edges exist.
    # 
    # Actually, let's just do a full check after assigning all edges. 
    # But that's only 1 check at the end. The problem is finding a valid assignment.
    #
    # Greedy heuristic: for each v, prefer the color that minimizes the number of 
    # "dangerous" configurations. But for simplicity, let's try a few random assignments.
    
    # Given time constraints, let's try a deterministic greedy:
    # For each v from 0 to n_base-1, try color 0 first.
    # After all assigned, check validity. If invalid, try flipping some edges.
    
    # Actually, the simplest: just try color 0 for all edges, check. If fail, try color 1 for all.
    # If both fail, give up on extension.
    
    # Try all-0
    new_adj_try = new_adj.copy()
    for v in range(n_base):
        new_adj_try[n_base, v] = 0
        new_adj_try[v, n_base] = 0
    # Build colors for this try
    colors_try = []
    for i in range(n_new):
        for j in range(i + 1, n_new):
            colors_try.append(int(new_adj_try[i, j]))
    if _check_valid(new_adj_try):
        return True, new_adj_try, colors_try
    
    # Try all-1
    new_adj_try = new_adj.copy()
    for v in range(n_base):
        new_adj_try[n_base, v] = 1
        new_adj_try[v, n_base] = 1
    colors_try = []
    for i in range(n_new):
        for j in range(i + 1, n_new):
            colors_try.append(int(new_adj_try[i, j]))
    if _check_valid(new_adj_try):
        return True, new_adj_try, colors_try
    
    # Try mixed: alternate
    new_adj_try = new_adj.copy()
    for v in range(n_base):
        c = v % 2
        new_adj_try[n_base, v] = c
        new_adj_try[v, n_base] = c
    colors_try = []
    for i in range(n_new):
        for j in range(i + 1, n_new):
            colors_try.append(int(new_adj_try[i, j]))
    if _check_valid(new_adj_try):
        return True, new_adj_try, colors_try
    
    return False, None, None

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
    best_result = {'n': 1, 'colors': [0]}
    
    # Try Paley 29
    if time.time() < DEADLINE:
        adj29, colors29 = _paley_coloring(29)
        if _check_valid(adj29):
            best_result = {'n': 29, 'colors': colors29}
            # Try to extend to 30
            if time.time() < DEADLINE - 30:
                success, adj30, colors30 = _try_extend(adj29, 29, DEADLINE)
                if success:
                    best_result = {'n': 30, 'colors': colors30}
            return best_result
    
    # Fallback: Paley 17
    if time.time() < DEADLINE:
        adj17, colors17 = _paley_coloring(17)
        if _check_valid(adj17):
            best_result = {'n': 17, 'colors': colors17}
            return best_result
    
    # Fallback: Paley 13
    if time.time() < DEADLINE:
        adj13, colors13 = _paley_coloring(13)
        if _check_valid(adj13):
            best_result = {'n': 13, 'colors': colors13}
            return best_result
    
    # Fallback: Paley 5
    if time.time() < DEADLINE:
        adj5, colors5 = _paley_coloring(5)
        if _check_valid(adj5):
            best_result = {'n': 5, 'colors': colors5}
            return best_result
    
    return best_result

# EVOLVE-BLOCK-END
