"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _paley_coloring(n):
    """Generate a Paley-type coloring for n vertices (requires n prime, n ≡ 1 mod 4).
    Returns (colors_list, adj_matrix) or (None, None) if n is not suitable."""
    if n < 2:
        return None, None
    # Check if n is prime
    if n < 2:
        return None, None
    is_prime = True
    for p in range(2, int(n**0.5) + 1):
        if n % p == 0:
            is_prime = False
            break
    if not is_prime:
        return None, None
    if n % 4 != 1:
        return None, None
    
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            c = 1 if d in squares else 0
            colors.append(c)
            adj[i][j] = c
            adj[j][i] = c
    return colors, adj

def _contains_k222_fast(adj, color_val):
    """Check if the subgraph of edges with value color_val contains K2,2,2.
    K2,2,2 is the complete tripartite graph K_{2,2,2} (octahedron).
    Uses bitmask approach for speed."""
    n = adj.shape[0]
    
    # Build neighbor bitmasks for the given color
    neigh = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and adj[i, j] == color_val:
                mask |= (1 << j)
        neigh[i] = mask
    
    # K2,2,2 = complete tripartite with parts of size 2.
    # Need 6 vertices a,b,c,d,e,f such that all 12 cross-edges exist.
    # The structure: three pairs {a,b}, {c,d}, {e,f} with all edges between pairs.
    # 
    # Algorithm: For each pair (a,b) that are NOT connected in this color (i.e., 
    # they could form one part of the tripartition), compute S = N(a) ∩ N(b).
    # Then for each pair (c,d) in S, check if N(c) ∩ N(d) ∩ S has at least 2 elements.
    #
    # Actually, in a non-induced subgraph, a and b CAN be connected. But if we want
    # K_{2,2,2} as a subgraph, the 12 edges between the three pairs must exist.
    # a-b edge is within a part, so it's not required.
    #
    # Key: For any 6 vertices forming K_{2,2,2}, there exist 3 pairs such that all
    # cross-pair edges exist. Let's fix the first pair as (a,b). Then c,d,e,f must all
    # be in N(a) ∩ N(b). And we need c~d, c~e, c~f, d~e, d~f (all edges between 
    # {c,d} and {e,f}). So {e,f} ⊆ N(c) ∩ N(d) ∩ S where S = N(a)∩N(b).
    
    for a in range(n):
        for b in range(a + 1, n):
            S = neigh[a] & neigh[b]
            # Remove a and b from S (they shouldn't be in their own neighborhoods,
            # but if a~b, then a is in N(b) and b is in N(a), but a is not in N(a))
            S &= ~(1 << a)
            S &= ~(1 << b)
            
            if bin(S).count('1') < 4:
                continue
            
            # Get list of vertices in S
            verts_in_S = []
            for v in range(n):
                if S & (1 << v):
                    verts_in_S.append(v)
            
            # For each pair (c,d) in S
            for ci in range(len(verts_in_S)):
                c = verts_in_S[ci]
                for di in range(ci + 1, len(verts_in_S)):
                    d = verts_in_S[di]
                    # Check if there exist e,f in S such that e,f ∈ N(c)∩N(d)
                    T = neigh[c] & neigh[d] & S
                    if bin(T).count('1') >= 2:
                        return True
    return False

def _check_valid(adj):
    """Check if neither color class contains K2,2,2."""
    if _contains_k222_fast(adj, 1):
        return False
    if _contains_k222_fast(adj, 0):
        return False
    return True

def _build_colors_from_adj(adj):
    """Build colors list from adjacency matrix in lexicographic order."""
    n = adj.shape[0]
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(int(adj[i, j]))
    return colors

def _try_extend_to_n(adj_base, n_base, target_n, deadline):
    """Try to extend a valid coloring of n_base to target_n via local search.
    Returns (colors, adj) for target_n if found valid, else (None, None)."""
    if n_base >= target_n:
        return None, None
    
    # Start with the base coloring and random assignments for new edges
    adj = np.zeros((target_n, target_n), dtype=np.int8)
    adj[:n_base, :n_base] = adj_base[:n_base, :n_base]
    
    # Randomly assign new edges
    rng = np.random.RandomState(42)
    for i in range(n_base, target_n):
        for j in range(i + 1, target_n):
            adj[i, j] = rng.randint(0, 2)
            adj[j, i] = adj[i, j]
    for i in range(n_base):
        for j in range(n_base, target_n):
            adj[i, j] = rng.randint(0, 2)
            adj[j, i] = adj[i, j]
    
    # Local search: flip edges to eliminate K2,2,2 violations
    for attempt in range(200):
        if time.time() > deadline:
            break
        # Check validity
        if _check_valid(adj):
            colors = _build_colors_from_adj(adj)
            return colors, adj
        # Flip a random edge
        i = rng.randint(0, target_n)
        j = rng.randint(0, target_n)
        if i != j:
            adj[i, j] = 1 - adj[i, j]
            adj[j, i] = adj[i, j]
    
    return None, None

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 35s safety margin before 150s kill
    
    best_result = {'n': 1, 'colors': []}
    
    # Try Paley constructions in order of size (largest first)
    for p in [29, 17, 13, 5]:
        if time.time() > DEADLINE:
            break
        colors, adj = _paley_coloring(p)
        if colors is not None:
            if _check_valid(adj):
                best_result = {'n': p, 'colors': colors}
                break  # Found the largest valid Paley
        # If invalid, continue to next
    
    # If we have a valid coloring with n < 29, try to extend
    if best_result['n'] < 29 and time.time() < DEADLINE:
        # Rebuild adj from best_result for extension
        n_cur = best_result['n']
        adj_cur = np.zeros((n_cur, n_cur), dtype=np.int8)
        idx = 0
        for i in range(n_cur):
            for j in range(i + 1, n_cur):
                c = best_result['colors'][idx]
                adj_cur[i, j] = c
                adj_cur[j, i] = c
                idx += 1
        
        # Try extending to 29
        ext_colors, ext_adj = _try_extend_to_n(adj_cur, n_cur, 29, DEADLINE)
        if ext_colors is not None:
            best_result = {'n': 29, 'colors': ext_colors}
    
    # Fallback: if nothing valid found, return n=1
    if best_result['n'] < 2:
        best_result = {'n': 1, 'colors': []}
    
    return best_result

# EVOLVE-BLOCK-END
