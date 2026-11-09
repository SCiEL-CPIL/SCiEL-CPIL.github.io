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

def _contains_k222(adj, color_val):
    """Check if the subgraph of edges with value color_val contains K2,2,2.
    K2,2,2 is the complete tripartite graph with parts of size 2 (octahedron).
    Uses bitmask approach for speed."""
    n = adj.shape[0]
    
    # Build neighbor bitmasks for the given color
    neigh = np.zeros(n, dtype=np.int64)
    # For each vertex i, neigh[i] has bit j set if adj[i][j] == color_val
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and adj[i, j] == color_val:
                mask |= (1 << j)
        neigh[i] = mask
    
    # For K2,2,2: need three disjoint pairs {a,b}, {c,d}, {e,f} such that
    # all cross-edges exist (12 edges total).
    # Equivalent: exist a,b,c,d,e,f with:
    #   a,b in N(c)∩N(d)∩N(e)∩N(f) -- no wait
    #   Actually: a~c, a~d, a~e, a~f, b~c, b~d, b~e, b~f,
    #             c~e, c~f, d~e, d~f
    #   (a,b not connected to each other needed? No, subgraph not induced)
    #   
    # Key insight: Let S = N(a) ∩ N(b). If |S| >= 4, then pick any 4 vertices c,d,e,f from S.
    # We need c~e, c~f, d~e, d~f, i.e., {e,f} ⊆ N(c) ∩ N(d).
    # So: for pair (a,b), compute S = N(a)∩N(b).
    # For each pair (c,d) in S, check if |N(c)∩N(d)∩S| >= 2.
    
    for a in range(n):
        for b in range(a + 1, n):
            S = neigh[a] & neigh[b]
            # S should not contain a or b (they're not neighbors of themselves)
            # But if a~b, then a is in N(b) but not N(a), so a not in S. Same for b.
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
    n = adj.shape[0]
    if _contains_k222(adj, 1):
        return False
    if _contains_k222(adj, 0):
        return False
    return True

def _local_search_extend(adj_base, n_base, target_n, deadline):
    """Try to extend a valid coloring of n_base to target_n via local search.
    Returns (colors, adj) for target_n if found valid, else (None, None)."""
    if n_base >= target_n:
        return None, None
    
    # Start with the base coloring and random assignments for new edges
    adj = np.zeros((target_n, target_n), dtype=np.int8)
    adj[:n_base, :n_base] = adj_base[:n_base, :n_base]
    
    # Randomly assign new edges
    np.random.seed(42)
    for i in range(n_base, target_n):
        for j in range(i + 1, target_n):
            adj[i, j] = np.random.randint(0, 2)
            adj[j, i] = adj[i, j]
    for i in range(n_base):
        for j in range(n_base, target_n):
            adj[i, j] = np.random.randint(0, 2)
            adj[j, i] = adj[i, j]
    
    # Local search: flip edges to eliminate K2,2,2 violations
    # This is complex; for now, just try a few random restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
        # Check validity
        if _check_valid(adj):
            # Build colors list
            colors = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    colors.append(int(adj[i, j]))
            return colors, adj
        # Flip a random edge
        i = np.random.randint(0, target_n)
        j = np.random.randint(0, target_n)
        if i != j:
            adj[i, j] = 1 - adj[i, j]
            adj[j, i] = adj[i, j]
    
    return None, None

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 35s safety margin before 150s kill
    
    best_result = {'n': 1, 'colors': [0]}
    
    # Try Paley constructions in order of size
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
        ext_colors, ext_adj = _local_search_extend(adj_cur, n_cur, 29, DEADLINE)
        if ext_colors is not None:
            best_result = {'n': 29, 'colors': ext_colors}
    
    # Fallback: if nothing valid found, return n=1
    if best_result['n'] < 2:
        best_result = {'n': 1, 'colors': []}
    
    return best_result

# EVOLVE-BLOCK-END
