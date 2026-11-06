"""
Constructs a 2-coloring of K_n without a monochromatic K_{2,2,2}.
The maximum n is 29, achieved by the Paley graph on 29 vertices.
"""
import time

def _has_k222(adj, n):
    """
    Check if graph with adjacency bitmasks adj on n vertices contains K_{2,2,2} as a subgraph.
    K_{2,2,2} is a complete tripartite graph with parts of size 2.
    It contains two vertices u, v in one part, and 4 vertices in the other two parts that are all connected to u and v.
    Specifically, if u and v are adjacent, let C = N(u) ∩ N(v). If C contains a K_{2,2} subgraph, then we have K_{2,2,2}.
    A set of 4 vertices induces K_{2,2} if there exist two disjoint pairs (a,b) and (c,d) such that a-c, a-d, b-c, b-d are edges.
    Equivalently, in the induced subgraph on C, there exist two vertices x,y such that |N(x) ∩ N(y) ∩ C| >= 2.
    """
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            # Common neighbors of a and b
            cn_mask = adj[a] & adj[b]
            # Exclude a and b themselves
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            
            # Need at least 4 common neighbors to form the other two parts of K_{2,2,2}
            if cn_mask.bit_count() < 4:
                continue
            
            cn_list = [i for i in range(n) if (cn_mask >> i) & 1]
            
            # Check if the induced subgraph on cn_list contains K_{2,2}
            # This happens if there are two vertices u, v in cn_list with at least 2 common neighbors in cn_list
            cn_bitset = cn_mask
            for i in range(len(cn_list)):
                u = cn_list[i]
                adj_u = adj[u]
                for j in range(i + 1, len(cn_list)):
                    v = cn_list[j]
                    common = adj_u & adj[v] & cn_bitset
                    if common.bit_count() >= 2:
                        return True
    return False

def _build_adj_from_colors(colors, n):
    """Build adjacency bitmask list from colors list."""
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
    return adj

def _is_valid_fast(adj, n):
    """Check both color classes are K_{2,2,2}-free. Returns True if valid."""
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    
    if _has_k222(adj, n):
        return False
    if _has_k222(comp, n):
        return False
    return True

def _get_colors_from_adj(adj, n):
    """Get colors list from adjacency bitmasks."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _paley_adj(q):
    """Generate Paley graph adjacency bitmasks on q vertices (q prime, q ≡ 1 mod 4)."""
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # The maximum n is 29. The Paley graph on 29 vertices is a valid solution.
    # We attempt to construct it and verify.
    
    # Try n=29 first
    q = 29
    adj = _paley_adj(q)
    if _is_valid_fast(adj, q):
        colors = _get_colors_from_adj(adj, q)
        return {'n': q, 'colors': colors}
    
    # Fallbacks if n=29 fails (should not happen)
    for q_try in [17, 13, 5]:
        if time.time() > deadline:
            break
        adj_t = _paley_adj(q_try)
        if _is_valid_fast(adj_t, q_try):
            colors = _get_colors_from_adj(adj_t, q_try)
            return {'n': q_try, 'colors': colors}
    
    # If all fails, return a trivial valid coloring (small n)
    return {'n': 1, 'colors': []}
