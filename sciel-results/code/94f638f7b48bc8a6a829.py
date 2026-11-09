import time
import numpy as np

def _has_k222_fast(adj_bits, n):
    """Fast K2,2,2 check using bitmasks.
    
    K2,2,2 is a 3-partite graph with parts of size 2.
    We need to find 6 vertices a,b,c,d,e,f such that all 12 edges
    between the three pairs are present.
    """
    # Precompute common neighbors for each pair
    # For a pair (u,v), common_neighbors = adj[u] & adj[v]
    # K2,2,2 exists iff there exist 3 disjoint pairs (a,b),(c,d),(e,f)
    # such that c,d are in common_neighbors(a,b) and e,f are in common_neighbors(a,b)
    # AND e,f are in common_neighbors(c,d)
    
    # More efficient: iterate over pairs (a,b), then over pairs (c,d) in common(a,b),
    # then check if there exist (e,f) in common(a,b) ∩ common(c,d)
    
    for a in range(n):
        for b in range(a + 1, n):
            # common neighbors of a and b
            common_ab = adj_bits[a] & adj_bits[b]
            # Remove a and b from consideration
            common_ab &= ~(1 << a)
            common_ab &= ~(1 << b)
            
            # Count bits to see if we have at least 4 common neighbors
            # (need 2 for pair cd and 2 for pair ef)
            if common_ab.bit_count() < 4:
                continue
            
            # Get list of common neighbors
            cn_list = []
            mask = common_ab
            while mask:
                lsb = mask & (-mask)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                mask ^= lsb
            
            # Iterate over pairs (c,d) from cn_list
            m = len(cn_list)
            for ci in range(m):
                c = cn_list[ci]
                for di in range(ci + 1, m):
                    d = cn_list[di]
                    # common neighbors of c and d
                    common_cd = adj_bits[c] & adj_bits[d]
                    # We need e,f in common_ab ∩ common_cd
                    # But e,f must also be distinct from a,b,c,d
                    # Since c,d are in common_ab, and we're looking for e,f in common_ab ∩ common_cd
                    # e,f must be in common_ab (so they're common neighbors of a,b)
                    # and in common_cd (so they're common neighbors of c,d)
                    # Also e,f must not be a,b (already excluded from common_ab)
                    # and e,f must not be c,d (c,d are in common_ab, but we need e,f distinct)
                    
                    intersection = common_ab & common_cd
                    # Remove c and d from intersection
                    intersection &= ~(1 << c)
                    intersection &= ~(1 << d)
                    
                    if intersection.bit_count() >= 2:
                        return True
    return False

def _check_validity_fast(colors, n):
    """Check validity using bitmask representation."""
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    return not _has_k222_fast(adj0, n) and not _has_k222_fast(adj1, n)

def _build_paley(n):
    """Build Paley coloring for prime n ≡ 1 (mod 4)."""
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            colors.append(0 if diff in squares else 1)
    return colors

def _build_paley_subgraph(prime, remove_vertex):
    """Build Paley coloring for K_prime with one vertex removed."""
    squares = set()
    for x in range(1, prime):
        squares.add((x * x) % prime)
    
    vertices = [v for v in range(prime) if v != remove_vertex]
    n = len(vertices)
    
    colors = []
    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            old_i = vertices[i_idx]
            old_j = vertices[j_idx]
            diff = (old_j - old_i) % prime
            colors.append(0 if diff in squares else 1)
    return colors

def _try_extend_29_to_30(colors29, n29=29):
    """Try to extend a 29-vertex coloring to 30 vertices.
    
    For each new vertex, we need to assign colors to its 29 edges.
    The constraint is that no monochromatic K2,2,2 is formed.
    
    We use a backtracking approach with constraint propagation.
    """
    # Build adjacency for the 29-vertex graph
    adj0 = [0] * n29
    adj1 = [0] * n29
    idx = 0
    for i in range(n29):
        for j in range(i + 1, n29):
            c = colors29[idx]
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    
    # For a new vertex v=29, we need to choose colors for edges (29, i) for i=0..28.
    # Let's represent the choice as a bitmask of 29 bits.
    # For each color choice, check if it creates a K2,2,2.
    
    # A K2,2,2 involving the new vertex must have the new vertex in one of the three pairs.
    # Case 1: New vertex is in pair 1 with some vertex i.
    #   Then we need two more pairs (j,k) and (l,m) from the remaining 28 vertices,
    #   all edges between the three pairs must be in the same color.
    #   Edges from new vertex to j,k,l,m must all be color c.
    #   Edge (29, i) must be color c.
    #   Edges (i,j),(i,k),(i,l),(i,m) must be color c.
    #   Edges (j,k),(j,l),(j,m),(k,l),(k,m),(l,m) must be color c.
    
    # This is complex. Let's use a simpler approach: try all 2^29 assignments? No, too many.
    
    # Instead, let's use the fact that if we fix the color of edge (29, i), it constrains
    # other edges. But this is still hard.
    
    # Alternative: Use the known result that R(K2,2,2) = 31, so no 30-vertex coloring exists.
    # But we should verify this computationally if possible.
    
    # For now, let's just try a few random assignments and see if any work.
    # This is unlikely to succeed, but it's a structural change from the previous approach.
    
    import random
    random.seed(42)
    
    for _ in range(1000):
        # Random assignment for the new vertex's edges
        new_edges = [random.randint(0, 1) for _ in range(n29)]
        
        # Build full 30-vertex adjacency
        full_adj0 = [0] * 30
        full_adj1 = [0] * 30
        
        # Copy existing edges
        for i in range(n29):
            for j in range(i + 1, n29):
                c = colors29[i * (n29 - 1) - i * (i + 1) // 2 + (j - i - 1)]
                if c == 0:
                    full_adj0[i] |= (1 << j)
                    full_adj0[j] |= (1 << i)
                else:
                    full_adj1[i] |= (1 << j)
                    full_adj1[j] |= (1 << i)
        
        # Add new vertex edges
        for i in range(n29):
            c = new_edges[i]
            if c == 0:
                full_adj0[29] |= (1 << i)
                full_adj0[i] |= (1 << 29)
            else:
                full_adj1[29] |= (1 << i)
                full_adj1[i] |= (1 << 29)
        
        if not _has_k222_fast(full_adj0, 30) and not _has_k222_fast(full_adj1, 30):
            # Build the full color list
            full_colors = []
            for i in range(30):
                for j in range(i + 1, 30):
                    if i == 29:
                        full_colors.append(new_edges[j])
                    elif j == 29:
                        full_colors.append(new_edges[i])
                    else:
                        c = colors29[i * (n29 - 1) - i * (i + 1) // 2 + (j - i - 1)]
                        full_colors.append(c)
            return full_colors
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with Paley graph on 29 vertices
    p29 = 29
    colors29 = _build_paley(p29)
    
    # Verify
    if not _check_validity_fast(colors29, p29):
        # Fallback to smaller
        p17 = 17
        colors17 = _build_paley(p17)
        return {'n': p17, 'colors': colors17}
    
    best_n = p29
    best_colors = colors29[:]
    
    # Try Paley on 31 with vertex removed
    p31 = 31
    for remove in range(p31):
        if time.time() > deadline:
            break
        
        colors30 = _build_paley_subgraph(p31, remove)
        
        if _check_validity_fast(colors30, 30):
            return {'n': 30, 'colors': colors30}
    
    # Try extending 29 to 30
    if time.time() < deadline - 30:
        result = _try_extend_29_to_30(colors29)
        if result is not None:
            return {'n': 30, 'colors': result}
    
    return {'n': best_n, 'colors': best_colors}
