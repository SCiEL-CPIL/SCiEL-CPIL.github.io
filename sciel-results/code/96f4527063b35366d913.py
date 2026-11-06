import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Seed solution: n=9
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    best = {'n': seed_n, 'colors': seed_colors}
    
    # Try spectral construction for larger n
    for n in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
            
        # Time budget: more time for larger n
        budget = 20.0 if n >= 16 else 10.0
        end_time = time.time() + budget
        
        res = _spectral_search(n, end_time)
        if res:
            return res
            
    return best

def _spectral_search(n, deadline):
    """
    Use spectral graph theory to derive a 4-coloring:
    1. Generate a random graph
    2. Compute top 4 eigenvectors of its Laplacian
    3. Assign colors based on quadrant signs
    4. Fix C6 violations via local search
    """
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Try multiple spectral initializations
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Generate a random d-regular-ish graph
        d = max(2, n // 3)  # moderate degree
        G = _random_graph(n, d)
        
        # Compute Laplacian eigenvectors
        L = _laplacian(G)
        eigvecs = _top_eigenvectors(L, 4)
        
        # Assign colors based on quadrant signs of top 2 eigenvectors
        colors = _assign_colors_quadrant(eigvecs, n)
        
        # Validate and fix via local search
        colors = _local_search_fix(colors, n, edges, deadline)
        
        if colors is not None:
            return {'n': n, 'colors': colors}
    
    return None

def _random_graph(n, d):
    """Generate a random graph with approximately d edges per vertex."""
    G = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        # Choose d random neighbors
        candidates = list(range(n))
        candidates.remove(i)
        random.shuffle(candidates)
        for j in candidates[:d]:
            G[i, j] = 1
            G[j, i] = 1
    return G

def _laplacian(G):
    """Compute the Laplacian matrix L = D - A."""
    D = np.diag(G.sum(axis=1))
    return D - G

def _top_eigenvectors(L, k):
    """Compute top k eigenvectors of a symmetric matrix using power iteration."""
    n = L.shape[0]
    eigvecs = []
    L_work = L.copy()
    
    for i in range(k):
        # Power iteration for largest eigenvalue
        v = np.random.randn(n)
        v /= np.linalg.norm(v)
        
        for _ in range(100):
            w = L_work @ v
            norm = np.linalg.norm(w)
            if norm < 1e-10:
                break
            v = w / norm
        
        eigvecs.append(v)
        
        # Deflate: remove this eigenvector's contribution
        L_work = L_work - np.outer(v, v) * (v @ L_work @ v)
    
    return eigvecs

def _assign_colors_quadrant(eigvecs, n):
    """
    Assign colors based on quadrant signs of first two eigenvectors.
    Color 0: (+, +)
    Color 1: (-, +)
    Color 2: (-, -)
    Color 3: (+, -)
    """
    v1 = eigvecs[0]
    v2 = eigvecs[1]
    
    colors = [-1] * (n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Use the sign pattern of the vertex pair
            # Actually, we need a coloring of edges, not vertices.
            # Let's use a different approach: assign each vertex a "type" based on quadrant,
            # then color edge (i,j) based on some function of types.
            pass
    
    # Better approach: assign each vertex a 2D vector from eigenvectors,
    # then color edge (i,j) based on the angle/direction
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Compute a hash based on the relative positions
            # Use the sign of differences
            sign1 = 1 if v1[j] - v1[i] > 0 else 0
            sign2 = 1 if v2[j] - v2[i] > 0 else 0
            color = sign1 * 2 + sign2
            colors.append(color)
    
    return colors

def _local_search_fix(colors, n, edges, deadline):
    """
    Local search to fix C6 violations.
    colors: list of 4 colors for each edge in lexicographic order
    Returns fixed colors or None if failed.
    """
    colors = colors[:]
    
    # Build adjacency
    adj = [[0] * n for _ in range(4)]
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        c = colors[idx]
        adj[c][i] |= (1 << j)
        adj[c][j] |= (1 << i)
    
    def check_c6(u, v, c):
        """Check if adding edge (u,v) with color c creates a C6."""
        nu = adj[c][u]
        nv = adj[c][v]
        
        x1_mask = nu
        while x1_mask:
            lsb = x1_mask & -x1_mask
            x1 = lsb.bit_length() - 1
            x1_mask ^= lsb
            if x1 == v:
                continue
            
            x4_mask = nv
            while x4_mask:
                lsb4 = x4_mask & -x4_mask
                x4 = lsb4.bit_length() - 1
                x4_mask ^= lsb4
                if x4 == u or x4 == x1:
                    continue
                
                forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                nx1 = adj[c][x1] & ~forbidden
                nx4 = adj[c][x4] & ~forbidden
                
                vx2 = nx1
                while vx2:
                    lsb2 = vx2 & -vx2
                    x2 = lsb2.bit_length() - 1
                    vx2 ^= lsb2
                    if adj[c][x2] & nx4:
                        return True
        return False
    
    def find_violation():
        """Find a C6 in the current coloring, return (edge_idx, new_color) or None."""
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            # Temporarily remove edge (u,v)
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            
            # Check if any color creates a C6 when (u,v) is added
            for new_c in range(4):
                if new_c == c:
                    continue
                if not check_c6(u, v, new_c):
                    # This color would be safe, but we want to find violations
                    pass
            
            # Restore
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
        
        # Actually, we need to find existing C6s in the current coloring
        # Let's check each color class for C6s
        for c in range(4):
            # Build adjacency list for color c
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask ^= lsb
            
            # Check for C6: iterate over all 6-tuples? Too slow.
            # Instead, check if any edge (u,v) is part of a C6
            # A C6 contains u-v as an edge if there's a path of length 5 from u to v avoiding u,v internally
            for i in range(n):
                for j in adj_list[i]:
                    if j > i:
                        # Check if there's a path of length 5 from i to j in G_c
                        # This is equivalent to: is there x1,x2,x3,x4 such that
                        # i-x1-x2-x3-x4-j and all distinct
                        nu = adj[c][i] & ~(1 << j)
                        nv = adj[c][j] & ~(1 << i)
                        
                        x1_mask = nu
                        while x1_mask:
                            lsb = x1_mask & -x1_mask
                            x1 = lsb.bit_length() - 1
                            x1_mask ^= lsb
                            
                            x4_mask = nv
                            while x4_mask:
                                lsb4 = x4_mask & -x4_mask
                                x4 = lsb4.bit_length() - 1
                                x4_mask ^= lsb4
                                if x4 == x1:
                                    continue
                                
                                forbidden = (1 << i) | (1 << j) | (1 << x1) | (1 << x4)
                                nx1 = adj[c][x1] & ~forbidden
                                nx4 = adj[c][x4] & ~forbidden
                                
                                vx2 = nx1
                                while vx2:
                                    lsb2 = vx2 & -vx2
                                    x2 = lsb2.bit_length() - 1
                                    vx2 ^= lsb2
                                    if adj[c][x2] & nx4:
                                        # Found a C6: i-x1-x2-x3-x4-j
                                        # Return the edge index of (i,j)
                                        return (edge_idx[(i, j)], None)
        
        return None
    
    # Local search: repeatedly find a C6 and recolor an edge in it
    for step in range(10000):
        if time.time() > deadline:
            break
        
        violation = find_violation()
        if violation is None:
            return colors
        
        edge_idx, _ = violation
        u, v = edges[edge_idx]
        old_c = colors[edge_idx]
        
        # Remove edge
        adj[old_c][u] &= ~(1 << v)
        adj[old_c][v] &= ~(1 << u)
        colors[edge_idx] = -1
        
        # Try to recolor with a different color
        for new_c in range(4):
            if new_c == old_c:
                continue
            if not check_c6(u, v, new_c):
                colors[edge_idx] = new_c
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                break
        else:
            # Couldn't recolor, restore
            colors[edge_idx] = old_c
            adj[old_c][u] |= (1 << v)
            adj[old_c][v] |= (1 << u)
            return None
    
    # Final check
    violation = find_violation()
    if violation is None:
        return colors
    return None
