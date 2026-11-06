import numpy as np
import time
import random

def _has_c6(adj, n):
    """
    Check if adjacency (list of sets) contains a C6.
    Optimized for small n.
    """
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0: continue
            for v5 in adj[v0]:
                if v5 == v1 or v5 <= v0: continue
                for v2 in adj[v1]:
                    if v2 == v0 or v2 == v5: continue
                    for v3 in adj[v2]:
                        if v3 == v0 or v3 == v1 or v3 == v5: continue
                        for v4 in adj[v3]:
                            if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v5: continue
                            if v5 in adj[v4]:
                                return True
    return False

def _build_adj_from_edge_colors(edge_colors, n):
    """Build adjacency lists for each color class from edge color list."""
    adj = [set() for _ in range(4)]
    for c in range(4):
        adj[c] = [set() for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = edge_colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    return adj

def _is_valid_coloring(edge_colors, n):
    """Check if the edge coloring has no C6 in any color class."""
    adj = _build_adj_from_edge_colors(edge_colors, n)
    for c in range(4):
        if _has_c6(adj[c], n):
            return False
    return True

def _generate_random_regular_graph(n, d, rng):
    """
    Generate a random d-regular graph on n vertices.
    Uses the configuration model with retries.
    Returns adjacency matrix.
    """
    if d >= n:
        d = n - 1
    if d % 2 != 0:
        d -= 1
    if d < 2:
        d = 2
    
    max_attempts = 100
    for attempt in range(max_attempts):
        # Configuration model: create d stubs for each vertex
        stubs = []
        for i in range(n):
            stubs.extend([i] * d)
        
        # Randomly pair stubs
        rng.shuffle(stubs)
        adj = np.zeros((n, n), dtype=int)
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            if adj[u, v] == 1:
                valid = False
                break
            adj[u, v] = 1
            adj[v, u] = 1
        
        if valid:
            np.fill_diagonal(adj, 0)
            return adj
    
    # Fallback: use a random graph with approximately d edges per vertex
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        rng.shuffle(candidates)
        for j in candidates[:d]:
            adj[i, j] = 1
            adj[j, i] = 1
    np.fill_diagonal(adj, 0)
    return adj

def _spectral_coloring(n, seed):
    """
    Generate edge colors using spectral embedding.
    Uses top 4 non-trivial eigenvectors of the Laplacian of a random regular graph.
    Assigns colors based on sign patterns of vertex embeddings.
    """
    rng = np.random.RandomState(seed)
    
    # Choose degree: aim for moderate connectivity
    d = min(max(4, n // 2), n - 1)
    if d % 2 == 1:
        d -= 1
    if d < 2:
        d = 2
    
    adj = _generate_random_regular_graph(n, d, rng)
    
    # Compute Laplacian
    degree = adj.sum(axis=1).astype(float)
    D = np.diag(degree)
    L = D - adj.astype(float)
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Take the 4 smallest non-zero eigenvalues (indices 1, 2, 3, 4)
    if n <= 5:
        # Fallback for very small n
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
    
    # Use eigenvectors 1 through 4 (skip the trivial one at index 0)
    vecs = eigvecs[:, 1:5]
    
    # Normalize each eigenvector to have unit norm
    norms = np.linalg.norm(vecs, axis=0)
    norms[norms == 0] = 1
    vecs = vecs / norms
    
    # Project vertices to R^4
    points = vecs.T  # Shape (4, n)
    
    # Apply a random orthogonal transformation to break symmetry
    R = rng.randn(4, 4)
    Q, _ = np.linalg.qr(R)
    rotated_points = Q @ points
    
    # For each vertex, compute its sign pattern in 4D
    # This gives 2^4 = 16 possible sign patterns
    # We'll use these to guide edge coloring
    
    # Strategy: For edge (i, j), look at the difference in sign patterns
    # and the geometric relationship in the 4D embedding
    
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Compute the vector difference
            diff = rotated_points[:, j] - rotated_points[:, i]
            
            # Get sign patterns
            signs_i = np.sign(rotated_points[:, i])
            signs_j = np.sign(rotated_points[:, j])
            signs_i[signs_i == 0] = 1
            signs_j[signs_j == 0] = 1
            
            # Count how many coordinates have the same sign
            same_sign_count = np.sum(signs_i == signs_j)
            
            # Also consider the magnitude of the difference in each coordinate
            # Use a weighted combination to determine color
            
            # Method: Use the angle between the two vertex vectors in the 4D space
            # Project onto a random direction and use the sign of the projection difference
            # Combined with the sign pattern agreement
            
            # Simple approach: hash the sign pattern difference and the relative magnitude
            sign_diff = np.sum(signs_i != signs_j)  # 0 to 4
            
            # Use the first two coordinates' signs of the difference
            b0 = 1 if diff[0] > 0 else 0
            b1 = 1 if diff[1] > 0 else 0
            b2 = 1 if diff[2] > 0 else 0
            b3 = 1 if diff[3] > 0 else 0
            
            # Combine sign pattern disagreement with the difference signs
            # This should create a more balanced coloring
            idx = sign_diff * 4 + (b0 * 8 + b1 * 4 + b2 * 2 + b3) % 4
            color = idx % 4
            
            edge_colors.append(color)
    
    return edge_colors

def _repair_coloring(edge_colors, n, max_steps=3000, time_limit=30.0):
    """
    Local search to remove C6s.
    Uses smarter edge selection and color assignment.
    """
    start_time = time.time()
    colors = list(edge_colors)
    
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    def build_adjs(cols):
        adjs = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = cols[get_edge_idx(i, j)]
                adjs[c][i].add(j)
                adjs[c][j].add(i)
        return adjs

    def find_c6(adj_c):
        for v0 in range(n):
            for v1 in adj_c[v0]:
                if v1 <= v0: continue
                for v5 in adj_c[v0]:
                    if v5 == v1 or v5 <= v0: continue
                    for v2 in adj_c[v1]:
                        if v2 == v0 or v2 == v5: continue
                        for v3 in adj_c[v2]:
                            if v3 == v0 or v3 == v1 or v3 == v5: continue
                            for v4 in adj_c[v3]:
                                if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v5: continue
                                if v5 in adj_c[v4]:
                                    return [v0, v1, v2, v3, v4, v5]
        return None

    def has_c6_through_edge(adj, u, v):
        """Check if edge (u,v) is part of any C6 in the graph."""
        for w1 in adj[v]:
            if w1 == u: continue
            for w2 in adj[w1]:
                if w2 == u or w2 == v: continue
                for w3 in adj[w2]:
                    if w3 == u or w3 == v or w3 == w1: continue
                    for w4 in adj[w3]:
                        if w4 == u or w4 == v or w4 == w1 or w4 == w2: continue
                        if u in adj[w4]:
                            return True
        return False

    adjs = build_adjs(colors)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Find a C6 in any color class
        c6_found = None
        c_found = -1
        for c in range(4):
            c6 = find_c6(adjs[c])
            if c6:
                c6_found = c6
                c_found = c
                break
        
        if not c6_found:
            return colors  # Valid!
            
        # We have a C6 in color c_found
        edges_in_cycle = [
            (c6_found[0], c6_found[1]),
            (c6_found[1], c6_found[2]),
            (c6_found[2], c6_found[3]),
            (c6_found[3], c6_found[4]),
            (c6_found[4], c6_found[5]),
            (c6_found[5], c6_found[0])
        ]
        
        # Try each edge in the cycle, pick the one that's easiest to recolor
        best_edge = None
        best_new_color = None
        
        for u, v in edges_in_cycle:
            idx = get_edge_idx(u, v)
            old_color = colors[idx]
            
            for new_color in range(4):
                if new_color == old_color:
                    continue
                
                # Check if recoloring this edge to new_color creates a C6
                if has_c6_through_edge(adjs[new_color], u, v):
                    continue
                
                # Also check if removing from old_color is safe (it should be, since we're removing an edge)
                # But we need to make sure old_color doesn't gain a C6 from the removal? No, removal can't create a C6.
                
                best_edge = (u, v, idx)
                best_new_color = new_color
                break
            
            if best_edge is not None:
                break
        
        if best_edge is None:
            # No safe recoloring found, try random
            u, v = random.choice(edges_in_cycle)
            idx = get_edge_idx(u, v)
            old_color = colors[idx]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                new_color = (new_color + 1) % 4
            best_edge = (u, v, idx)
            best_new_color = new_color
        else:
            u, v, idx = best_edge
        
        old_color = colors[idx]
        new_color = best_new_color
        
        # Tentatively change color
        colors[idx] = new_color
        adjs[old_color][u].remove(v)
        adjs[old_color][v].remove(u)
        adjs[new_color][u].add(v)
        adjs[new_color][v].add(u)
        
        # Verify no C6 was created in new_color
        if has_c6_through_edge(adjs[new_color], u, v):
            # Revert
            colors[idx] = old_color
            adjs[new_color][u].remove(v)
            adjs[new_color][v].remove(u)
            adjs[old_color][u].add(v)
            adjs[old_color][v].add(u)
        else:
            # Accept change - continue
            pass
    
    return colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    total_start = time.time()
    best_n = 1
    best_colors = [0]
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() - total_start > 100:
            break
        
        found = False
        # Try multiple seeds
        for seed in range(30):
            if time.time() - total_start > 100:
                break
            
            # Generate spectral coloring
            colors = _spectral_coloring(n, seed)
            
            # Repair
            colors = _repair_coloring(colors, n, max_steps=2000, time_limit=20.0)
            
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback for smaller n if 17-10 failed
    for n in range(9, 1, -1):
        if time.time() - total_start > 110:
            break
        for seed in range(20):
            if time.time() - total_start > 110:
                break
            colors = _spectral_coloring(n, seed)
            colors = _repair_coloring(colors, n, max_steps=1000, time_limit=5.0)
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}
