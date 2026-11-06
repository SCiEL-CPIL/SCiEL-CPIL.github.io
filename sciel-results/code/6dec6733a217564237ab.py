import numpy as np
import time
import random

def _has_c6(adj, n):
    """
    Check if adjacency (list of sets) contains a C6.
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
    adj = [[set() for _ in range(n)] for _ in range(4)]
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
    """
    if d >= n:
        d = n - 1
    if d % 2 != 0:
        d -= 1
    if d < 2:
        d = 2
        
    adj = np.zeros((n, n), dtype=int)
    # Use configuration model with retries
    for _ in range(100):
        degrees = [d] * n
        half_edges = d * n // 2
        # Create list of stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * d)
        
        # Shuffle stubs
        rng.shuffle(stubs)
        
        # Pair them up
        adj_candidate = np.zeros((n, n), dtype=int)
        valid = True
        for k in range(0, len(stubs), 2):
            u = stubs[k]
            v = stubs[k+1]
            if u == v:
                valid = False
                break
            if adj_candidate[u, v] == 1:
                valid = False
                break
            adj_candidate[u, v] = 1
            adj_candidate[v, u] = 1
        
        if valid:
            return adj_candidate
            
    # Fallback: Return a cycle or complete graph if generation fails
    # For n <= 20, a cycle is C6-free if n != 6. But we want random.
    # Let's just return a random graph with fixed edge count.
    adj = np.zeros((n, n), dtype=int)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj

def _spectral_coloring(n, seed):
    """
    Generate edge colors using spectral embedding.
    """
    rng = np.random.RandomState(seed)
    
    # Generate a random graph
    # For spectral methods, regular graphs are preferred.
    # d=4 or 5 is good for n=17.
    d = 4
    if n < 5:
        d = 2
    adj = _generate_random_regular_graph(n, d, rng)
    
    # Compute Laplacian
    degree = adj.sum(axis=1)
    D = np.diag(degree)
    L = D - adj
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Take the 4 smallest non-zero eigenvalues
    # eigvals[0] is 0 (constant vector).
    # We want indices 1, 2, 3, 4.
    if n <= 4:
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
        
    # Ensure we have at least 4 non-zero eigenvalues
    # If graph is disconnected, there will be more zeros.
    # For a connected regular graph, only one zero.
    # We take eigvecs[:, 1:5]
    vecs = eigvecs[:, 1:5]
    
    # Normalize
    # Project vertices to R^4
    points = vecs.T # Shape (4, n)
    
    # Random rotation to break symmetry
    R = rng.randn(4, 4)
    R, _ = np.linalg.qr(R)
    rotated_points = R @ points
    
    # Define two random unit vectors in R^4 for projection
    # This creates a random 2D subspace
    u1 = rng.randn(4)
    u1 /= np.linalg.norm(u1)
    u2 = rng.randn(4)
    # Gram-Schmidt to make u2 orthogonal to u1
    u2 -= np.dot(u2, u1) * u1
    u2 /= np.linalg.norm(u2)
    
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = rotated_points[:, j] - rotated_points[:, i]
            x = np.dot(diff, u1)
            y = np.dot(diff, u2)
            
            # Determine quadrant
            if x >= 0 and y >= 0:
                color = 0
            elif x < 0 and y >= 0:
                color = 1
            elif x < 0 and y < 0:
                color = 2
            else:
                color = 3
                
            edge_colors.append(color)
            
    return edge_colors

def _repair_coloring(edge_colors, n, max_steps=1000):
    """
    Local search to remove C6s.
    """
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

    adjs = build_adjs(colors)
    
    for step in range(max_steps):
        c6_found = None
        c_found = -1
        for c in range(4):
            c6 = find_c6(adjs[c])
            if c6:
                c6_found = c6
                c_found = c
                break
        
        if not c6_found:
            return colors
            
        edges_in_cycle = [
            (c6_found[0], c6_found[1]),
            (c6_found[1], c6_found[2]),
            (c6_found[2], c6_found[3]),
            (c6_found[3], c6_found[4]),
            (c6_found[4], c6_found[5]),
            (c6_found[5], c6_found[0])
        ]
        
        # Try to fix by changing one edge in the cycle
        # We try all 6 edges and all 3 alternative colors
        # We pick the change that minimizes the number of new C6s?
        # For speed, we just pick a random one that doesn't immediately create a C6 through that edge.
        
        improved = False
        for u, v in edges_in_cycle:
            idx = get_edge_idx(u, v)
            old_color = colors[idx]
            for new_color in range(4):
                if new_color == old_color:
                    continue
                # Tentatively change
                colors[idx] = new_color
                adjs[old_color][u].remove(v)
                adjs[old_color][v].remove(u)
                adjs[new_color][u].add(v)
                adjs[new_color][v].add(u)
                
                # Check if this specific edge is in a C6 in new_color
                def has_c6_through_edge(adj, u, v):
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

                if not has_c6_through_edge(adjs[new_color], u, v):
                    improved = True
                    break
                else:
                    # Revert
                    colors[idx] = old_color
                    adjs[new_color][u].remove(v)
                    adjs[new_color][v].remove(u)
                    adjs[old_color][u].add(v)
                    adjs[old_color][v].add(u)
            
            if improved:
                break
                
        if not improved:
            # If we can't fix it locally, revert last change or restart?
            # For now, just break and let the outer loop try a new seed
            break

    return colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        found = False
        # Try multiple seeds
        for seed in range(100):
            colors = _spectral_coloring(n, seed)
            colors = _repair_coloring(colors, n, max_steps=500)
            
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback for smaller n
    for n in range(9, 1, -1):
        for seed in range(50):
            colors = _spectral_coloring(n, seed)
            colors = _repair_coloring(colors, n, max_steps=500)
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}
