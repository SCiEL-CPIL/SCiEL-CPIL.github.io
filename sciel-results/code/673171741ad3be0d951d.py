import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Seed solution: n=9 with a simple valid coloring
    seed_n = 9
    seed_colors = []
    for i in range(seed_n):
        for j in range(i + 1, seed_n):
            # Use distance-based coloring mod 4
            d = j - i
            seed_colors.append(d % 4)
    
    best = {'n': seed_n, 'colors': seed_colors}
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 10:
            break
        
        # Time budget: more time for larger n
        budget = 30.0 if n >= 16 else 15.0
        end_time = time.time() + budget
        
        result = _spectral_local_search(n, end_time)
        if result:
            best = result
            return result
    
    return best

def _get_edge_idx(i, j, n):
    """Get index of edge (i,j) in lexicographic order for n vertices."""
    if i > j:
        i, j = j, i
    # Number of edges before vertex i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
    # Plus offset within vertex i's edges: j - i - 1
    idx = i * (2 * n - i - 1) // 2 + (j - i - 1)
    return idx

def _build_adj_from_colors(colors, n):
    """Build adjacency bitmasks for each color from edge color list."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _find_c6(adj_c, n):
    """
    Find a C6 in color class adj_c. Returns (v0,v1,v2,v3,v4,v5) or None.
    Uses bitset operations for speed.
    """
    for v0 in range(n):
        # v0's neighbors
        n0 = adj_c[v0]
        if n0 == 0:
            continue
        
        # Iterate v1 in N(v0)
        v1_mask = n0
        while v1_mask:
            lsb1 = v1_mask & -v1_mask
            v1 = lsb1.bit_length() - 1
            v1_mask ^= lsb1
            
            if v1 <= v0:
                continue
            
            # v5 must be in N(v0), v5 != v1
            v5_mask = n0 & ~(1 << v1)
            while v5_mask:
                lsb5 = v5_mask & -v5_mask
                v5 = lsb5.bit_length() - 1
                v5_mask ^= lsb5
                
                if v5 <= v0:
                    continue
                
                # Need path v1-v2-v3-v4-v5 with v2,v3,v4 distinct from v0,v1,v5
                n1 = adj_c[v1] & ~(1 << v0) & ~(1 << v5)
                
                v2_mask = n1
                while v2_mask:
                    lsb2 = v2_mask & -v2_mask
                    v2 = lsb2.bit_length() - 1
                    v2_mask ^= lsb2
                    
                    n2 = adj_c[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v5)
                    
                    v3_mask = n2
                    while v3_mask:
                        lsb3 = v3_mask & -v3_mask
                        v3 = lsb3.bit_length() - 1
                        v3_mask ^= lsb3
                        
                        n3 = adj_c[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v5)
                        
                        v4_mask = n3
                        while v4_mask:
                            lsb4 = v4_mask & -v4_mask
                            v4 = lsb4.bit_length() - 1
                            v4_mask ^= lsb4
                            
                            # Check if v4 is connected to v5
                            if adj_c[v4] & (1 << v5):
                                return (v0, v1, v2, v3, v4, v5)
    
    return None

def _count_c6s(adj, n):
    """Count total C6s across all color classes."""
    total = 0
    for c in range(4):
        if _find_c6(adj[c], n) is not None:
            total += 1
    return total

def _spectral_initialization(n, seed):
    """
    Generate edge colors using spectral embedding of a random regular graph.
    Uses Laplacian eigenvectors and quadrant-based color assignment.
    """
    rng = np.random.RandomState(seed)
    
    # Generate a random d-regular graph
    d = min(8, n - 1)
    if d % 2 == 1:
        d -= 1
    if d < 2:
        d = 2
    
    # Build random regular graph using configuration model
    adj = np.zeros((n, n), dtype=int)
    half_d = d // 2
    
    for _ in range(half_d):
        perm = rng.permutation(n)
        for i in range(n // 2):
            u = perm[2 * i]
            v = perm[2 * i + 1]
            if u != v and adj[u, v] == 0:
                adj[u, v] = 1
                adj[v, u] = 1
    
    np.fill_diagonal(adj, 0)
    
    # Compute Laplacian
    degree = adj.sum(axis=1)
    D = np.diag(degree)
    L = D - adj.astype(float)
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Take first 4 non-trivial eigenvectors (indices 1-4)
    if n <= 4:
        # Fallback
        return [(i + j) % 4 for i in range(n) for j in range(i + 1, n)]
    
    vecs = eigvecs[:, 1:5]  # Shape (n, 4)
    
    # Random rotation to break symmetry
    R = rng.randn(4, 4)
    R, _ = np.linalg.qr(R)
    
    rotated = vecs @ R.T  # Shape (n, 4)
    
    # Assign colors based on quadrant in 4D space
    # For each edge (i,j), compute diff = rotated[j] - rotated[i]
    # Use sign pattern to assign color
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = rotated[j] - rotated[i]
            # Get sign pattern (4 bits)
            signs = np.sign(diff)
            signs[signs == 0] = 1
            
            b0 = 1 if signs[0] > 0 else 0
            b1 = 1 if signs[1] > 0 else 0
            b2 = 1 if signs[2] > 0 else 0
            b3 = 1 if signs[3] > 0 else 0
            
            # Map 4-bit pattern to color 0-3
            # Use balanced mapping
            idx = b0 * 8 + b1 * 4 + b2 * 2 + b3
            color = idx % 4
            
            edge_colors.append(color)
    
    return edge_colors

def _local_search(colors, n, deadline):
    """
    Local search to eliminate C6s.
    Strategy: Find a C6, then try recoloring one of its edges to break it.
    Use simulated annealing-like approach.
    """
    colors = list(colors)
    num_edges = len(colors)
    
    def get_adj():
        return _build_adj_from_colors(colors, n)
    
    # Initial C6 count
    adj = get_adj()
    c6_count = _count_c6s(adj, n)
    
    if c6_count == 0:
        return colors
    
    best_colors = colors[:]
    best_c6_count = c6_count
    temperature = 10.0
    cooling = 0.95
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        adj = get_adj()
        c6_count = _count_c6s(adj, n)
        
        if c6_count == 0:
            return colors
        
        if c6_count < best_c6_count:
            best_c6_count = c6_count
            best_colors = colors[:]
        
        # Find a C6 to break
        c6 = None
        for c in range(4):
            c6 = _find_c6(adj[c], n)
            if c6:
                break
        
        if c6 is None:
            break
        
        # Pick a random edge from the C6 to recolor
        cycle_edges = []
        for k in range(6):
            u, v = c6[k], c6[(k + 1) % 6]
            if u > v:
                u, v = v, u
            cycle_edges.append((u, v))
        
        u, v = rng.choice(cycle_edges)
        idx = _get_edge_idx(u, v, n)
        old_color = colors[idx]
        
        # Try a different color
        new_color = rng.choice([c for c in range(4) if c != old_color])
        colors[idx] = new_color
        
        # Evaluate new state
        new_adj = get_adj()
        new_c6_count = _count_c6s(new_adj, n)
        
        # Accept if better or with probability (simulated annealing)
        delta = new_c6_count - c6_count
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            pass  # Accept
        else:
            # Reject: revert
            colors[idx] = old_color
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    # Return best found
    if best_c6_count == 0:
        return best_colors
    return None

def _spectral_local_search(n, deadline):
    """
    Try multiple spectral initializations and local search.
    """
    rng = random.Random()
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        seed = rng.randint(0, 2**31 - 1)
        
        # Generate spectral initialization
        colors = _spectral_initialization(n, seed)
        
        # Quick check: does it have any C6s?
        adj = _build_adj_from_colors(colors, n)
        has_c6 = any(_find_c6(adj[c], n) is not None for c in range(4))
        
        if not has_c6:
            return {'n': n, 'colors': colors}
        
        # Try local search to fix C6s
        time_left = deadline - time.time()
        if time_left < 2:
            break
        
        # Give local search a portion of remaining time
        ls_deadline = time.time() + min(3.0, time_left * 0.5)
        
        result = _local_search(colors, n, ls_deadline)
        if result:
            # Verify
            adj = _build_adj_from_colors(result, n)
            if all(_find_c6(adj[c], n) is None for c in range(4)):
                return {'n': n, 'colors': result}
    
    return None
