"""Spectral-init + SA: target n=50 via Clebsch-derived construction."""
import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 110.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32 (guaranteed valid floor)."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def _seed_50():
    """
    Construct a 4-coloring of K_50 from the Clebsch graph structure.
    The Clebsch graph is the complement of the folded 5-cube (K_{32} minus perfect matching).
    We use the known fact that R(3,3,3,3) > 50 via a specific construction.
    
    Strategy: Use the Paley-type construction over GF(16) for the within-block coloring,
    and color 3 for between-block edges, but with a 50-vertex structure.
    
    Actually, let's use a simpler approach: start from the 32-vertex Greenwood-Gleason
    and extend it. Or better: use the known 50-vertex construction directly.
    
    For now, we'll use a doubling construction: take two copies of a triangle-free 3-coloring
    of K_25 (if it exists) with color 3 between. But K_25 3-coloring triangle-free doesn't exist
    (R(3,3,3) = 17).
    
    Let's use a different approach: the Clebsch graph gives a 4-coloring of K_32.
    For 50, we need a different construction. Let's use random init + SA.
    """
    return None

def _compute_index(n, i, j):
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    count = 0
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _spectral_init(n):
    """
    Spectral initialization: assign colors based on sign patterns of random
    symmetric matrix eigenvectors. This creates a structured initial coloring
    that may be closer to a valid solution than random.
    """
    m = n * (n - 1) // 2
    colors = [0] * m
    
    # Generate a random symmetric matrix
    import numpy as np
    np.random.seed(42)
    A = np.random.randn(n, n)
    A = (A + A.T) / 2.0
    
    # Get top 4 eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    # Take the 4 largest magnitude eigenvectors
    idx_sorted = np.argsort(np.abs(eigenvalues))[::-1][:4]
    top_vecs = eigenvectors[:, idx_sorted]
    
    # Assign colors based on which eigenvector has the largest |component|
    # For each edge (i,j), look at the sign pattern
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Use the sign of the dot product of vertex embeddings
            vi = top_vecs[i]
            vj = top_vecs[j]
            # Compute 4 values: for each eigenvector, the product of components
            vals = []
            for k in range(4):
                vals.append(vi[k] * vj[k])
            # Assign color based on which value is most negative (creates separation)
            color = int(np.argmin(vals))
            colors[idx] = color
            idx += 1
    
    return colors

def _random_init(n):
    """Random initialization."""
    m = n * (n - 1) // 2
    return [random.randint(0, 3) for _ in range(m)]

def _greedy_init(n):
    """Build vertex-by-vertex, choosing the color minimizing immediate triangles."""
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                cost = (adj[c][u] & ((1 << u) - 1)).bit_count()
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
    return colors

def _sa_refine(n, colors, time_budget):
    """Simulated annealing to reduce monochromatic triangles."""
    m = n * (n - 1) // 2
    start_time = time.time()
    adj = _build_adj(colors, n)
    tri_count = _count_mono_triangles(adj, n)
    
    best_tri = tri_count
    best_colors = colors[:]
    idx_to_edge = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    while T > T_min and time.time() - start_time < time_budget and _time_left() > 2.0:
        # Multiple sweeps per temperature
        for _ in range(20):
            if time.time() - start_time > time_budget or _time_left() < 2.0:
                break
            ri = random.randint(0, m - 1)
            old_c = colors[ri]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            u, v = idx_to_edge[ri]
            removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
            added = (adj[new_c][u] & adj[new_c][v]).bit_count()
            delta = added - removed
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                colors[ri] = new_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                tri_count += delta
                
                if tri_count < best_tri:
                    best_tri = tri_count
                    best_colors = colors[:]
                    if tri_count == 0:
                        return best_colors, 0
        T *= cooling
    
    return best_colors, best_tri

def construct():
    floor_colors = _seed_32()
    floor_n = 32
    
    # Try spectral + SA for n=50, 40, 36
    for n in (50, 40, 36):
        if _time_left() < 15.0:
            break
        budget = min(30.0, _time_left() - 5.0)
        if budget <= 0:
            break
        
        # Try multiple initializations
        best_tri = float('inf')
        best_colors = None
        
        # Spectral init
        colors = _spectral_init(n)
        colors, tri = _sa_refine(n, colors, budget / 3.0)
        if tri < best_tri:
            best_tri = tri
            best_colors = colors[:]
        if tri == 0:
            return {'n': n, 'colors': best_colors}
        
        if _time_left() < 10.0:
            break
        
        # Greedy init
        colors = _greedy_init(n)
        colors, tri = _sa_refine(n, colors, budget / 3.0)
        if tri < best_tri:
            best_tri = tri
            best_colors = colors[:]
        if tri == 0:
            return {'n': n, 'colors': best_colors}
        
        if _time_left() < 10.0:
            break
        
        # Random init
        colors = _random_init(n)
        colors, tri = _sa_refine(n, colors, budget / 3.0)
        if tri < best_tri:
            best_tri = tri
            best_colors = colors[:]
        if tri == 0:
            return {'n': n, 'colors': best_colors}
    
    return {'n': floor_n, 'colors': floor_colors}
