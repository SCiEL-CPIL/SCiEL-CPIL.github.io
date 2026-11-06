import time
import random
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0  # Leave 5s safety margin for the 150s limit

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _compute_index(n, i, j):
    """Compute the flat index for edge (i, j) in lexicographic order."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for each color and vertex."""
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
    """Count total monochromatic triangles."""
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

def _geometric_init(n, seed=42):
    """
    Initialize coloring using geometric embedding in 4D space.
    Vertices are placed on a 4D sphere, and edge colors are determined
    by the sign pattern of the dot product between vertex vectors.
    """
    rng = np.random.RandomState(seed)
    
    # Place n vertices on the 4D unit sphere
    points = rng.randn(n, 4).astype(np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # For each pair (i, j), compute the dot product
    # Color is determined by which quadrant/sign pattern the dot product falls into
    # We use 4 colors based on the sign of the dot product and a secondary criterion
    
    m = n * (n - 1) // 2
    colors = [0] * m
    idx = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            dot = float(np.dot(points[i], points[j]))
            
            # Map dot product to one of 4 colors
            # Use the sign and magnitude to determine color
            if dot > 0.25:
                c = 0
            elif dot > -0.25:
                # Use a secondary criterion: angle-based
                # Compute cross product magnitude as a proxy for angle
                c = 1
            elif dot > -0.6:
                c = 2
            else:
                c = 3
            
            colors[idx] = c
            idx += 1
    
    return colors

def _geometric_local_search(n, colors, time_budget, seed=0):
    """
    Local search that perturbs vertex positions in 4D space and re-derives colors.
    This is a geometric optimization approach.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    m = n * (n - 1) // 2
    
    # Initialize points from current colors (inverse mapping)
    # For now, start with random points and optimize
    points = rng.randn(n, 4).astype(np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Compute initial coloring from geometry
    def _colors_from_points(pts):
        cols = [0] * m
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dot = float(np.dot(pts[i], pts[j]))
                if dot > 0.25:
                    c = 0
                elif dot > -0.25:
                    c = 1
                elif dot > -0.6:
                    c = 2
                else:
                    c = 3
                cols[idx] = c
                idx += 1
        return cols
    
    current_colors = _colors_from_points(points)
    adj = _build_adj(current_colors, n)
    tri_count = _count_mono_triangles(adj, n)
    
    best_tri = tri_count
    best_colors = current_colors[:]
    best_points = points.copy()
    
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    while T > T_min and time.time() - start_time < time_budget and _time_left() > 5.0:
        # Perturb one random point
        vi = rng.randint(0, n)
        old_point = points[vi].copy()
        
        # Small random perturbation
        perturbation = rng.randn(4) * 0.1
        new_point = points[vi] + perturbation
        new_norm = np.linalg.norm(new_point)
        if new_norm < 1e-10:
            continue
        new_point = new_point / new_norm
        
        points[vi] = new_point
        
        # Recompute colors involving vertex vi
        new_colors = current_colors[:]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if i == vi or j == vi:
                    dot = float(np.dot(points[i], points[j]))
                    if dot > 0.25:
                        c = 0
                    elif dot > -0.25:
                        c = 1
                    elif dot > -0.6:
                        c = 2
                    else:
                        c = 3
                    new_colors[idx] = c
                idx += 1
        
        # Count triangles only for those involving vertex vi (approximate)
        # For efficiency, count all triangles but this is O(n^2) per step
        new_adj = _build_adj(new_colors, n)
        new_tri = _count_mono_triangles(new_adj, n)
        
        delta = new_tri - tri_count
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_colors = new_colors
            tri_count = new_tri
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_colors = current_colors[:]
                best_points = points.copy()
                if tri_count == 0:
                    return best_colors, 0
        else:
            # Reject
            points[vi] = old_point
        
        T *= cooling
        
    return best_colors, best_tri

def _hybrid_search(n, time_budget):
    """
    Hybrid approach: combine geometric initialization with standard local search.
    """
    start_time = time.time()
    best_colors = None
    best_tri = float('inf')
    
    # Try multiple geometric seeds
    for seed in range(5):
        if time.time() - start_time > time_budget * 0.6 or _time_left() < 10.0:
            break
        
        colors = _geometric_init(n, seed=seed)
        colors, tri = _geometric_local_search(n, colors, time_budget * 0.4, seed=seed)
        
        if tri < best_tri:
            best_tri = tri
            best_colors = colors[:]
            if tri == 0:
                return best_colors, 0
    
    return best_colors, best_tri

def construct():
    """
    Main function to find the largest n with a valid 4-coloring of K_n
    with no monochromatic triangles.
    """
    # Try n=50 first using geometric approach
    n_target = 50
    time_budget = 40.0
    
    if _time_left() > time_budget + 10.0:
        colors, tri = _hybrid_search(n_target, time_budget)
        if tri == 0:
            return {'n': n_target, 'colors': colors}
    
    # If geometric approach failed, fall back to standard SA with n=50
    if _time_left() > 25.0:
        # Use a simple random init + SA
        m = n_target * (n_target - 1) // 2
        colors = [random.randint(0, 3) for _ in range(m)]
        adj = _build_adj(colors, n_target)
        tri_count = _count_mono_triangles(adj, n_target)
        
        best_tri = tri_count
        best_colors = colors[:]
        
        idx_to_edge = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                idx_to_edge.append((i, j))
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        sa_start = time.time()
        
        while T > T_min and time.time() - sa_start < 20.0 and _time_left() > 5.0:
            for _ in range(50):
                if time.time() - sa_start > 20.0 or _time_left() < 5.0:
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
                            return {'n': n_target, 'colors': best_colors}
            T *= cooling
        
        if best_tri == 0:
            return {'n': n_target, 'colors': best_colors}
    
    # Try n=40
    if _time_left() > 15.0:
        n_target = 40
        m = n_target * (n_target - 1) // 2
        for _ in range(3):
            colors = [random.randint(0, 3) for _ in range(m)]
            adj = _build_adj(colors, n_target)
            tri_count = _count_mono_triangles(adj, n_target)
            
            best_tri = tri_count
            best_colors = colors[:]
            
            idx_to_edge = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    idx_to_edge.append((i, j))
            
            T = 10.0
            T_min = 0.01
            cooling = 0.999
            sa_start = time.time()
            budget = 5.0
            
            while T > T_min and time.time() - sa_start < budget and _time_left() > 5.0:
                for _ in range(50):
                    if time.time() - sa_start > budget or _time_left() < 5.0:
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
                                return {'n': n_target, 'colors': best_colors}
                T *= cooling
            
            if best_tri == 0:
                return {'n': n_target, 'colors': best_colors}
    
    # Try n=32 (should be very easy)
    if _time_left() > 10.0:
        n_target = 32
        m = n_target * (n_target - 1) // 2
        for _ in range(5):
            colors = [random.randint(0, 3) for _ in range(m)]
            adj = _build_adj(colors, n_target)
            tri_count = _count_mono_triangles(adj, n_target)
            
            best_tri = tri_count
            best_colors = colors[:]
            
            idx_to_edge = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    idx_to_edge.append((i, j))
            
            T = 10.0
            T_min = 0.01
            cooling = 0.999
            sa_start = time.time()
            budget = 3.0
            
            while T > T_min and time.time() - sa_start < budget and _time_left() > 3.0:
                for _ in range(50):
                    if time.time() - sa_start > budget or _time_left() < 3.0:
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
                                return {'n': n_target, 'colors': best_colors}
                T *= cooling
            
            if best_tri == 0:
                return {'n': n_target, 'colors': best_colors}
    
    # Fallback
    return {'n': 1, 'colors': [0]}
