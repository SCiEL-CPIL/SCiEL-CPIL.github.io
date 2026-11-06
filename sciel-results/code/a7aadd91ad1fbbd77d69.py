import time
import random
import numpy as np

def _build_adj_np(colors, n):
    A = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                A[i, j] = 1
                A[j, i] = 1
            idx += 1
    return A

def _has_k222_count_np(A, n):
    count = 0
    for c in range(n - 1):
        for d in range(c + 1, n):
            if A[c, d] == 0:
                continue
            common = A[:, c] & A[:, d]
            common[c] = 0
            common[d] = 0
            common_verts = np.where(common)[0]
            m_size = len(common_verts)
            if m_size < 4:
                continue
            for i in range(m_size - 1):
                a = common_verts[i]
                for j in range(i + 1, m_size):
                    b = common_verts[j]
                    neighbors = (A[:, a] & A[:, b]) & common
                    if np.sum(neighbors) >= 2:
                        count += 1
    return count

def _is_valid_fast_np(A, n):
    return _has_k222_count_np(A, n) == 0

def _local_search_np(colors, n, max_iters=10000, time_limit=30):
    start_time = time.time()
    A = _build_adj_np(colors, n)
    current_cost = _has_k222_count_np(A, n)
    
    if current_cost == 0:
        return colors, True
        
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        idx = random.randint(0, len(colors) - 1)
        
        i = 0
        base = 0
        while True:
            next_base = (i + 1) * (2 * n - i - 2) // 2
            if idx < next_base:
                break
            i += 1
            base = next_base
        j = i + 1 + (idx - base)
        
        old_val = A[i, j]
        new_val = 1 - old_val
        A[i, j] = new_val
        A[j, i] = new_val
        colors[idx] = new_val
        
        new_cost = _has_k222_count_np(A, n)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if current_cost == 0:
                return colors, True
        else:
            A[i, j] = old_val
            A[j, i] = old_val
            colors[idx] = old_val

    return colors, False

def _paley(n, p):
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _spectral_init(n, seed=42):
    """Initialize coloring from eigenvectors of Laplacian of random graph."""
    rng = np.random.RandomState(seed)
    # Create a random dense graph
    A_rand = (rng.rand(n, n) > 0.5).astype(np.int8)
    np.fill_diagonal(A_rand, 0)
    A_rand = (A_rand + A_rand.T) // 2
    
    # Compute Laplacian
    degree = A_rand.sum(axis=1).astype(float)
    D = np.diag(degree)
    L = D - A_rand.astype(float)
    
    # Get eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except:
        # Fallback to random
        return [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    
    # Use the second and third eigenvectors (Fiedler vectors)
    # to create a spectral embedding
    # vertices with similar embeddings get same color
    vec1 = eigenvectors[:, 1]  # Fiedler vector
    vec2 = eigenvectors[:, 2] if n > 2 else np.zeros(n)
    
    # Assign colors based on sign of spectral coordinates
    # For edge (i,j), color based on whether i and j are "close" in spectral space
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff1 = vec1[i] - vec1[j]
            diff2 = vec2[i] - vec2[j]
            dist = diff1 * diff1 + diff2 * diff2
            # If close in spectral space, color 0, else color 1
            # Use a threshold
            threshold = np.median([(vec1[a] - vec1[b])**2 + (vec2[a] - vec2[b])**2 
                                   for a in range(n) for b in range(a+1, n)])
            colors.append(0 if dist < threshold else 1)
    
    return colors

def _find_violation_edges_np(A, n):
    """Find edges that are part of K2,2,2 violations."""
    violation_edges = set()
    for c in range(n - 1):
        for d in range(c + 1, n):
            if A[c, d] == 0:
                continue
            common = A[:, c] & A[:, d]
            common[c] = 0
            common[d] = 0
            common_verts = np.where(common)[0]
            m_size = len(common_verts)
            if m_size < 4:
                continue
            for i in range(m_size - 1):
                a = common_verts[i]
                for j in range(i + 1, m_size):
                    b = common_verts[j]
                    neighbors = (A[:, a] & A[:, b]) & common
                    if np.sum(neighbors) >= 2:
                        # Violation found: edges (c,d), (c,a), (d,a), (c,b), (d,b),
                        # and edges among {a,b} and common neighbors
                        violation_edges.add((min(c,d), max(c,d)))
                        violation_edges.add((min(c,a), max(c,a)))
                        violation_edges.add((min(d,a), max(d,a)))
                        violation_edges.add((min(c,b), max(c,b)))
                        violation_edges.add((min(d,b), max(d,b)))
    return violation_edges

def _targeted_local_search_np(colors, n, max_iters=5000, time_limit=20):
    """Local search that targets edges in violations."""
    start_time = time.time()
    A = _build_adj_np(colors, n)
    current_cost = _has_k222_count_np(A, n)
    
    if current_cost == 0:
        return colors, True
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find violation edges
        violation_edges = _find_violation_edges_np(A, n)
        
        if not violation_edges:
            return colors, True
        
        # Pick a random violation edge to flip
        c, d = random.choice(list(violation_edges))
        idx = c * (2 * n - c - 1) // 2 + (d - c - 1)
        
        old_val = A[c, d]
        new_val = 1 - old_val
        A[c, d] = new_val
        A[d, c] = new_val
        colors[idx] = new_val
        
        new_cost = _has_k222_count_np(A, n)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if current_cost == 0:
                return colors, True
        else:
            A[c, d] = old_val
            A[d, c] = old_val
            colors[idx] = old_val

    return colors, False

def construct():
    deadline = time.time() + 140
    best_n = 0
    best_colors = None
    
    target_n = 29
    p = 29
    
    # 1. Try Paley(29)
    colors = _paley(target_n, p)
    if colors:
        A = _build_adj_np(colors, target_n)
        if _is_valid_fast_np(A, target_n):
            return {'n': target_n, 'colors': colors}
        
    # 2. Try targeted local search from Paley
    colors_copy = colors[:]
    _, success = _targeted_local_search_np(colors_copy, target_n, max_iters=5000, time_limit=40)
    if success:
        return {'n': target_n, 'colors': colors_copy}
    
    # 3. Try spectral initialization + local search
    if time.time() < deadline - 30:
        colors_spectral = _spectral_init(target_n, seed=123)
        _, success = _targeted_local_search_np(colors_spectral, target_n, max_iters=3000, time_limit=25)
        if success:
            return {'n': target_n, 'colors': colors_spectral}
    
    # 4. Random restarts with targeted local search
    for attempt in range(30):
        if time.time() > deadline - 20:
            break
        colors = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        _, success = _targeted_local_search_np(colors, target_n, max_iters=1500, time_limit=8)
        if success:
            return {'n': target_n, 'colors': colors}
    
    # Fallback: try smaller n
    for n in range(28, 10, -1):
        if time.time() > deadline:
            break
        p_candidate = None
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                p_candidate = pp
                break
        if not p_candidate:
            continue
        colors = _paley(n, p_candidate)
        if colors:
            A = _build_adj_np(colors, n)
            if _is_valid_fast_np(A, n):
                return {'n': n, 'colors': colors}
                
    return {'n': 1, 'colors': [0]}
