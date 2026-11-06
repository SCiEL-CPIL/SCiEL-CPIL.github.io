# EVOLVE-BLOCK-START
import time
import random
import sys
import math
import numpy as np

def _mul(a, b):
    """Multiply in GF(16) with primitive polynomial 0b10011."""
    r = 0
    for i in range(4):
        if b >> i & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if r >> i & 1:
            r ^= 0b10011 << (i - 4)
    return r

def _build_seed_32():
    """Greenwood-Gleason colouring of K32 (two copies of K16 with colour 3 between)."""
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def _color_index(n, i, j):
    """Index in the flat colors list for pair (i,j), i<j, n vertices."""
    return i * (i - 1) // 2 + (j - i - 1)

def _count_mono_triangles(colors, n):
    """Count monochromatic triangles."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            cij = colors[_color_index(n, i, j)]
            for k in range(j + 1, n):
                if colors[_color_index(n, i, k)] == cij and colors[_color_index(n, j, k)] == cij:
                    count += 1
    return count

def _find_mono_triangles(colors, n, limit=1000):
    """Find monochromatic triangles. Returns list of (i,j,k)."""
    tri = []
    for i in range(n):
        for j in range(i + 1, n):
            cij = colors[_color_index(n, i, j)]
            for k in range(j + 1, n):
                if len(tri) >= limit:
                    return tri
                if colors[_color_index(n, i, k)] == cij and colors[_color_index(n, j, k)] == cij:
                    tri.append((i, j, k))
    return tri

def _spectral_init(n, rng):
    """
    Spectral initialization: 
    1. Generate a random symmetric matrix A.
    2. Compute top 4 eigenvectors.
    3. For each vertex i, get its 4-dimensional embedding.
    4. Assign edge colors based on which eigenvector component has the largest absolute value difference? 
       Actually, let's use a different mapping: 
       Map each vertex to a point in R^4. The color of edge (i,j) is determined by 
       the quadrant of (v_i - v_j) in the 4D space? No, that gives 16 quadrants.
       
       Better: Use the sign pattern of the top 4 eigenvectors of a Laplacian-like matrix?
       Or simply: Assign each vertex a random vector in R^4. 
       Color(i,j) = argmax_k | (v_i[k] - v_j[k]) |? No.
       
       Let's stick to the prompt's suggestion: "sign pattern of the top four eigenvectors".
       This usually implies partitioning. But we need 4 colors.
       
       Alternative Spectral Construction for 4-Coloring:
       We want to minimize mono triangles. 
       Let's try: 
       Generate random symmetric matrix M.
       Get eigenvectors e_1, e_2, e_3, e_4.
       For each vertex i, we have a 4-vector x_i = [e_1[i], e_2[i], e_3[i], e_4[i]].
       This doesn't directly give a color for edge (i,j).
       
       Let's try a different spectral approach: 
       Use the eigenvectors of a random matrix to define a "distance" or "phase".
       
       Actually, a known construction for R(3,3,3,3) uses finite fields. 
       The "spectral" idea might be best implemented as:
       1. Random symmetric matrix A.
       2. Eigendecomposition.
       3. Use the top 4 eigenvectors to assign each vertex a "phase" in [0, 2pi) or a vector.
       4. Color(i,j) based on the relative position.
       
       Let's try: 
       Assign each vertex i a random vector v_i in R^4.
       Color(i,j) = argmin_k | v_i[k] - v_j[k] |? No.
       
       How about: 
       Color(i,j) = floor(4 * |angle(v_i, v_j)| / pi) ?
       
       Let's go with a simpler spectral heuristic that is distinct from random:
       1. Create a random symmetric matrix.
       2. Compute eigenvectors.
       3. The first 4 eigenvectors give a low-dimensional embedding.
       4. For edge (i,j), look at the 4-dimensional difference vector d = v_i - v_j.
       5. The color is the index of the component of d with the largest magnitude.
       This creates a structure where colors are spatially correlated.
    """
    M = np.random.randn(n, n)
    M = (M + M.T) / 2.0
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # Take top 4 eigenvectors (largest eigenvalues)
        # eigh returns ascending order
        top_indices = np.argsort(eigenvalues)[-4:]
        V = eigenvectors[:, top_indices]  # n x 4
    except:
        V = np.random.randn(n, 4)
    
    colors = [0] * (n * (n - 1) // 2)
    for i in range(n):
        for j in range(i + 1, n):
            diff = V[i] - V[j]
            # Color is the index of the max absolute component
            c = np.argmax(np.abs(diff))
            colors[_color_index(n, i, j)] = int(c)
            
    return colors

def _local_search(colors, n, deadline, rng):
    """
    Local search to minimize monochromatic triangles.
    Returns (colors, triangle_count).
    """
    cur_colors = colors[:]
    cur_count = _count_mono_triangles(cur_colors, n)
    best_colors = cur_colors[:]
    best_count = cur_count
    
    # Simulated Annealing parameters
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if cur_count == 0:
            break
            
        # Pick a random monochromatic triangle
        found = False
        for _ in range(100): # Try 100 times to find a mono triangle
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            k = rng.randint(0, n - 1)
            if i == j or j == k or i == k:
                continue
            a, b, c = sorted([i, j, k])
            c_ab = cur_colors[_color_index(n, a, b)]
            c_ac = cur_colors[_color_index(n, a, c)]
            c_bc = cur_colors[_color_index(n, b, c)]
            if c_ab == c_ac == c_bc:
                found = True
                break
        
        if not found:
            cur_count = _count_mono_triangles(cur_colors, n)
            if cur_count == 0:
                break
            continue

        # We found a mono triangle (a,b,c) with color c_ab
        old_c = c_ab
        # Try changing one of the edges. Randomly pick which edge.
        edge_choice = rng.randint(0, 2)
        if edge_choice == 0:
            u, v = a, b
        elif edge_choice == 1:
            u, v = a, c
        else:
            u, v = b, c
            
        new_c = (old_c + rng.randint(1, 3)) % 4
        
        # Calculate delta
        # Triangles broken: (u,v,x) where c_ux == old_c and c_vx == old_c
        # Triangles created: (u,v,x) where c_ux == new_c and c_vx == new_c
        # Note: The triangle (a,b,c) is one of the broken ones.
        
        broken = 0
        created = 0
        
        for x in range(n):
            if x == u or x == v:
                continue
            c_ux = cur_colors[_color_index(n, u, x)] if u < x else cur_colors[_color_index(n, x, u)]
            c_vx = cur_colors[_color_index(n, v, x)] if v < x else cur_colors[_color_index(n, x, v)]
            
            if c_ux == old_c and c_vx == old_c:
                broken += 1
            if c_ux == new_c and c_vx == new_c:
                created += 1
        
        delta = created - broken
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if rng.random() < 0.5:
                accept = True
        else:
            if rng.random() < temp:
                accept = True
        
        if accept:
            idx = _color_index(n, u, v)
            cur_colors[idx] = new_c
            cur_count += delta
            temp *= cooling
            if cur_count < best_count:
                best_count = cur_count
                best_colors = cur_colors[:]
                
    return best_colors, best_count

def _greedy_construct(n, deadline, rng):
    """
    Greedy randomized construction: iteratively add vertices,
    selecting edge colors to minimize immediate triangle creation.
    """
    colors = [0] * (n * (n - 1) // 2)
    
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            best_count = float('inf')
            for c in range(4):
                cnt = 0
                for w in range(u):
                    if colors[_color_index(n, w, u)] == c and colors[_color_index(n, w, v)] == c:
                        cnt += 1
                if cnt < best_count:
                    best_count = cnt
                    best_c = c
                elif cnt == best_count:
                    if rng.random() < 0.5:
                        best_c = c
            colors[_color_index(n, u, v)] = best_c
    
    tri_count = _count_mono_triangles(colors, n)
    return colors, tri_count

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Keep margin before 150s kill
    
    rng = random.Random(42)
    np.random.seed(42)
    
    best_result = {'n': 32, 'colors': _build_seed_32()}
    
    # Strategy: 
    # 1. Try n=50 with spectral init + local search.
    # 2. If fails, try greedy + local search.
    # 3. If fails, drop n.
    
    for n in range(50, 32, -1):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 5 # Limit attempts per n
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            if attempts == 1:
                # Spectral initialization
                colors = _spectral_init(n, rng)
            elif attempts == 2:
                # Greedy initialization
                colors, _ = _greedy_construct(n, deadline, rng)
            else:
                # Random initialization
                colors = [rng.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            
            # Run local search
            res_colors, res_count = _local_search(colors, n, deadline, rng)
            
            if res_count == 0:
                if n > best_result['n']:
                    best_result = {'n': n, 'colors': res_colors}
                found = True
                break
        
        if best_result['n'] == n:
            break
            
    # Verify best result
    if best_result['n'] >= 3:
        tri = _find_mono_triangles(best_result['colors'], best_result['n'], limit=1)
        if tri:
            best_result = {'n': 32, 'colors': _build_seed_32()}
            
    return best_result
# EVOLVE-BLOCK-END
