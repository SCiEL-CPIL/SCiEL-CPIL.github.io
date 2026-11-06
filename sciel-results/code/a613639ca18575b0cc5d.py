import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Construct GF(64)
    # Primitive polynomial x^6 + x + 1
    # Reduction rule: x^6 = x + 1
    # If bit i (i>=6) is set, XOR with x^i + x^(i-5) + x^(i-6)
    
    def mul(a, b):
        r = 0
        for i in range(6):
            if b >> i & 1:
                r ^= a << i
        # Reduce
        for i in range(11, 5, -1):
            if r >> i & 1:
                # r ^= x^i + x^(i-5) + x^(i-6)
                r ^= (1 << i) | (1 << (i-5)) | (1 << (i-6))
        return r

    # Find primitive element and log table
    # alpha = 2 (x)
    alpha = 2
    log = {}
    x = 1
    for e in range(63):
        log[x] = e
        x = mul(x, alpha)
    
    # Verify x is 1 again
    # if x != 1: print("Error: not primitive")
    
    # 2. Generate colors for K_64
    n_max = 64
    # colors[i][j] for i < j
    # We store colors in a 2D list or dict
    # For efficiency in triangle checking, a 2D list is better.
    # But n=64 is small, 64x64 list is fine.
    
    color_matrix = [[0]*n_max for _ in range(n_max)]
    
    for i in range(n_max):
        for j in range(i+1, n_max):
            # Addition in GF(2^6) is XOR
            d = i ^ j
            if d == 0:
                continue
            e = log[d]
            c = e % 4
            color_matrix[i][j] = c
            color_matrix[j][i] = c
            
    # 3. Find monochromatic triangles
    # List of tuples (i, j, k)
    triangles = []
    for i in range(n_max):
        for j in range(i+1, n_max):
            c1 = color_matrix[i][j]
            for k in range(j+1, n_max):
                c2 = color_matrix[j][k]
                c3 = color_matrix[i][k]
                if c1 == c2 == c3:
                    triangles.append((i, j, k))
                    
    # 4. Prune
    active = [True] * n_max
    # To speed up, we can just re-scan or maintain a set.
    # Given n=64, re-scanning or filtering is fast.
    # But we can be smarter.
    
    # Greedy removal
    # We need to remove vertices to cover all triangles.
    # We will repeatedly find a triangle, pick the vertex with max degree in the triangle hypergraph, and remove it.
    
    # Optimization: Pre-compute for each vertex, which triangles it is in?
    # Or just iterate.
    
    removed = set()
    remaining_triangles = list(triangles)
    
    while remaining_triangles:
        if time.time() > deadline:
            break
            
        # Pick the first triangle (or one with max potential)
        # For simplicity, pick the first one.
        tri = remaining_triangles[0]
        i, j, k = tri
        
        # If any vertex in this triangle is already removed, remove this triangle from list and continue
        if i in removed or j in removed or k in removed:
            remaining_triangles.pop(0)
            continue
            
        # Count occurrences of i, j, k in remaining_triangles
        # This is O(N) where N is number of triangles.
        # We can do this for the whole list to find the best vertex to remove globally?
        # Or just for this triangle?
        # Global greedy is better.
        
        # Let's find the vertex in {i,j,k} that appears most frequently in remaining_triangles
        # Actually, let's find the vertex in the ENTIRE remaining_triangles that has the highest degree.
        # That might be expensive if we do it from scratch every time.
        
        # Simpler: For the current triangle, check degrees of i, j, k in the current list.
        # But a vertex not in this triangle might have higher degree.
        # However, we MUST remove one of i, j, k to kill this specific triangle.
        # So we should pick the best among i, j, k.
        
        count_i = 0
        count_j = 0
        count_k = 0
        for t in remaining_triangles:
            if i in t: count_i += 1
            if j in t: count_j += 1
            if k in t: count_k += 1
            
        # Pick the one with max count
        if count_i >= count_j and count_i >= count_k:
            to_remove = i
        elif count_j >= count_i and count_j >= count_k:
            to_remove = j
        else:
            to_remove = k
            
        removed.add(to_remove)
        
        # Remove all triangles containing to_remove
        new_triangles = []
        for t in remaining_triangles:
            if to_remove not in t:
                new_triangles.append(t)
        remaining_triangles = new_triangles
        
    # 5. Construct result
    # Get active vertices
    active_vertices = [v for v in range(n_max) if v not in removed]
    n = len(active_vertices)
    
    # Map old indices to new indices 0..n-1
    # old_idx -> new_idx
    # We need to output colors for pairs (i,j) in new indexing.
    # The task requires colors for pairs (i,j) of vertices 0..n-1.
    # So we iterate new_i < new_j, get old_i, old_j, and fetch color from color_matrix.
    
    colors = []
    for i in range(n):
        old_i = active_vertices[i]
        for j in range(i+1, n):
            old_j = active_vertices[j]
            colors.append(color_matrix[old_i][old_j])
            
    return {'n': n, 'colors': colors}
