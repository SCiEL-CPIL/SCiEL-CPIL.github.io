def construct():
    import numpy as np
    import random
    import time

    # The task asks for a 4-coloring of K_n without monochromatic triangles.
    # The current program returns n=32 (score 1.0).
    # We aim to reach n=50 (Chung's construction) or higher.
    # A direct algebraic construction for n=50 is complex.
    # We use a fast randomized local search (Hill Climbing) with NumPy acceleration.
    
    # Target n: We will try to find a valid coloring for n=50.
    # If that fails within the time limit, we fallback to a known smaller valid coloring or the seed's 32.
    
    # Known valid coloring for n=32 (from seed) can be used as a safe fallback.
    # But we want to beat it.
    
    # Let's try to find a valid coloring for n=50.
    # If we find one, return it.
    # If not, we might try n=40, etc., or just return the seed's 32.
    # However, the seed's 32 is derived from GF(16) doubling.
    # Let's implement the local search for n=50.
    
    n_target = 50
    time_limit = 110  # seconds, leave 10s buffer
    
    # Data structure:
    # color_matrix: n x n int array, color_matrix[i][j] is color of edge (i,j). Symmetric.
    # To check triangles efficiently, we maintain for each color c, the adjacency matrix A_c.
    # A_c[i][j] = 1 if color(i,j) == c.
    # Number of triangles in color c is (1/6) * trace(A_c^3).
    # We can update triangle counts incrementally when changing an edge color.
    
    # However, maintaining A_c and updating trace(A_c^3) incrementally is tricky.
    # Alternative: Count triangles involving a specific edge (i,j) when considering a change.
    # If we change color of (i,j) from c to c', the change in triangle count for c is:
    # - Number of k such that (i,k) and (j,k) are both color c. (These triangles are broken)
    # The change in triangle count for c' is:
    # + Number of k such that (i,k) and (j,k) are both color c'. (These triangles are formed)
    
    # We want total triangles to be 0.
    
    def count_common_neighbors_color(i, j, c, color_matrix):
        # Returns number of k != i,j such that color_matrix[i][k] == c and color_matrix[j][k] == c
        # Using NumPy:
        # row_i = color_matrix[i]
        # row_j = color_matrix[j]
        # mask = (row_i == c) & (row_j == c)
        # mask[i] = 0; mask[j] = 0
        # return np.sum(mask)
        row_i = color_matrix[i]
        row_j = color_matrix[j]
        # Create mask
        # To avoid allocating large arrays repeatedly, we can use logical operations.
        # But n=50 is small.
        common = (row_i == c) & (row_j == c)
        common[i] = False
        common[j] = False
        return np.sum(common)
    
    def local_search(n, time_limit):
        # Initialize random coloring
        # Colors 0, 1, 2, 3
        color_matrix = np.random.randint(0, 4, (n, n), dtype=np.int8)
        # Make symmetric and zero diagonal
        color_matrix = (color_matrix + color_matrix.T) // 2
        np.fill_diagonal(color_matrix, -1) # -1 for diagonal, ignored
        
        # Initial total triangles
        total_triangles = 0
        # Calculate initial triangles for each color
        # For each color c, count triangles.
        # We can do this by iterating over edges or using matrix powers.
        # For n=50, matrix power is fast enough for initialization?
        # Or just sum over edges: for each edge (i,j) of color c, count common neighbors of color c.
        # This is O(n^3) roughly. 50^3 = 125,000. Very fast.
        
        def calc_total_triangles(cm):
            count = 0
            for c in range(4):
                # Build adjacency matrix for color c
                A = (cm == c).astype(np.int8)
                np.fill_diagonal(A, 0)
                # Triangles = trace(A^3) / 6
                # A^3 diagonal elements are number of triangles incident to node i in color c.
                # Sum of diagonal / 3 = number of triangles?
                # trace(A^3) = sum_i (A^3)_ii.
                # (A^3)_ii = sum_j sum_k A_ij A_jk A_ki.
                # Each triangle (i,j,k) contributes to (i,j,k), (j,k,i), etc.
                # Actually, for a symmetric 0-1 matrix with 0 diagonal, trace(A^3) is exactly the number of triangles * 6?
                # No, trace(A^3) counts the number of closed walks of length 3.
                # Each triangle {i,j,k} gives 6 closed walks: i-j-k-i, i-k-j-i, j-i-k-j, etc.
                # So trace(A^3) = 6 * (# triangles).
                # So # triangles = trace(A^3) // 6.
                A3 = A @ A @ A
                count += np.trace(A3) // 6
            return count
        
        total_triangles = calc_total_triangles(color_matrix)
        
        start_time = time.time()
        
        # Hill Climbing
        # We pick a random edge, try to change its color to reduce total triangles.
        # If total triangles == 0, we are done.
        
        # To speed up, we can maintain the triangle count per color?
        # Or just recalculate the delta for the specific edge change.
        
        # Let's maintain a list of edges to avoid checking diagonal.
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j))
        
        while time.time() - start_time < time_limit:
            if total_triangles == 0:
                return color_matrix
            
            # Pick random edge
            idx = np.random.randint(len(edges))
            i, j = edges[idx]
            c_old = color_matrix[i][j]
            
            # Try other colors
            # We want to minimize total triangles.
            best_c = c_old
            best_delta = 0 # delta in total triangles
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                
                # Calculate delta for color c_old (losing triangles)
                # Triangles lost = # common neighbors k of i,j with color c_old
                lost = count_common_neighbors_color(i, j, c_old, color_matrix)
                
                # Calculate delta for color c_new (gaining triangles)
                # Triangles gained = # common neighbors k of i,j with color c_new
                gained = count_common_neighbors_color(i, j, c_new, color_matrix)
                
                delta = gained - lost
                
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
            
            # If best_delta < 0, accept the change
            if best_delta < 0:
                # Update matrix
                color_matrix[i][j] = best_c
                color_matrix[j][i] = best_c
                total_triangles += best_delta
            else:
                # If no improvement, maybe accept a neutral move (delta=0) with some probability to escape local minima?
                # Or just break if stuck?
                # For now, let's accept delta=0 moves with low probability to allow exploration.
                if best_delta == 0 and np.random.random() < 0.1:
                    color_matrix[i][j] = best_c
                    color_matrix[j][i] = best_c
                    # total_triangles unchanged
                else:
                    # Stuck? Try a random restart?
                    # Or just continue, maybe we find a better edge later.
                    pass
            
            # Periodic restart if stuck?
            # If we haven't improved in many steps, restart.
            # Let's keep it simple first.
            
        return None # Failed to find within time limit

    # Try to find n=50
    coloring_50 = local_search(50, 100)
    if coloring_50 is not None:
        # Convert to list format
        colors = []
        for i in range(50):
            for j in range(i+1, 50):
                colors.append(int(coloring_50[i][j]))
        return {'n': 50, 'colors': colors}
    
    # If n=50 failed, try n=40? Or just fallback to seed's 32.
    # Let's try n=40 with remaining time.
    # But the seed's 32 is guaranteed.
    # Let's try to find a valid 40.
    coloring_40 = local_search(40, 10)
    if coloring_40 is not None:
        colors = []
        for i in range(40):
            for j in range(i+1, 40):
                colors.append(int(coloring_40[i][j]))
        return {'n': 40, 'colors': colors}
        
    # Fallback to seed's 32 construction (GF(16) based)
    # This is the exact code from the seed's fallback.
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
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}

