import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # 1. Define the n=32 coloring (Greenwood-Gleason doubled)
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
        
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    base_n = 32
    base_colors = colors_32
    
    # 2. Extend to n=50
    target_n = 50
    # Initialize colors for K_50
    # We store colors in a list of lists or a flat list.
    # Flat list is easier for the interface.
    # Index: i * (n-1) - i*(i+1)//2 + (j-i-1) ? 
    # Simpler: Use a 2D array for internal processing, flatten at the end.
    
    n = target_n
    # color_matrix[i][j] for i < j
    color_matrix = np.zeros((n, n), dtype=np.uint8)
    
    # Fill base 32
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            color_matrix[i, j] = base_colors[idx]
            color_matrix[j, i] = base_colors[idx]
            idx += 1
            
    # Fill remaining edges randomly
    for i in range(32, n):
        for j in range(i + 1, n):
            c = random.randint(0, 3)
            color_matrix[i, j] = c
            color_matrix[j, i] = c
    # Edges between base and new
    for i in range(32):
        for j in range(32, n):
            c = random.randint(0, 3)
            color_matrix[i, j] = c
            color_matrix[j, i] = c

    # 3. Simulated Annealing
    # Objective: Minimize number of monochromatic triangles
    
    def count_triangles(cm):
        # cm is n x n
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = cm[i, j]
                # For each k > j, check if (i,k) and (j,k) are c
                # This is O(n^3). Too slow for frequent calls.
                # We need incremental updates.
                pass
        return count

    # Incremental update
    # State: color_matrix
    # Current triangle count: tri_count
    
    # Initial count
    tri_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = color_matrix[i, j]
            for k in range(j + 1, n):
                if color_matrix[i, k] == c and color_matrix[j, k] == c:
                    tri_count += 1
                    # Also triangles where i,j,k are not ordered?
                    # The loop i<j<k covers all unique triangles exactly once.
                    # So this is correct.
                    
    best_tri_count = tri_count
    best_color_matrix = color_matrix.copy()
    
    T = 1.0
    T_min = 0.001
    alpha = 0.999
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random edge (u, v)
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        old_c = color_matrix[u, v]
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
            
        # Calculate delta in triangle count
        delta = 0
        for w in range(n):
            if w == u or w == v:
                continue
            # Triangle (u, v, w)
            # Colors: (u,v), (u,w), (v,w)
            cuw = color_matrix[u, w]
            cvw = color_matrix[v, w]
            
            # Old state: (old_c, cuw, cvw)
            if old_c == cuw and old_c == cvw:
                delta -= 1  # Removed a mono triangle
            
            # New state: (new_c, cuw, cvw)
            if new_c == cuw and new_c == cvw:
                delta += 1  # Added a mono triangle
                
        new_tri_count = tri_count + delta
        
        # Accept if better or with probability exp(-delta/T) if delta > 0
        if delta < 0 or random.random() < np.exp(-delta / T):
            color_matrix[u, v] = new_c
            color_matrix[v, u] = new_c
            tri_count = new_tri_count
            
            if tri_count < best_tri_count:
                best_tri_count = tri_count
                best_color_matrix = color_matrix.copy()
                if best_tri_count == 0:
                    break
                    
        T *= alpha
        
    # 4. Return result
    if best_tri_count == 0:
        # Flatten best_color_matrix
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(best_color_matrix[i, j]))
        return {'n': n, 'colors': colors}
    else:
        # Return base n=32
        return {'n': 32, 'colors': base_colors}
