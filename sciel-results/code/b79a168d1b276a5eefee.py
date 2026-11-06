import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 105.0  # Safety margin before 120s limit

    # First, try to find a valid 50-vertex coloring via vectorized local search
    n_target = 50
    colors = None

    def count_mono_triangles_vec(c):
        """Count monochromatic triangles in a color matrix using vectorized ops."""
        # For each color, count triangles where all 3 edges have that color
        total = 0
        for col in range(4):
            # Adjacency matrix for this color
            A = (c == col).astype(np.int64)
            np.fill_diagonal(A, 0)
            # Number of triangles = trace(A^3) / 6
            A2 = A @ A
            A3 = A2 @ A
            total += np.trace(A3)
        return total // 6

    def try_local_search(n, initial_colors, max_time):
        """Run vectorized local search to eliminate monochromatic triangles."""
        c = initial_colors.copy()
        count = count_mono_triangles_vec(c)
        
        if count == 0:
            return c, 0
        
        step = 0
        while time.time() < max_time:
            step += 1
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_color = c[i, j]
            best_delta = float('inf')
            best_new_color = old_color
            
            # For each possible new color, compute delta in monochromatic triangles
            for new_c in range(4):
                if new_c == old_color:
                    continue
                # Triangles involving edge (i,j): for each k != i,j
                # Old mono: colors[i,k]==colors[j,k]==old_color
                # New mono: colors[i,k]==colors[j,k]==new_c
                ik = c[i, :]
                jk = c[j, :]
                # Exclude i and j
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                
                old_mono = np.sum((ik[mask] == old_color) & (jk[mask] == old_color))
                new_mono = np.sum((ik[mask] == new_c) & (jk[mask] == new_c))
                delta = new_mono - old_mono
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_c
            
            # Accept if improvement, or with small probability if neutral
            if best_delta < 0 or (best_delta == 0 and random.random() < 0.1):
                c[i, j] = best_new_color
                c[j, i] = best_new_color
                count += best_delta
                
                if count == 0:
                    return c, 0
            else:
                # Occasional random restart
                if random.random() < 0.005:
                    c = np.random.randint(0, 4, size=(n, n))
                    c = (c + c.T) // 2
                    np.fill_diagonal(c, 0)
                    count = count_mono_triangles_vec(c)
                    if count == 0:
                        return c, 0
        
        return c, count

    # Try multiple random restarts for n=50
    best_n = 0
    best_colors_list = None
    
    # Try a few different initializations
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        init_colors = np.random.randint(0, 4, size=(n_target, n_target))
        init_colors = (init_colors + init_colors.T) // 2
        np.fill_diagonal(init_colors, 0)
        
        result_colors, result_count = try_local_search(
            n_target, init_colors, deadline - (time.time() - start_time)
        )
        
        if result_count == 0:
            # Found valid 50-vertex coloring
            colors_list = [int(result_colors[i, j]) for i in range(n_target) for j in range(i + 1, n_target)]
            return {'n': n_target, 'colors': colors_list}
    
    # Fallback: use the known 32-vertex Greenwood-Gleason construction
    # This is a valid 4-coloring of K_32 with no monochromatic triangles
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)
    
    colors32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors32.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors32.append(3)
    
    return {'n': 32, 'colors': colors32}
