"""Construct a 4-coloring of K_n with no monochromatic triangles, maximizing n."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Construct the 32-vertex coloring
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
    
    n_base = 32
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            colors_base.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    # Store base colors in a 2D array for easy access
    n_total = 50
    color_mat = [[0]*n_total for _ in range(n_total)]
    
    # Fill base colors
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = colors_base[idx]
            color_mat[i][j] = c
            color_mat[j][i] = c
            idx += 1
            
    # 2. Add new vertices 32..49
    # Initialize edges involving new vertices randomly
    for i in range(n_base):
        for j in range(n_base, n_total):
            c = random.randint(0, 3)
            color_mat[i][j] = c
            color_mat[j][i] = c
            
    for i in range(n_base, n_total):
        for j in range(i + 1, n_total):
            c = random.randint(0, 3)
            color_mat[i][j] = c
            color_mat[j][i] = c
            
    # 3. Define cost function: number of monochromatic triangles involving at least one new vertex
    def count_bad_triangles():
        count = 0
        for k in range(32, n_total):
            for j in range(k):
                for i in range(j):
                    c1 = color_mat[i][j]
                    c2 = color_mat[j][k]
                    c3 = color_mat[i][k]
                    if c1 == c2 == c3:
                        count += 1
        return count

    current_cost = count_bad_triangles()
    best_cost = current_cost
    best_mat = [row[:] for row in color_mat]
    
    # 4. Local Search with Simulated Annealing
    T = 1.0
    T_min = 0.01
    decay = 0.999
    
    def get_delta_cost(u, v, new_color):
        old_color = color_mat[u][v]
        if old_color == new_color:
            return 0
        delta = 0
        for w in range(n_total):
            if w == u or w == v:
                continue
            c_vw = color_mat[v][w]
            c_uw = color_mat[u][w]
            was_mono = (old_color == c_vw == c_uw)
            is_mono = (new_color == c_vw == c_uw)
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        return delta

    while time.time() < deadline:
        if current_cost == 0:
            break
            
        u = random.randint(0, n_total - 1)
        v = random.randint(0, n_total - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        best_delta = 0
        best_new_color = color_mat[u][v]
        
        for c in range(4):
            if c == color_mat[u][v]:
                continue
            d = get_delta_cost(u, v, c)
            if d < best_delta:
                best_delta = d
                best_new_color = c
                
        if best_delta < 0:
            color_mat[u][v] = best_new_color
            color_mat[v][u] = best_new_color
            current_cost += best_delta
        else:
            if best_delta == 0:
                if random.random() < 0.5:
                    color_mat[u][v] = best_new_color
                    color_mat[v][u] = best_new_color
            else:
                prob = 2.71828 ** (-best_delta / T)
                if random.random() < prob:
                    color_mat[u][v] = best_new_color
                    color_mat[v][u] = best_new_color
                    current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_mat = [row[:] for row in color_mat]
            
        T *= decay
        if T < T_min:
            T = T_min
            
    if best_cost == 0:
        colors_out = []
        for i in range(n_total):
            for j in range(i + 1, n_total):
                colors_out.append(best_mat[i][j])
        return {'n': n_total, 'colors': colors_out}
    else:
        return {'n': 32, 'colors': colors_base}
