import time
import random
import math

def construct():
    deadline = time.time() + 140
    
    # Seed n=32
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
    seed_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            seed_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    best_n = 32
    best_colors = seed_colors[:]
    
    # Precompute index maps for n up to 50
    idx_maps = {}
    for n in range(1, 51):
        idx_map = {}
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_map[(i, j)] = k
                k += 1
        idx_maps[n] = idx_map
        
    def get_idx(n, i, j):
        if i > j: i, j = j, i
        return idx_maps[n][(i, j)]

    def count_mono_triangles(n, colors):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[get_idx(n, i, j)]
                for k in range(j + 1, n):
                    c_ik = colors[get_idx(n, i, k)]
                    c_jk = colors[get_idx(n, j, k)]
                    if c_ij == c_ik == c_jk:
                        count += 1
        return count

    def simulate_annealing(n, colors, time_limit=5.0):
        energy = count_mono_triangles(n, colors)
        if energy == 0:
            return colors, energy
        
        T = 1.0
        cooling = 0.999
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            idx = get_idx(n, u, v)
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            delta = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                c_uw = colors[get_idx(n, u, w)]
                c_vw = colors[get_idx(n, v, w)]
                
                was_mono = (old_c == c_uw == c_vw)
                is_mono = (new_c == c_uw == c_vw)
                
                if was_mono and not is_mono:
                    delta -= 1
                elif not was_mono and is_mono:
                    delta += 1
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[idx] = new_c
                energy += delta
            
            T *= cooling
            if energy == 0:
                break
        
        return colors, energy

    current_n = 32
    current_colors = seed_colors[:]
    
    while current_n < 50 and time.time() < deadline - 10:
        next_n = current_n + 1
        next_colors = [0] * (next_n * (next_n - 1) // 2)
        
        # Copy existing
        for i in range(current_n):
            for j in range(i + 1, current_n):
                next_colors[get_idx(next_n, i, j)] = current_colors[get_idx(current_n, i, j)]
        
        # Init new vertex edges
        new_v = next_n - 1
        success = False
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random init for new edges
            for u in range(new_v):
                next_colors[get_idx(next_n, u, new_v)] = random.randint(0, 3)
            
            # Run SA
            next_colors, energy = simulate_annealing(next_n, next_colors, time_limit=2.0)
            
            if energy == 0:
                success = True
                break
        
        if success:
            current_n = next_n
            current_colors = next_colors
            best_n = current_n
            best_colors = current_colors[:]
        else:
            # If we fail to extend, we might still try to improve current_n?
            # Or just stop.
            break
            
    return {'n': best_n, 'colors': best_colors}
