import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    
    # Seed construction for n=32
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
    best_colors = seed_colors

    # Build 32x32 numpy matrix
    mat_32 = np.zeros((32, 32), dtype=int)
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            mat_32[i, j] = seed_colors[idx]
            mat_32[j, i] = seed_colors[idx]
            idx += 1

    def get_total_triangles_np(colors, n):
        total = 0
        for c in range(4):
            M = (colors == c).astype(int)
            M2 = M @ M
            M3 = M2 @ M
            total += np.trace(M3) / 6
        return int(total)

    def local_search(colors, n, time_limit):
        current_triangles = get_total_triangles_np(colors, n)
        if current_triangles == 0:
            return True, colors
            
        start_time = time.time()
        temp = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        while time.time() - start_time < time_limit and time.time() < deadline - 2.0:
            if current_triangles == 0:
                return True, colors
                
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j: continue
            if i > j: i, j = j, i
            
            old_c = colors[i, j]
            new_c = random.choice([0, 1, 2, 3])
            if new_c == old_c: continue
            
            old_contrib = 0
            new_contrib = 0
            # Vectorized check?
            # colors[i, :] == old_c and colors[j, :] == old_c
            # Exclude i, j
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            
            old_match = (colors[i, mask] == old_c) & (colors[j, mask] == old_c)
            new_match = (colors[i, mask] == new_c) & (colors[j, mask] == new_c)
            
            old_contrib = np.sum(old_match)
            new_contrib = np.sum(new_match)
            
            delta = new_contrib - old_contrib
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                if temp > min_temp:
                    accept = random.random() < temp
            
            if accept:
                colors[i, j] = new_c
                colors[j, i] = new_c
                current_triangles += delta
                
            if temp > min_temp:
                temp *= cooling
        
        return current_triangles == 0, colors

    # Try to find larger n
    for target_n in range(33, 51):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        for restart in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize matrix
            colors = np.zeros((target_n, target_n), dtype=int)
            # Copy seed
            colors[:32, :32] = mat_32
            
            # Fill new edges
            for i in range(32, target_n):
                for j in range(i):
                    c = random.randint(0, 3)
                    colors[i, j] = c
                    colors[j, i] = c
            
            # Run local search
            ok, final_colors = local_search(colors, target_n, time_limit=0.5)
            if ok:
                # Convert to list
                res_colors = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        res_colors.append(int(final_colors[i, j]))
                best_n = target_n
                best_colors = res_colors
                found = True
                break
        
    return {'n': best_n, 'colors': best_colors}
