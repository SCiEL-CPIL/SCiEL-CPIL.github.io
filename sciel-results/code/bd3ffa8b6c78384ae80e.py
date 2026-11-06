import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # --- Base n=32 Coloring ---
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
        
    base_32_list = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_32_list.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                base_32_list.append(3)
                
    # --- Target n=50 ---
    N = 50
    # Use a 2D numpy array for colors
    # C[i, j] is color of edge (i, j)
    C = np.zeros((N, N), dtype=np.uint8)
    
    # Fill base 32
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            c = base_32_list[idx]
            C[i, j] = c
            C[j, i] = c
            idx += 1
            
    # Fill rest randomly
    # Create a mask for edges to fill
    # We can just iterate i, j where i < j and (i >= 32 or j >= 32)
    for i in range(N):
        for j in range(i + 1, N):
            if i < 32 and j < 32:
                continue
            c = random.randint(0, 3)
            C[i, j] = c
            C[j, i] = c
            
    def get_energy(C):
        # Count mono triangles
        # i < j < k
        energy = 0
        for i in range(N):
            for j in range(i + 1, N):
                c_ij = C[i, j]
                # k > j
                for k in range(j + 1, N):
                    if C[i, k] == c_ij and C[j, k] == c_ij:
                        energy += 1
        return energy

    current_energy = get_energy(C)
    best_energy = current_energy
    best_C = C.copy()
    
    if best_energy == 0:
        # Convert to list
        colors_list = []
        for i in range(N):
            for j in range(i + 1, N):
                colors_list.append(int(C[i, j]))
        return {'n': N, 'colors': colors_list}

    T = 1.0
    T_min = 0.01
    alpha = 0.9999 # Slower decay
    
    while time.time() < deadline:
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        old_color = C[u, v]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
            
        # Delta E using NumPy
        # Count w where C[u, w] == new_color and C[v, w] == new_color
        # Exclude u and v themselves (diagonal is 0, but we check all w)
        # C[u, u] is 0. If new_color is 0, it might count u.
        # But C[u, u] is not an edge.
        # So we must mask out u and v.
        
        # Vectorized count
        mask_new = (C[u, :] == new_color) & (C[v, :] == new_color)
        mask_old = (C[u, :] == old_color) & (C[v, :] == old_color)
        
        # Exclude u and v from count
        # C[u, u] and C[v, v] are 0.
        # If new_color == 0, mask_new[u] is (C[u,u]==0) & (C[v,u]==0) -> (1) & (C[v,u]==0).
        # We don't want to count w=u or w=v.
        
        count_new = np.sum(mask_new)
        count_old = np.sum(mask_old)
        
        # Correction for w=u
        if C[u, u] == new_color and C[v, u] == new_color:
            count_new -= 1
        if C[u, u] == old_color and C[v, u] == old_color:
            count_old -= 1
        # Correction for w=v
        if C[u, v] == new_color and C[v, v] == new_color:
            count_new -= 1
        if C[u, v] == old_color and C[v, v] == old_color:
            count_old -= 1
        
        # Note: C[u, v] is old_color initially.
        # So for new_color, C[u, v] != new_color (since we picked new != old).
        # So C[u, v] == new_color is False.
        # So the term for w=v in mask_new is False anyway?
        # mask_new[v] = (C[u, v] == new_color) & (C[v, v] == new_color)
        # Since C[u, v] is old_color, and new != old, this is False.
        # So w=v is never counted in mask_new.
        # Similarly for mask_old:
        # mask_old[v] = (C[u, v] == old_color) & (C[v, v] == old_color)
        # C[u, v] is old_color. C[v, v] is 0.
        # If old_color == 0, then mask_old[v] is True.
        # So we might count w=v in mask_old if old_color is 0.
        # But w=v is not a valid third vertex for triangle (u, v, w).
        # So we must subtract it if it was counted.
        
        # Similarly for w=u:
        # mask_new[u] = (C[u, u] == new_color) & (C[v, u] == new_color)
        # C[u, u] is 0. If new_color is 0, and C[v, u] is 0, it counts.
        # mask_old[u] = (C[u, u] == old_color) & (C[v, u] == old_color)
        # C[u, u] is 0. If old_color is 0, and C[v, u] is 0, it counts.

        delta_e = count_new - count_old
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            C[u, v] = new_color
            C[v, u] = new_color
            current_energy += delta_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_C = C.copy()
                if best_energy == 0:
                    colors_list = []
                    for i in range(N):
                        for j in range(i + 1, N):
                            colors_list.append(int(C[i, j]))
                    return {'n': N, 'colors': colors_list}
            
        T *= alpha
        if T < T_min:
            T = T_min

    # Fallback
    return {'n': 32, 'colors': base_32_list}
