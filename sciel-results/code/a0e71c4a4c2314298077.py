import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin

    def has_mono_k222_fast(n, colors):
        """
        Fast check for monochromatic K2,2,2 using bitmasks.
        colors: list of 0/1 of length n*(n-1)/2
        """
        # Build adjacency bitmasks for each color
        adj = [[0] * n for _ in range(2)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in [0, 1]:
            for u in range(n):
                for v in range(u + 1, n):
                    mask_c = adj[c][u] & adj[c][v]
                    if mask_c.bit_count() < 4:
                        continue
                    
                    verts = [i for i in range(n) if (mask_c >> i) & 1]
                    
                    found = False
                    for i_idx in range(len(verts)):
                        x = verts[i_idx]
                        mask_x = adj[c][x]
                        for j_idx in range(i_idx + 1, len(verts)):
                            y = verts[j_idx]
                            common_xy = mask_x & adj[c][y] & mask_c
                            if common_xy.bit_count() >= 2:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return True
        return False

    def generate_paley(n):
        if n % 4 != 1:
            return None
        squares = {x * x % n for x in range(1, n)}
        colors = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
        return colors

    # Try n=29 first (Paley)
    colors_29 = generate_paley(29)
    if colors_29:
        if not has_mono_k222_fast(29, colors_29):
            return {'n': 29, 'colors': colors_29}

    # Try n=17 (Paley)
    colors_17 = generate_paley(17)
    if colors_17:
        if not has_mono_k222_fast(17, colors_17):
            return {'n': 17, 'colors': colors_17}

    # Simulated Annealing for n=30 (limited time)
    if time.time() < start_time + 100:
        n_target = 30
        for _ in range(10): # 10 restarts
            if time.time() > start_time + 130:
                break
            colors = [random.randint(0, 1) for _ in range(n_target * (n_target - 1) // 2)]
            if not has_mono_k222_fast(n_target, colors):
                return {'n': n_target, 'colors': colors}
            for _ in range(100):
                if time.time() > start_time + 130:
                    break
                idx = random.randint(0, len(colors) - 1)
                colors[idx] ^= 1
                if not has_mono_k222_fast(n_target, colors):
                    return {'n': n_target, 'colors': colors}

    # Fallback
    n = 17
    colors = generate_paley(17)
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
