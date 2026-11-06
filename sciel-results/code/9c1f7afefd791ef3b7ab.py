import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 140.0

    def get_seed_coloring():
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

    def find_one_violation(color_matrix, n):
        # Randomize order to avoid bias
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for i, j in pairs:
            c = color_matrix[i][j]
            for k in range(n):
                if k == i or k == j: continue
                # Check triangle i, j, k
                # Need color(i, k) and color(j, k)
                ik = color_matrix[i][k] if i < k else color_matrix[k][i]
                jk = color_matrix[j][k] if j < k else color_matrix[k][j]
                if ik == c and jk == c:
                    return (i, j, k, c)
        return None

    def try_recolor(color_matrix, i, j, new_color, n):
        old_color = color_matrix[i][j]
        if old_color == new_color:
            return True
        
        for k in range(n):
            if k == i or k == j:
                continue
            ik = color_matrix[i][k] if i < k else color_matrix[k][i]
            jk = color_matrix[j][k] if j < k else color_matrix[k][j]
            if ik == new_color and jk == new_color:
                return False
        return True

    def local_search(n):
        # Initialize
        color_matrix = [[random.randint(0, 3) for _ in range(n)] for _ in range(n)]
        
        max_iters = 5000 # Cap iterations per restart to prevent infinite loops if stuck
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                return None
            
            viol = find_one_violation(color_matrix, n)
            if not viol:
                return color_matrix
            
            i, j, k, c = viol
            edges = [(i, j), (i, k), (j, k)]
            random.shuffle(edges)
            
            moved = False
            for u, v in edges:
                if u > v: u, v = v, u
                current_c = color_matrix[u][v]
                # Shuffle colors to try
                colors_to_try = [0, 1, 2, 3]
                random.shuffle(colors_to_try)
                for new_c in colors_to_try:
                    if new_c == current_c:
                        continue
                    if try_recolor(color_matrix, u, v, new_c, n):
                        color_matrix[u][v] = new_c
                        moved = True
                        break
                if moved:
                    break
            
            if not moved:
                # Kick
                for _ in range(3):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        if u > v: u, v = v, u
                        color_matrix[u][v] = random.randint(0, 3)

    # Try n=50
    for target_n in [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() - start_time > time_limit - 10:
            break
        
        # Try a few restarts
        for restart in range(10):
            if time.time() - start_time > time_limit - 10:
                break
            matrix = local_search(target_n)
            if matrix:
                colors = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        colors.append(matrix[i][j])
                return {'n': target_n, 'colors': colors}

    return get_seed_coloring()
