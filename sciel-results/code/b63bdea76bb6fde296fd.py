import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Generate the seed solution (n=32)
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
    
    seed_result = {'n': 32, 'colors': seed_colors}
    best_n = 32
    best_colors = seed_colors
    
    # Helper to convert matrix to list of colors
    def matrix_to_colors(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(mat[i, j]))
        return colors

    # Incremental Triangle Counting
    # We maintain a 2D array `tri_count[c][i][j]`? No, just total count per color.
    # Actually, we just need the total number of monochromatic triangles.
    # Let's maintain `cost` as the total number of monochromatic triangles.
    
    # To update cost when changing edge (i, j) from c_old to c_new:
    # For each k != i, j:
    #   If mat[i,k] == mat[j,k] == c_old, then we were counting a triangle (i,j,k) in c_old.
    #   Now it's not. So cost -= 1.
    #   If mat[i,k] == mat[j,k] == c_new, then we are now counting a triangle (i,j,k) in c_new.
    #   So cost += 1.
    
    # We need to store the matrix to check neighbors.
    
    def try_solve(n_target):
        mat = np.zeros((n_target, n_target), dtype=int)
        
        # Fill first 32x32 with seed if n_target >= 32
        if n_target >= 32:
            idx = 0
            for i in range(32):
                for j in range(i + 1, 32):
                    mat[i, j] = seed_colors[idx]
                    mat[j, i] = seed_colors[idx]
                    idx += 1
            
            # Fill rest randomly
            for i in range(32, n_target):
                for j in range(i + 1, n_target):
                    c = random.randint(0, 3)
                    mat[i, j] = c
                    mat[j, i] = c
            for i in range(32):
                for j in range(32, n_target):
                    c = random.randint(0, 3)
                    mat[i, j] = c
                    mat[j, i] = c
        else:
            # Just random
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    c = random.randint(0, 3)
                    mat[i, j] = c
                    mat[j, i] = c

        # Calculate initial cost
        cost = 0
        for c in range(4):
            A = (mat == c).astype(int)
            np.fill_diagonal(A, 0)
            cost += int(np.trace(A @ A @ A) / 6)
        
        T = 1000.0
        cooling = 0.999
        
        while time.time() < deadline - 1.0:
            if cost == 0:
                return n_target, matrix_to_colors(mat, n_target)
            
            i = random.randint(0, n_target - 1)
            j = random.randint(0, n_target - 1)
            if i == j:
                continue
            
            c_old = mat[i, j]
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
            
            # Calculate delta
            delta = 0
            for k in range(n_target):
                if k == i or k == j:
                    continue
                if mat[i, k] == c_old and mat[j, k] == c_old:
                    delta -= 1
                if mat[i, k] == c_new and mat[j, k] == c_new:
                    delta += 1
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                mat[i, j] = c_new
                mat[j, i] = c_new
                cost += delta
            T *= cooling
            
        if cost == 0:
            return n_target, matrix_to_colors(mat, n_target)
        return None

    # Try n=50
    if time.time() < deadline - 5.0:
        res = try_solve(50)
        if res:
            best_n, best_colors = res
    
    # If 50 failed, try 49?
    if best_n < 50 and time.time() < deadline - 5.0:
        res = try_solve(49)
        if res:
            best_n, best_colors = res
            
    # If 49 failed, try 48?
    if best_n < 49 and time.time() < deadline - 5.0:
        res = try_solve(48)
        if res:
            best_n, best_colors = res

    return {'n': best_n, 'colors': best_colors}
