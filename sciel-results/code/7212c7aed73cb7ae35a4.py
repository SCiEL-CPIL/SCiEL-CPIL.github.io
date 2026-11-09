import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def get_seed_solution():
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

    seed_sol = get_seed_solution()

    def try_solve(n, deadline):
        # Initialize colors
        # Use a deterministic pseudo-random seed for reproducibility? 
        # No, random restarts are better.
        np.random.seed() # Reseed? Or just use global random.
        
        colors = np.random.randint(0, 4, size=(n, n))
        colors = (colors + colors.T) // 2
        np.fill_diagonal(colors, 0)
        
        def get_flat_colors(c_mat):
            flat = []
            for i in range(n):
                for j in range(i+1, n):
                    flat.append(int(c_mat[i, j]))
            return flat

        def count_bad(c_mat):
            bad = 0.0
            for c in range(4):
                mask = (c_mat == c).astype(np.int8)
                # Number of triangles in graph defined by mask
                # trace(mask^3) / 6
                M3 = mask @ mask @ mask
                bad += np.trace(M3) / 6.0
            return int(bad)

        current_bad = count_bad(colors)
        
        edge_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        max_steps = 50000
        steps = 0
        no_improve_count = 0
        
        while current_bad > 0 and steps < max_steps and time.time() < deadline - 2:
            steps += 1
            
            # Pick a random edge
            idx = np.random.randint(len(edge_indices))
            i, j = edge_indices[idx]
            old_c = colors[i, j]
            
            best_c = old_c
            min_bad = current_bad
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                colors[i, j] = new_c
                colors[j, i] = new_c
                new_bad = count_bad(colors)
                if new_bad < min_bad:
                    min_bad = new_bad
                    best_c = new_c
                colors[i, j] = old_c
                colors[j, i] = old_c
            
            if best_c != old_c:
                colors[i, j] = best_c
                colors[j, i] = best_c
                current_bad = min_bad
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If stuck, randomize
                if no_improve_count > 500:
                    # Perturb
                    for _ in range(10):
                        pi, pj = edge_indices[np.random.randint(len(edge_indices))]
                        pc = np.random.randint(0, 4)
                        colors[pi, pj] = pc
                        colors[pj, pi] = pc
                    current_bad = count_bad(colors)
                    no_improve_count = 0

        if current_bad == 0:
            return get_flat_colors(colors)
        return None

    # Try n=50 down to 33
    for n in range(50, 32, -1):
        if time.time() > deadline:
            break
        # Multiple restarts
        for attempt in range(10):
            if time.time() > deadline:
                break
            res = try_solve(n, deadline)
            if res is not None:
                return {'n': n, 'colors': res}
            
    return seed_sol
