import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

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

    best_result = get_seed_coloring()

    def run_sa_fast(n, time_limit, seed=None):
        if time.time() > time_limit:
            return None

        if seed is not None:
            random.seed(seed)

        # Bitmask representation: adj[k][i] is a bitmask of neighbors of i with color k
        adj = [[0] * n for _ in range(4)]

        # Color array: color[i*n + j] for i < j, stored as flat
        # We'll use a 2D list for simplicity, or a flat array
        # Flat array: index = i*n + j for i<j
        # Actually, let's use a dict or 2D list for O(1) access
        # 2D list is fine for n up to 62
        color = [[-1] * n for _ in range(n)]

        # Initialize with random colors
        for i in range(n):
            for j in range(i + 1, n):
                k = random.randint(0, 3)
                color[i][j] = k
                color[j][i] = k
                adj[k][i] |= (1 << j)
                adj[k][j] |= (1 << i)

        # Count total mono triangles
        def count_all():
            total = 0
            for k in range(4):
                for i in range(n):
                    # For each edge (i,j) of color k, count common neighbors of i and j with color k
                    mask_i = adj[k][i]
                    # Iterate over set bits in mask_i
                    m = mask_i
                    while m:
                        lb = m & (-m)
                        j = lb.bit_length() - 1
                        # Count common neighbors: adj[k][i] & adj[k][j]
                        common = adj[k][i] & adj[k][j]
                        total += common.bit_count()
                        m ^= lb
            return total // 6

        current_score = count_all()
        if current_score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(color[i][j])
            return {'n': n, 'colors': colors}

        best_score = current_score
        best_color = [row[:] for row in color]

        T = 2.0
        T_min = 0.01
        cooling_rate = 0.9995
        steps_since_improve = 0
        max_steps_no_improve = 20000

        while time.time() < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            old_k = color[i][j]

            # Pick new color
            new_k = random.randint(0, 3)
            if new_k == old_k:
                continue

            # Calculate delta in mono triangles
            # Removing edge (i,j) from color old_k:
            # For each common neighbor w of i and j with color old_k, we lose 1 triangle
            common_old = adj[old_k][i] & adj[old_k][j]
            delta_remove = common_old.bit_count()

            # Adding edge (i,j) to color new_k:
            # For each common neighbor w of i and j with color new_k, we gain 1 triangle
            common_new = adj[new_k][i] & adj[new_k][j]
            delta_add = common_new.bit_count()

            delta = delta_add - delta_remove

            # Accept or reject
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Apply change
                adj[old_k][i] ^= (1 << j)
                adj[old_k][j] ^= (1 << i)
                adj[new_k][i] |= (1 << j)
                adj[new_k][j] |= (1 << i)
                color[i][j] = new_k
                color[j][i] = new_k
                current_score += delta

                steps_since_improve += 1
                if current_score < best_score:
                    best_score = current_score
                    best_color = [row[:] for row in color]
                    steps_since_improve = 0
                    if best_score == 0:
                        colors = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                colors.append(color[i][j])
                        return {'n': n, 'colors': colors}
            else:
                pass  # No change made

            T *= cooling_rate
            if T < T_min:
                T = T_min

            if steps_since_improve > max_steps_no_improve:
                # Restart from best
                color = [row[:] for row in best_color]
                # Rebuild adj from best_color
                adj = [[0] * n for _ in range(4)]
                for i in range(n):
                    for j in range(i + 1, n):
                        k = color[i][j]
                        adj[k][i] |= (1 << j)
                        adj[k][j] |= (1 << i)
                current_score = best_score
                T = 2.0
                steps_since_improve = 0

        if best_score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(best_color[i][j])
            return {'n': n, 'colors': colors}

        return None

    # Focus on n=50 with multiple restarts
    n_target = 50
    num_restarts = 20

    for attempt in range(num_restarts):
        if time.time() > deadline:
            break

        time_left = deadline - time.time()
        if time_left < 3:
            break

        # Allocate time per attempt
        alloc = min(5.0, time_left - 2.0)

        result = run_sa_fast(n_target, time.time() + alloc, seed=attempt * 1000 + 42)

        if result:
            return result

    # Try smaller n if 50 fails
    for n in range(49, 32, -1):
        if time.time() > deadline:
            break

        time_left = deadline - time.time()
        if time_left < 2:
            break

        alloc = min(3.0, time_left - 1.0)

        result = run_sa_fast(n, time.time() + alloc, seed=n * 100 + 7)

        if result:
            return result

    return best_result
