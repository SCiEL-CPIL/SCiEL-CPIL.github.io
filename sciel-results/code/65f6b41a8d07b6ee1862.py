import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Baseline n=32 coloring
    def get_baseline_32():
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
        return colors

    baseline_colors = get_baseline_32()
    baseline_n = 32

    # Try to find n=36
    n = 36
    edges = n * (n - 1) // 2

    # Precompute edge indices
    # edge_idx[u][v] = index in colors list
    edge_idx = [[0] * n for _ in range(n)]
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx[u][v] = idx
            edge_idx[v][u] = idx
            idx += 1

    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(edges)]

    # Build adjacency bitsets
    adj = [[0] * n for _ in range(4)]

    def rebuild_adj():
        for c in range(4):
            for i in range(n):
                adj[c][i] = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = colors[edge_idx[u][v]]
                adj[c][u] |= 1 << v
                adj[c][v] |= 1 << u

    rebuild_adj()

    # Calculate total mono triangles
    def count_triangles():
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = colors[edge_idx[u][v]]
                total += bin(adj[c][u] & adj[c][v]).count('1')
        return total // 3

    current_triangles = count_triangles()
    best_triangles = current_triangles
    best_colors = colors[:]

    # Local Search
    temp = 1.0
    cooling = 0.9995
    min_temp = 0.01

    while time.time() < deadline:
        if current_triangles == 0:
            return {'n': n, 'colors': colors}

        if current_triangles < best_triangles:
            best_triangles = current_triangles
            best_colors = colors[:]

        if temp < min_temp:
            break

        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u

        idx_e = edge_idx[u][v]
        c_old = colors[idx_e]

        # Pick new color different from old
        c_new = random.choice([c for c in range(4) if c != c_old])

        # Calculate delta
        loss = bin(adj[c_old][u] & adj[c_old][v]).count('1')
        gain = bin(adj[c_new][u] & adj[c_new][v]).count('1')

        delta = gain - loss
        new_triangles = current_triangles + delta

        # Acceptance criterion
        if delta <= 0 or random.random() < pow(2.71828, -delta / temp):
            # Accept
            adj[c_old][u] &= ~(1 << v)
            adj[c_old][v] &= ~(1 << u)
            adj[c_new][u] |= 1 << v
            adj[c_new][v] |= 1 << u

            colors[idx_e] = c_new
            current_triangles = new_triangles

        temp *= cooling

    # If we didn't find 0 triangles, return baseline
    return {'n': baseline_n, 'colors': baseline_colors}
