"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

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

    # Build the 32-vertex Greenwood-Gleason coloring
    n_start = 32
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)

    def adj_from_colors(n, colors):
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i, j] = c
                adj[j, i] = c
                idx += 1
        return adj

    def colors_from_adj(adj):
        n = adj.shape[0]
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(adj[i, j]))
        return colors

    def build_color_graphs(adj, n):
        """Build G[c][i] = set of neighbors j of i via color-c edges."""
        G = {}
        for c in range(4):
            G[c] = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == c:
                        G[c][i].add(j)
                        G[c][j].add(i)
        return G

    def try_add_vertex(adj, G, max_backtracks=1000000):
        """Try to add a new vertex to the coloring. Returns (new_adj, new_G) or (None, None)."""
        n = adj.shape[0]
        assignment = [-1] * n

        def is_consistent(pos, c):
            for j in range(pos):
                if assignment[j] == c and j in G[c][pos]:
                    return False
            return True

        def backtrack(pos, backtrack_count):
            if pos == n:
                return True
            if backtrack_count[0] > max_backtracks:
                return False

            # Try colors in a smart order: prefer colors where fewer conflicts exist
            # Sort by number of already-assigned neighbors with that color
            color_scores = []
            for c in range(4):
                conflicts = 0
                for j in range(pos):
                    if assignment[j] == c and j in G[c][pos]:
                        conflicts += 1
                # Prefer colors with fewer conflicts (more likely to be consistent)
                # But also prefer colors that are "available"
                color_scores.append((conflicts, c))
            color_scores.sort()

            # Add some randomness to avoid getting stuck in local patterns
            if random.random() < 0.3:
                color_scores = [(random.randint(0, 100), c) for c in range(4)]
                color_scores.sort()

            for _, c in color_scores:
                if is_consistent(pos, c):
                    assignment[pos] = c
                    backtrack_count[0] += 1
                    if backtrack(pos + 1, backtrack_count):
                        return True
                    assignment[pos] = -1

            return False

        backtrack_count = [0]
        if backtrack(0, backtrack_count):
            new_adj = np.zeros((n + 1, n + 1), dtype=np.int8)
            new_adj[:n, :n] = adj
            for i in range(n):
                new_adj[n, i] = assignment[i]
                new_adj[i, n] = assignment[i]

            # Update G for the new vertex
            new_G = {}
            for c in range(4):
                new_G[c] = [set() for _ in range(n + 1)]
                for i in range(n):
                    new_G[c][i] = G[c][i].copy()
            for i in range(n):
                c = assignment[i]
                new_G[c][n].add(i)
                new_G[c][i].add(n)

            return new_adj, new_G

        return None, None

    def try_add_vertex_random(adj, G, num_trials=500):
        """Try random assignments to add a vertex. Returns (new_adj, new_G) or (None, None)."""
        n = adj.shape[0]

        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break

            # Random assignment
            assignment = [random.randint(0, 3) for _ in range(n)]

            # Check validity
            valid = True
            for c in range(4):
                verts = [i for i in range(n) if assignment[i] == c]
                for a in range(len(verts)):
                    if not valid:
                        break
                    u1 = verts[a]
                    for b in range(a + 1, len(verts)):
                        u2 = verts[b]
                        if u1 in G[c][u2]:  # edge (u1, u2) has color c
                            valid = False
                            break

            if valid:
                new_adj = np.zeros((n + 1, n + 1), dtype=np.int8)
                new_adj[:n, :n] = adj
                for i in range(n):
                    new_adj[n, i] = assignment[i]
                    new_adj[i, n] = assignment[i]

                new_G = {}
                for c in range(4):
                    new_G[c] = [set() for _ in range(n + 1)]
                    for i in range(n):
                        new_G[c][i] = G[c][i].copy()
                for i in range(n):
                    c = assignment[i]
                    new_G[c][n].add(i)
                    new_G[c][i].add(n)

                return new_adj, new_G

        return None, None

    adj = adj_from_colors(n_start, colors)
    G = build_color_graphs(adj, n_start)
    best_adj = adj.copy()
    best_n = n_start

    # Main loop: try to add vertices one at a time
    while time.time() < deadline - 10:
        # Try backtracking first (more systematic)
        new_adj, new_G = try_add_vertex(adj, G, max_backtracks=200000)

        if new_adj is None:
            # Try random approach as fallback
            new_adj, new_G = try_add_vertex_random(adj, G, num_trials=200)

        if new_adj is None:
            # Can't add another vertex with this strategy
            # Try a different strategy: restart with different random seed
            # or try adding multiple vertices at once (more complex)
            # For now, just break
            break

        adj = new_adj
        G = new_G
        best_adj = adj.copy()
        best_n = adj.shape[0]

        # If we've made significant progress, we could try to continue
        # but be careful about time

    best_colors = colors_from_adj(best_adj)
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
