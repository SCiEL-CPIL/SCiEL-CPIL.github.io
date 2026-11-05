"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using incremental local search."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    def edge_index(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def build_adjacency(colors, n):
        """Build adjacency lists per color: adj[c][v] = set of neighbors."""
        adj = [[set() for _ in range(n)] for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        return adj

    def _count_c6s_through_edge(adj, n, u, v):
        """Count C6s in this adjacency that use edge (u,v).
        A C6 using (u,v): u-v-a-b-c-d-u
        """
        count = 0
        for a in adj[v]:
            if a == u:
                continue
            for b in adj[a]:
                if b in (u, v, a):
                    continue
                for c in adj[b]:
                    if c in (u, v, a, b):
                        continue
                    for d in adj[c]:
                        if d in (u, v, a, b, c):
                            continue
                        if u in adj[d]:
                            count += 1
        return count

    def total_mono_c6(adj, n):
        """Total monochromatic C6s across all 4 colors."""
        total = 0
        for c in range(4):
            for u in range(n):
                for v in adj[c][u]:
                    if v > u:
                        total += _count_c6s_through_edge(adj[c], n, u, v)
        return total

    def _edge_to_vertices(ei, n):
        """Convert edge index to (u, v) with u < v."""
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cum = mid * (n - 1) - mid * (mid - 1) // 2
            if cum <= ei:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        cum_before = i * (n - 1) - i * (i - 1) // 2
        j = i + 1 + (ei - cum_before)
        return i, j

    def structured_init(n):
        """Create a structured initial coloring using modular arithmetic."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                # Use a hash-like function to distribute colors
                c = (i * 3 + j * 7 + d * 11) % 4
                colors.append(c)
        return colors

    def local_search(colors, n, max_iters=50000, deadline=130.0):
        """Local search with incremental C6 counting."""
        num_edges = n * (n - 1) // 2
        colors_arr = np.array(colors, dtype=np.int32)
        adj = build_adjacency(colors_arr.tolist(), n)
        
        current_count = total_mono_c6(adj, n)
        if current_count == 0:
            return colors_arr.tolist(), 0

        T = 3.0
        T_min = 0.05
        no_improve = 0
        max_no_improve = 8000

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if no_improve > max_no_improve:
                break
            if current_count == 0:
                return colors_arr.tolist(), 0

            # Select a random edge
            ei = random.randint(0, num_edges - 1)
            u, v = _edge_to_vertices(ei, n)

            old_color = int(colors_arr[ei])
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue

            # Compute delta
            delta = 0
            # Remove C6s that used (u,v) in old_color
            delta -= _count_c6s_through_edge(adj[old_color], n, u, v)
            # Add C6s that use (u,v) in new_color
            delta += _count_c6s_through_edge(adj[new_color], n, u, v)

            if delta <= 0 or random.random() < np.exp(-delta / max(0.1, T)):
                # Accept: update adjacency
                adj[old_color][u].discard(v)
                adj[old_color][v].discard(u)
                adj[new_color][u].add(v)
                adj[new_color][v].add(u)
                colors_arr[ei] = new_color
                current_count += delta
                no_improve = 0
            else:
                no_improve += 1

            # Cool down
            T = max(T_min, T * 0.99995)

        return colors_arr.tolist(), current_count

    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline:
            break

        num_edges = n * (n - 1) // 2
        remaining = deadline - time.time()
        if remaining < 5:
            break

        found = False

        # Multiple diverse starting points
        max_restarts = max(3, int(remaining * 1.5))
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break

            if attempt == 0:
                # Structured initialization
                colors = structured_init(n)
            elif attempt == 1:
                # Another structured variant
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = (j - i) % n
                        c = (i * 5 + j * 13 + d * 17) % 4
                        colors.append(c)
            elif attempt == 2:
                # Symmetric-ish initialization
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = (j - i) % n
                        c = (d * 5) % 4
                        colors.append(c)
            else:
                # Random
                colors = [random.randint(0, 3) for _ in range(num_edges)]

            colors, count = local_search(colors, n, max_iters=30000, deadline=deadline)

            if count == 0:
                best_n = n
                best_colors = colors
                found = True
                break

        if not found and n < 17:
            break

    # Fallback
    if best_n < 9:
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

    return {'n': best_n, 'colors': best_colors}
