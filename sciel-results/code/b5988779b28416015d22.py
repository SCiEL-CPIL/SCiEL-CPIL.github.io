"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _is_valid_coloring(n, colors):
    """Check if coloring has no monochromatic C4 using numpy matrix products."""
    if len(colors) != n * (n - 1) // 2:
        return False
    # Build adjacency matrices per color
    adjs = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 4:
                return False
            adjs[c][i, j] = 1
            adjs[c][j, i] = 1
            idx += 1
    # For each color, A^2 off-diagonal entries must be <= 1
    for c in range(5):
        A2 = adjs[c] @ adjs[c]
        # Zero out diagonal for check
        np.fill_diagonal(A2, 0)
        if np.max(A2) > 1:
            return False
    return True

def _lazy_woldar_25():
    """Construct a valid 5-coloring of K_25 using Z_5 x Z_5."""
    n = 25
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = i // 5, i % 5
            xj, yj = j // 5, j % 5
            if xi == xj:
                c = 4
            else:
                dx = (xj - xi) % 5
                dy = (yj - yi) % 5
                # dx^{-1} mod 5
                inv = pow(dx, -1, 5)
                c = (dy * inv) % 5
            colors.append(c)
    return n, colors

def _try_extend_to_26(base_colors, deadline):
    """Try to extend a 25-vertex coloring to 26 vertices via backtracking."""
    n_base = 25
    # Precompute incompatibility: for each color c0, incompatible[c0] is a set of pairs (a,c)
    # where a and c share a common neighbor in color c0.
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj[c].add((i, j))
            idx += 1

    neigh = [set() for _ in range(n_base)]
    for c in range(5):
        for (i, j) in adj[c]:
            neigh[i].add(j)
            neigh[j].add(i)

    incompatible = [set() for _ in range(5)]
    for c0 in range(5):
        for a in range(n_base):
            for c in range(a + 1, n_base):
                if len(neigh[a] & neigh[c]) >= 1:
                    incompatible[c0].add((a, c))

    edge_colors = {}

    def try_assign():
        if time.time() > deadline:
            return None

        if len(edge_colors) == n_base:
            return edge_colors.copy()

        # MRV heuristic: pick vertex with fewest valid colors
        best_vertex = -1
        best_options = None
        min_count = 6

        for v in range(n_base):
            if v in edge_colors:
                continue
            valid = []
            for c0 in range(5):
                ok = True
                for u, cu in edge_colors.items():
                    if cu == c0:
                        pair = (min(v, u), max(v, u))
                        if pair in incompatible[c0]:
                            ok = False
                            break
                if ok:
                    valid.append(c0)
            if len(valid) < min_count:
                min_count = len(valid)
                best_vertex = v
                best_options = valid
                if min_count == 0:
                    break

        if best_vertex == -1 or not best_options:
            return None

        for c0 in best_options:
            edge_colors[best_vertex] = c0
            result = try_assign()
            if result is not None:
                return result
            del edge_colors[best_vertex]

        return None

    return try_assign()

def _local_search_26(n, initial_colors, deadline):
    """Local search to find valid 26-vertex coloring, starting from initial guess."""
    # Precompute edge index mapping
    n_edges = n * (n - 1) // 2
    
    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - (i * (i - 1)) // 2 + (j - i - 1)

    # Build initial adjacency bitsets
    # adj[c] is list of n integers (bitmasks)
    adj = [0] * (5 * n)  # adj[c*n + v] = bitmask of neighbors of v in color c
    
    def rebuild_adj(colors):
        for c in range(5):
            for v in range(n):
                adj[c * n + v] = 0
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c * n + i] |= (1 << j)
                adj[c * n + j] |= (1 << i)
                idx += 1

    def count_conflicts():
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c * n + i] & adj[c * n + j]
                    cnt = bin(common).count('1')
                    if cnt > 1:
                        total += cnt - 1
        return total

    rebuild_adj(initial_colors)
    current_conf = count_conflicts()
    if current_conf == 0:
        return initial_colors

    best_conf = current_conf
    best_colors = initial_colors[:]

    step = 0
    while time.time() < deadline:
        step += 1
        if current_conf == 0:
            return initial_colors[:]

        # Pick a random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i

        idx = get_edge_idx(i, j)
        old_c = initial_colors[idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue

        # Remove edge from old color
        adj[old_c * n + i] &= ~(1 << j)
        adj[old_c * n + j] &= ~(1 << i)
        # Add edge to new color
        adj[new_c * n + i] |= (1 << j)
        adj[new_c * n + j] |= (1 << i)

        new_conf = count_conflicts()
        if new_conf < current_conf or (new_conf == current_conf and random.random() < 0.3):
            current_conf = new_conf
            initial_colors[idx] = new_c
            if new_conf < best_conf:
                best_conf = new_conf
                best_colors = initial_colors[:]
        else:
            # Revert
            adj[new_c * n + i] &= ~(1 << j)
            adj[new_c * n + j] &= ~(1 << i)
            adj[old_c * n + i] |= (1 << j)
            adj[old_c * n + j] |= (1 << i)

        if step % 1000 == 0 and current_conf > best_conf:
            # Occasionally restart from best
            if random.random() < 0.1:
                initial_colors = best_colors[:]
                rebuild_adj(initial_colors)
                current_conf = best_conf

    if best_conf == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 50s safety margin

    # Start with the known 25-vertex construction
    n, colors = _lazy_woldar_25()
    best_n, best_colors = n, colors[:]

    # Verify base construction
    if not _is_valid_coloring(n, colors):
        n_small = 11
        return {'n': n_small, 'colors': [min(j - i, n_small - j + i) - 1 for i in range(n_small) for j in range(i + 1, n_small)]}

    # Try deterministic backtracking to extend to 26
    if time.time() < deadline - 30:
        result = _try_extend_to_26(colors, deadline - 30)
        if result is not None:
            new_colors = colors[:] + [result[i] for i in range(25)]
            if _is_valid_coloring(26, new_colors):
                return {'n': 26, 'colors': new_colors}

    # Try local search for 26 vertices
    if time.time() < deadline - 10:
        # Initialize edges from vertex 25 to 0..24 randomly
        initial_26 = colors[:] + [random.randint(0, 4) for _ in range(25)]
        ls_result = _local_search_26(26, initial_26, deadline - 5)
        if ls_result is not None:
            if _is_valid_coloring(26, ls_result):
                return {'n': 26, 'colors': ls_result}

    # If all else fails, return the 25-vertex solution
    return {'n': 25, 'colors': best_colors}
# EVOLVE-BLOCK-END
