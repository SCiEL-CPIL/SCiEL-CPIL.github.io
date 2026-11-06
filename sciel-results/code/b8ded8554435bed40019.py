import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for the 150s evaluator limit

    def get_idx(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def count_violations_numba(n, colors):
        """Count C4 violations using bitmasks. Returns count."""
        if n < 4:
            return 0
        # Build adjacency bitmasks per color
        N = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                common = N[c][i] & N[c][j]
                if bin(common).count('1') >= 2:
                    violations += 1
        return violations

    def build_N(n, colors):
        """Build adjacency bitmasks N[c][v] for each color c and vertex v."""
        N = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        return N

    def count_violations_from_N(n, N, colors):
        """Count violations given precomputed N."""
        if n < 4:
            return 0
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                common = N[c][i] & N[c][j]
                if bin(common).count('1') >= 2:
                    violations += 1
        return violations

    def local_search(n, max_iters, deadline, initial_colors=None):
        """
        Local search with simulated annealing to find a valid C4-free 5-coloring.
        Returns (colors, violations) tuple.
        """
        n_edges = n * (n - 1) // 2
        if n_edges == 0:
            return ([], 0)

        # Edge list
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]

        if initial_colors is None:
            colors = [random.randint(0, 4) for _ in range(n_edges)]
        else:
            colors = list(initial_colors)

        N = build_N(n, colors)
        cur_viol = count_violations_from_N(n, N, colors)

        if cur_viol == 0:
            return (colors, 0)

        best_colors = list(colors)
        best_viol = cur_viol

        # Simulated annealing parameters
        T = 1.0
        T_min = 0.001
        cooling = 0.9995

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Pick a random edge
            k = random.randint(0, n_edges - 1)
            i, j = edge_list[k]
            c = colors[k]

            # Try a new color
            new_c = random.randint(0, 4)
            if new_c == c:
                continue

            # Compute delta in violations
            # Removing (i,j) from color c:
            #   - Edge (i,j) itself: was it violating?
            #   - Edges (i,v) for v in N[c][i]\{j}: removing j from N[c][i] may fix or create violations
            #   - Edges (j,v) for v in N[c][j]\{i}: similarly
            # Adding (i,j) to color new_c:
            #   - Edge (i,j) with new_c
            #   - Edges (i,v) for v in N[new_c][i]: adding j to N[new_c][i]
            #   - Edges (j,v) for v in N[new_c][j]: adding i to N[new_c][j]

            # This is complex. Let's use a simpler approach:
            # Compute violations before and after for affected edges only.
            # Affected edges: (i,j) and all (i,v), (j,v) where v is neighbor in color c or new_c.

            # For efficiency, let's just compute the delta by checking:
            # 1. Edge (i,j): old contribution vs new contribution
            # 2. For each v != i,j: edge (i,v) old vs new, edge (j,v) old vs new
            #    but only for v that are neighbors of i or j in color c or new_c

            # Actually, the simplest correct approach:
            # Temporarily change, count violations for affected edges, revert if worse.

            # Let's compute delta_viol properly.
            # An edge (a,b) with color col contributes 1 to violations if |N[col][a] & N[col][b]| >= 2.

            # When we change edge (i,j) from c to new_c:
            # Affected edges are those where the common-neighbor count might change.
            # These are edges (i,v) and (j,v) for all v, plus (i,j) itself.
            # But actually, only edges where one endpoint is i or j AND the other endpoint's
            # neighbor set in the relevant color changes.

            # Let's use the direct approach: compute violations for all edges involving i or j.
            # This is O(n) per move, which is acceptable.

            def edge_violation(a, b, col, N):
                """Return 1 if edge (a,b) with color col is a violation, else 0."""
                if a == b:
                    return 0
                common = N[col][a] & N[col][b]
                return 1 if bin(common).count('1') >= 2 else 0

            # Old violations for edges involving i or j
            old_v = 0
            for v in range(n):
                if v == i or v == j:
                    continue
                # Edge (i,v)
                idx_iv = get_idx(min(i, v), max(i, v), n)
                col_iv = colors[idx_iv]
                old_v += edge_violation(i, v, col_iv, N)
                # Edge (j,v)
                idx_jv = get_idx(min(j, v), max(j, v), n)
                col_jv = colors[idx_jv]
                old_v += edge_violation(j, v, col_jv, N)
            # Edge (i,j)
            old_v += edge_violation(i, j, c, N)

            # Apply change temporarily
            N[c][i] &= ~(1 << j)
            N[c][j] &= ~(1 << i)
            N[new_c][i] |= (1 << j)
            N[new_c][j] |= (1 << i)
            colors[k] = new_c

            # New violations for edges involving i or j
            new_v = 0
            for v in range(n):
                if v == i or v == j:
                    continue
                idx_iv = get_idx(min(i, v), max(i, v), n)
                col_iv = colors[idx_iv]
                new_v += edge_violation(i, v, col_iv, N)
                idx_jv = get_idx(min(j, v), max(j, v), n)
                col_jv = colors[idx_jv]
                new_v += edge_violation(j, v, col_jv, N)
            new_v += edge_violation(i, j, new_c, N)

            delta = new_v - old_v

            # Accept if delta <= 0 or with probability exp(-delta/T)
            if delta <= 0 or random.random() < np.exp(-delta / T):
                cur_viol += delta
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best_colors = list(colors)
                if cur_viol == 0:
                    return (colors, 0)
            else:
                # Revert
                N[new_c][i] &= ~(1 << j)
                N[new_c][j] &= ~(1 << i)
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
                colors[k] = c

            T *= cooling
            if T < T_min:
                T = T_min

        return (best_colors, best_viol)

    # Start with a known valid coloring for small n, then try to grow
    # Known: the seed gives n=11. Let's try to find larger ones.

    # Strategy: Try n from 26 down to 12, with multiple restarts.
    # If we find a valid one, return it. Otherwise return the best we found.

    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]

    # Try to find valid colorings for n = 12, 13, ..., 26
    # Give more time to larger n since they're more valuable
    for n in range(12, 27):
        if time.time() > deadline:
            break

        # Time budget for this n
        time_left = deadline - time.time()
        if time_left < 2.0:
            break

        # Number of restarts based on time
        n_restarts = max(1, int(time_left / 3.0))
        n_restarts = min(n_restarts, 50)

        found = False
        for r in range(n_restarts):
            if time.time() > deadline:
                break

            # Use local search with random init
            max_iters = 50000
            colors, viol = local_search(n, max_iters, deadline)

            if viol == 0:
                best_n = n
                best_colors = colors
                found = True
                break

            # If we're close, try again with the best solution as start
            if viol <= 5 and r < n_restarts - 1:
                # Continue with this as starting point in next iteration
                pass

    return {'n': best_n, 'colors': best_colors}
