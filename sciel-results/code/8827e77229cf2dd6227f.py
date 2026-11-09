"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    random.seed(42)

    def _count_c6s(adj, n):
        """Count monochromatic C6s in the coloring."""
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)

        count = 0
        for c in range(4):
            for subset in combinations(range(n), 6):
                if not hasattr(_count_c6s, '_cycles'):
                    cycles = []
                    seen = set()
                    for perm in permutations(range(1, 6)):
                        cycle = (0,) + perm
                        edges = frozenset(frozenset((cycle[k], cycle[(k + 1) % 6])) for k in range(6))
                        if edges not in seen:
                            seen.add(edges)
                            cycles.append(cycle)
                    _count_c6s._cycles = cycles

                for cycle in _count_c6s._cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k + 1) % 6]]
                        if adj[u][w] != c:
                            ok = False
                            break
                    if ok:
                        count += 1
                        break
        return count

    def _local_search(n, max_iters=5000):
        """Try to find a valid 4-coloring of K_n with no monochromatic C6."""
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                adj[i][j] = random.randint(0, 3)
                adj[j][i] = adj[i][j]

        best_count = _count_c6s(adj, n)
        if best_count == 0:
            return adj

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue

            old_c = adj[i][j]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue

            adj[i][j] = new_c
            adj[j][i] = new_c
            new_count = _count_c6s(adj, n)

            if new_count < best_count:
                best_count = new_count
                if best_count == 0:
                    return adj
            else:
                # Revert
                adj[i][j] = old_c
                adj[j][i] = old_c

        return adj if best_count == 0 else None

    best_n = 1
    best_colors = []

    for n_target in range(6, 18):
        if time.time() > deadline - 10:
            break

        # Try multiple random restarts
        found = None
        attempts = 0
        while time.time() < deadline - 15 and attempts < 20:
            adj = _local_search(n_target, max_iters=2000)
            if adj is not None:
                found = adj
                break
            attempts += 1

        if found is not None:
            colors = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    colors.append(int(found[i][j]))
            best_n = n_target
            best_colors = colors

    if best_n < 6:
        # Fallback: return a trivially valid small coloring
        best_n = 5
        best_colors = [0] * 10

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
