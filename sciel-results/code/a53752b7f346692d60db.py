import numpy as np
import time
import random
import math

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    # Precompute triangle indices
    tri_list = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_list.append((i, j, k))
    tri_arr = np.array(tri_list)
    I = tri_arr[:, 0]
    J = tri_arr[:, 1]
    K = tri_arr[:, 2]
    n_tri = len(tri_arr)

    A = list(range(8))
    B = list(range(8, 16))
    all_cross = [(u, v) for u in A for v in B]
    all_int_A = list([])
    for i in range(8):
        for j in range(i + 1, 8):
            all_int_A.append((i, j))
    all_int_B = list([])
    for i in range(8, 16):
        for j in range(i + 1, 16):
            all_int_B.append((i, j))

    def get_score(adj):
        deg = adj.sum(axis=1)
        valid = (adj[I, J] == 1) & (adj[J, K] == 1) & (adj[K, I] == 1)
        if not np.any(valid):
            return 0
        sums = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
        return int(np.max(sums))

    def build_adj(removed_cross, int_A, int_B):
        adj = np.zeros((n, n), dtype=np.int8)
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        for u, v in removed_cross:
            adj[u, v] = 0
            adj[v, u] = 0
        for u, v in int_A:
            adj[u, v] = 1
            adj[v, u] = 1
        for u, v in int_B:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj

    # Phase 1: Systematic search over small perturbations of K_{8,8}
    best_adj = None
    best_score = 999

    # k=0: baseline (64 cross + 1 internal = 65)
    for u, v in all_int_A:
        adj = build_adj([], [(u, v)], [])
        if adj.sum() // 2 == target_edges:
            s = get_score(adj)
            if s < best_score:
                best_score = s
                best_adj = adj.copy()

    for u, v in all_int_B:
        adj = build_adj([], [], [(u, v)])
        if adj.sum() // 2 == target_edges:
            s = get_score(adj)
            if s < best_score:
                best_score = s
                best_adj = adj.copy()

    # k=1: remove 1 cross, add 2 internal
    if time.time() < deadline - 60:
        for rc in all_cross:
            for ia in all_int_A:
                for ib in all_int_B:
                    adj = build_adj([rc], [ia], [ib])
                    if adj.sum() // 2 == target_edges:
                        s = get_score(adj)
                        if s < best_score:
                            best_score = s
                            best_adj = adj.copy()
            # Also try 2 in A
            for i1 in range(len(all_int_A)):
                for i2 in range(i1 + 1, len(all_int_A)):
                    adj = build_adj([rc], [all_int_A[i1], all_int_A[i2]], [])
                    if adj.sum() // 2 == target_edges:
                        s = get_score(adj)
                        if s < best_score:
                            best_score = s
                            best_adj = adj.copy()
            # Also try 2 in B
            for i1 in range(len(all_int_B)):
                for i2 in range(i1 + 1, len(all_int_B)):
                    adj = build_adj([rc], [], [all_int_B[i1], all_int_B[i2]])
                    if adj.sum() // 2 == target_edges:
                        s = get_score(adj)
                        if s < best_score:
                            best_score = s
                            best_adj = adj.copy()
            if time.time() > deadline - 60:
                break

    # k=2: remove 2 cross, add 3 internal
    if time.time() < deadline - 40:
        cross_idx = list(range(len(all_cross)))
        random.seed(42)
        random.shuffle(cross_idx)
        # Try a subset of pairs to save time
        for idx in range(min(500, len(cross_idx) * (len(cross_idx) - 1) // 2)):
            # Generate unique pair indices
            i1 = idx % len(cross_idx)
            i2 = (idx // len(cross_idx)) % len(cross_idx)
            if i1 == i2:
                continue
            rc1 = all_cross[i1]
            rc2 = all_cross[i2]
            removed = [rc1, rc2]
            # Try various internal combinations
            for ia in all_int_A:
                for ib in all_int_B:
                    adj = build_adj(removed, [ia], [ib])
                    if adj.sum() // 2 == target_edges:
                        s = get_score(adj)
                        if s < best_score:
                            best_score = s
                            best_adj = adj.copy()
            if time.time() > deadline - 40:
                break

    # Phase 2: Vectorized Simulated Annealing
    if best_adj is None:
        best_adj = build_adj([], [(0, 1)], [])
        best_score = get_score(best_adj)

    current_adj = best_adj.copy()
    current_score = best_score

    # Precompute edge lists
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    edges.append((i, j))
        return edges

    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    non_edges.append((i, j))
        return non_edges

    edges = get_edges(current_adj)
    non_edges = get_non_edges(current_adj)

    # Simulated annealing parameters
    T = 4.0
    cooling = 0.9999
    min_T = 0.1
    rng = random.Random(12345)

    while time.time() < deadline - 20:
        # Choose move type
        r = rng.random()

        if r < 0.4:
            # Remove a cross edge, add an internal edge
            cross_edges = [(u, v) for u, v in all_cross if current_adj[u, v] == 1]
            if not cross_edges:
                continue
            u, v = cross_edges[rng.randint(0, len(cross_edges) - 1)]

            int_options = [(a, b) for a, b in all_int_A if current_adj[a, b] == 0] + \
                         [(a, b) for a, b in all_int_B if current_adj[a, b] == 0]
            if not int_options:
                continue
            x, y = int_options[rng.randint(0, len(int_options) - 1)]

            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_adj[x, y] = 1
            current_adj[y, x] = 1

            new_score = get_score(current_adj)

            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
            else:
                delta = new_score - current_score
                if rng.random() < math.exp(-delta / T):
                    current_score = new_score
                else:
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0

        elif r < 0.7:
            # Remove an internal edge, add a cross edge
            int_edges = [(a, b) for a, b in all_int_A if current_adj[a, b] == 1] + \
                        [(a, b) for a, b in all_int_B if current_adj[a, b] == 1]
            if not int_edges:
                continue
            x, y = int_edges[rng.randint(0, len(int_edges) - 1)]

            cross_options = [(u, v) for u, v in all_cross if current_adj[u, v] == 0]
            if not cross_options:
                continue
            u, v = cross_options[rng.randint(0, len(cross_options) - 1)]

            current_adj[x, y] = 0
            current_adj[y, x] = 0
            current_adj[u, v] = 1
            current_adj[v, u] = 1

            new_score = get_score(current_adj)

            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
            else:
                delta = new_score - current_score
                if rng.random() < math.exp(-delta / T):
                    current_score = new_score
                else:
                    current_adj[x, y] = 1
                    current_adj[y, x] = 1
                    current_adj[u, v] = 0
                    current_adj[v, u] = 0

        else:
            # General edge swap
            if not edges or not non_edges:
                continue
            u, v = edges[rng.randint(0, len(edges) - 1)]
            x, y = non_edges[rng.randint(0, len(non_edges) - 1)]

            if (u == x and v == y) or (u == y and v == x):
                continue

            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_adj[x, y] = 1
            current_adj[y, x] = 1

            new_score = get_score(current_adj)

            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
            else:
                delta = new_score - current_score
                if rng.random() < math.exp(-delta / T):
                    current_score = new_score
                else:
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0

        T *= cooling
        if T < min_T:
            T = min_T

    # Phase 3: Exhaustive local search on best_adj
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            edges = get_edges(best_adj)
            non_edges = get_non_edges(best_adj)

            for u, v in edges:
                if time.time() > deadline - 5:
                    break
                for x, y in non_edges:
                    if x == u or x == v or y == u or y == v:
                        continue

                    best_adj[u, v] = 0
                    best_adj[v, u] = 0
                    best_adj[x, y] = 1
                    best_adj[y, x] = 1

                    new_score = get_score(best_adj)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                    else:
                        best_adj[u, v] = 1
                        best_adj[v, u] = 1
                        best_adj[x, y] = 0
                        best_adj[y, x] = 0

    # Return result
    edges_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                edges_list.append([i, j])

    return {'n': n, 'edges': edges_list}
