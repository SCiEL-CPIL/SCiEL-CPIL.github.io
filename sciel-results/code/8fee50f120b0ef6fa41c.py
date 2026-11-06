import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_result = None

    def make_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    # Start with a known valid small solution as fallback
    best_result = make_fallback(6)

    def backtrack_search(n, time_budget, seed_offset=0):
        """Bitmask backtracking with incremental C6 detection."""
        if time.time() > deadline - 3:
            return None

        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        rng = random.Random(42 + seed_offset)

        # adj[c][v] = bitmask of neighbors of v in color c
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * num_edges

        def check_c6(u, v, c):
            """Check if adding edge (u,v) with color c creates a monochromatic C6."""
            nu = adj[c][u]
            nv = adj[c][v]

            # C6: u-x1-x2-x3-x4-v, all in color c
            # x1 in N(u), x4 in N(v), path x1-x2-x3-x4 in G_c
            # All 6 vertices distinct

            x1_mask = nu
            while x1_mask:
                lsb = x1_mask & (-x1_mask)
                x1 = lsb.bit_length() - 1
                x1_mask ^= lsb

                x4_mask = nv
                while x4_mask:
                    lsb4 = x4_mask & (-x4_mask)
                    x4 = lsb4.bit_length() - 1
                    x4_mask ^= lsb4

                    if x4 == u or x4 == x1:
                        continue

                    forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                    nx1 = adj[c][x1] & ~forbidden
                    nx4 = adj[c][x4] & ~forbidden

                    if nx1 == 0 or nx4 == 0:
                        continue

                    # Check if any x2 in nx1 connects to any x3 in nx4
                    vx2 = nx1
                    while vx2:
                        lsb2 = vx2 & (-vx2)
                        x2 = lsb2.bit_length() - 1
                        vx2 ^= lsb2
                        if adj[c][x2] & nx4:
                            return True

            return False

        def backtrack(idx):
            if time.time() > deadline - 3:
                return False

            if idx == num_edges:
                return True

            u, v = edges[idx]

            # Symmetry breaking for first few edges
            if idx == 0:
                candidates = [0]
            elif idx == 1:
                candidates = [0, 1]
                rng.shuffle(candidates)
            else:
                candidates = [0, 1, 2, 3]
                rng.shuffle(candidates)

            for c in candidates:
                if check_c6(u, v, c):
                    continue

                colors[idx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)

                if backtrack(idx + 1):
                    return True

                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[idx] = -1

            return False

        # Try multiple restarts
        max_attempts = 100000
        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break
            if attempt > 0 and time.time() > start_time + time_budget:
                break

            rng = random.Random(42 + seed_offset + attempt * 7919)
            adj = [[0] * n for _ in range(4)]
            colors = [-1] * num_edges

            if backtrack(0):
                return {'n': n, 'colors': colors}

        return None

    def sa_search(n, time_budget, seed_offset=0):
        """Simulated annealing with fast delta computation."""
        if time.time() > deadline - 3:
            return None

        m = n * (n - 1) // 2
        rng = random.Random(42 + seed_offset)
        np_rng = np.random.RandomState(42 + seed_offset)

        # Build cycle structure
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1

        def get_edge_idx(i, j):
            if i > j:
                i, j = j, i
            return edge_idx[(i, j)]

        # Generate all C6 cycles
        cycles_list = []
        for combo in itertools.combinations(range(n), 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(get_edge_idx(u, v))
                cycles_list.append(cycle_edges)

        if not cycles_list:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        cycles = np.array(cycles_list, dtype=np.int32)
        N = len(cycles)

        # Map edge to cycles containing it
        edge_to_cycles = [[] for _ in range(m)]
        for i in range(N):
            for e in cycles[i]:
                edge_to_cycles[e].append(i)

        # Precompute other edges for each edge
        edge_to_other_edges = [None] * m
        for e in range(m):
            cyc_ids = edge_to_cycles[e]
            if not cyc_ids:
                edge_to_other_edges[e] = np.array([], dtype=np.int32).reshape(0, 5)
                continue
            other_edges_list = []
            for cid in cyc_ids:
                ce = cycles[cid]
                mask = ce != e
                other_edges_list.append(ce[mask])
            edge_to_other_edges[e] = np.array(other_edges_list, dtype=np.int32)

        # Initial random coloring
        colors = np_rng.randint(0, 4, size=m).astype(np.uint8)

        def count_bad():
            if N == 0:
                return 0
            cycle_colors = colors[cycles]
            first = cycle_colors[:, 0]
            is_mono = np.all(cycle_colors == first[:, np.newaxis], axis=1)
            return int(np.sum(is_mono))

        bad_count = count_bad()
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        T_start = 1.0
        T_end = 0.05
        start_iter_time = time.time()

        while time.time() < start_iter_time + time_budget:
            if time.time() > deadline - 3:
                break

            e = int(np_rng.randint(0, m))
            c_old = colors[e]
            c_new = int(np_rng.randint(0, 4))
            while c_new == c_old:
                c_new = int(np_rng.randint(0, 4))

            other_edges = edge_to_other_edges[e]
            if len(other_edges) == 0:
                delta = 0
            else:
                other_colors = colors[other_edges]
                is_all_old = np.all(other_colors == c_old, axis=1)
                is_all_new = np.all(other_colors == c_new, axis=1)
                delta = int(np.sum(is_all_new) - np.sum(is_all_old))

            accept = False
            if delta <= 0:
                accept = True
            else:
                elapsed = time.time() - start_iter_time
                progress = min(1.0, elapsed / time_budget)
                T = T_start * (T_end / T_start) ** progress
                if delta / T > 50:
                    accept = False
                else:
                    prob = math.exp(-delta / T)
                    accept = rng.random() < prob

            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors.copy()
                    if best_bad == 0:
                        return {'n': n, 'colors': colors.tolist()}

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try targets from high to low
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    time_alloc = {17: 40, 16: 35, 15: 30, 14: 25, 13: 20, 12: 15, 11: 10, 10: 8, 9: 5, 8: 3, 7: 2, 6: 1}

    for n in targets:
        if time.time() > deadline - 10:
            break

        tb = time_alloc.get(n, 5)

        # Try backtracking first (more effective for this problem)
        bt_budget = tb * 0.6
        result = backtrack_search(n, bt_budget, seed_offset=n * 100)
        if result:
            best_result = result
            return best_result

        # Then try SA
        sa_budget = tb * 0.4
        if time.time() < deadline - 10:
            result = sa_search(n, sa_budget, seed_offset=n * 200)
            if result:
                best_result = result
                return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
