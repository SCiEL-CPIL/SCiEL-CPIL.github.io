import numpy as np
import itertools
import time
import random
import sys

def get_k6_cycles():
    """
    Generate all unique C6 edge-sets on vertices 0..5.
    """
    cycles = set()
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        v = (0,) + perm
        edges = set()
        for i in range(6):
            u, w = v[i], v[(i + 1) % 6]
            if u > w:
                u, w = w, u
            edges.add((u, w))
        cycles.add(frozenset(edges))
    return [list(c) for c in cycles]

K6_CYCLES = get_k6_cycles()


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def build_cycle_data(n):
        """
        Precompute for a given n:
          c6_edges_arr: shape (num_cycles, 6) of int32 global edge indices
          edge_to_cycles: list of lists mapping edge index -> [(cycle_idx, pos), ...]
          get_idx: function to map (i,j) to linear edge index
        """
        num_edges = n * (n - 1) // 2

        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

        c6_edges_list = []
        for comb in itertools.combinations(range(n), 6):
            for cycle_edges in K6_CYCLES:
                global_edges = []
                for u, v in cycle_edges:
                    gu = comb[u]
                    gv = comb[v]
                    if gu > gv:
                        gu, gv = gv, gu
                    global_edges.append(get_idx(gu, gv))
                c6_edges_list.append(global_edges)

        c6_edges_arr = np.array(c6_edges_list, dtype=np.int32)

        edge_to_cycles = [[] for _ in range(num_edges)]
        for ci, edges in enumerate(c6_edges_list):
            for pos, e in enumerate(edges):
                edge_to_cycles[e].append((ci, pos))

        return c6_edges_arr, edge_to_cycles, get_idx, num_edges

    def solve_n(n, time_limit, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        num_edges = n * (n - 1) // 2
        c6_edges_arr, edge_to_cycles, get_idx, num_edges = build_cycle_data(n)
        num_cycles = len(c6_edges_arr)

        if num_cycles == 0:
            return [0] * num_edges

        # Initialize: try a structured color assignment first (mod-based),
        # then random. Pick the best initial state.
        best_init = None
        best_init_score = float('inf')

        # Structured init: color(i,j) = (i + j) % 4
        colors_struct = np.empty(num_edges, dtype=np.int8)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                colors_struct[k] = (i + j) % 4
                k += 1
        # Also try (i * 3 + j) % 4
        colors_struct2 = np.empty(num_edges, dtype=np.int8)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                colors_struct2[k] = (i * 3 + j) % 4
                k += 1

        inits = [colors_struct, colors_struct2, np.random.randint(0, 4, num_edges, dtype=np.int8)]

        for init in inits:
            # Count monochromatic C6s
            mono_count = 0
            for ci in range(num_cycles):
                edges = c6_edges_arr[ci]
                c0 = 0
                for e in edges:
                    if init[e] == 0:
                        c0 += 1
                if c0 == 6:
                    mono_count += 1
                    continue
                c1 = 0
                for e in edges:
                    if init[e] == 1:
                        c1 += 1
                if c1 == 6:
                    mono_count += 1
                    continue
                c2 = 0
                for e in edges:
                    if init[e] == 2:
                        c2 += 1
                if c2 == 6:
                    mono_count += 1
                    continue
                c3 = 0
                for e in edges:
                    if init[e] == 3:
                        c3 += 1
                if c3 == 6:
                    mono_count += 1
            if mono_count < best_init_score:
                best_init_score = mono_count
                best_init = init.copy()

        if best_init_score == 0:
            return best_init.tolist()

        colors = best_init
        # Build cycle color counts
        cycle_color_count = np.zeros((num_cycles, 4), dtype=np.int32)
        for ci in range(num_cycles):
            edges = c6_edges_arr[ci]
            for e in edges:
                cycle_color_count[ci, colors[e]] += 1

        def get_current_score():
            return int(np.sum(np.max(cycle_color_count, axis=1) == 6))

        current_score = get_current_score()
        best_score = current_score
        best_colors = colors.copy()

        # Simulated Annealing with restarts
        T = 3.0
        cooling = 0.9997
        min_T = 0.05
        no_improve_count = 0
        max_no_improve = 5000

        while time.time() < start_time + time_limit:
            idx = np.random.randint(num_edges)
            old_color = colors[idx]
            new_color = np.random.randint(0, 4)
            if new_color == old_color:
                continue

            delta = 0
            affected = edge_to_cycles[idx]
            for (ci, pos) in affected:
                old_mono = 1 if cycle_color_count[ci].max() == 6 else 0
                cycle_color_count[ci, old_color] -= 1
                cycle_color_count[ci, new_color] += 1
                new_mono = 1 if cycle_color_count[ci].max() == 6 else 0
                delta += new_mono - old_mono

            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
                colors[idx] = new_color
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    no_improve_count = 0
                    if best_score == 0:
                        return best_colors.tolist()
                else:
                    no_improve_count += 1
            else:
                for (ci, pos) in affected:
                    cycle_color_count[ci, old_color] += 1
                    cycle_color_count[ci, new_color] -= 1

            T *= cooling
            if no_improve_count > max_no_improve:
                T = max(T * 2.5, 2.0)
                no_improve_count = 0
                if T > 5.0:
                    # Full restart
                    colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
                    cycle_color_count = np.zeros((num_cycles, 4), dtype=np.int32)
                    for ci in range(num_cycles):
                        edges = c6_edges_arr[ci]
                        for e in edges:
                            cycle_color_count[ci, colors[e]] += 1
                    current_score = get_current_score()
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors.copy()
                    T = 2.0
                    no_improve_count = 0

        if best_score == 0:
            return best_colors.tolist()
        return None

    # Strategy: try n=19 briefly, then n=18, then n=17, then n=16
    # n=19
    if time.time() < deadline - 85:
        res = solve_n(19, 5.0, seed=1001)
        if res is not None:
            return {'n': 19, 'colors': res}

    # n=18
    if time.time() < deadline - 20:
        t_start = time.time()
        time_limit = 70.0
        if deadline - t_start < time_limit:
            time_limit = deadline - t_start - 10.0
        if time_limit > 2.0:
            res = solve_n(18, time_limit, seed=2002)
            if res is not None:
                return {'n': 18, 'colors': res}

    # n=17
    if time.time() < deadline - 5:
        t_start = time.time()
        time_limit = 15.0
        if deadline - t_start < time_limit:
            time_limit = deadline - t_start - 2.0
        if time_limit > 1.0:
            res = solve_n(17, time_limit, seed=3003)
            if res is not None:
                return {'n': 17, 'colors': res}

    # n=16
    if time.time() < deadline - 2:
        t_start = time.time()
        time_limit = 5.0
        if deadline - t_start < time_limit:
            time_limit = deadline - t_start - 1.0
        if time_limit > 0.5:
            res = solve_n(16, time_limit, seed=4004)
            if res is not None:
                return {'n': 16, 'colors': res}

    # Hard fallback: n=9
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 4)
    return {'n': n, 'colors': colors}


if __name__ == "__main__":
    result = construct()
    print(f"Found n={result['n']}")
