import time
import random
import numpy as np
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all C6 edge-index arrays for a given n
    def precompute_c6s(n):
        # edge index map
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1
        total_edges = idx

        # 5 distinct Hamiltonian cycles on 6 labeled vertices (up to dihedral symmetry)
        # Vertices labeled 0..5, cycles as sequences of edges
        patterns = [
            [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)],
            [(0,1),(1,2),(2,4),(4,3),(3,5),(5,0)],
            [(0,1),(1,3),(3,2),(2,4),(4,5),(5,0)],
            [(0,1),(1,3),(3,4),(4,2),(2,5),(5,0)],
            [(0,1),(1,4),(4,2),(2,3),(3,5),(5,0)],
        ]

        c6_edge_indices = []  # list of arrays of 6 edge indices
        for comb in combinations(range(n), 6):
            v0, v1, v2, v3, v4, v5 = comb
            for pat in patterns:
                indices = []
                for (a, b) in pat:
                    u, v = (a, b) if a < b else (b, a)
                    indices.append(edge_idx[(u, v)])
                c6_edge_indices.append(np.array(indices, dtype=np.int32))

        c6_array = np.array(c6_edge_indices, dtype=np.int32)  # shape (num_c6s, 6)
        num_c6s = len(c6_edge_indices)

        # For each edge, list of C6 indices containing it
        edge_to_c6s = [[] for _ in range(total_edges)]
        for i in range(num_c6s):
            for e in c6_array[i]:
                edge_to_c6s[e].append(i)
        edge_to_c6s_arrays = [np.array(lst, dtype=np.int32) for lst in edge_to_c6s]

        return c6_array, edge_to_c6s_arrays, total_edges

    def solve(n, max_time):
        c6_array, edge_to_c6s_arrays, total_edges = precompute_c6s(n)
        num_c6s = len(c6_array)

        if num_c6s == 0:
            return np.zeros(total_edges, dtype=np.int32)

        # Precompute: for each edge, the C6 indices and the other 5 edge positions
        # For efficient delta: for edge e, C6s containing e are edge_to_c6s_arrays[e]
        # For each such C6, the other 5 edges are c6_array[c6_idx] excluding e

        # Build a structured lookup: for each edge e, store (c6_indices, other_edges_2d)
        # other_edges_2d has shape (num_c6s_containing_e, 5)
        edge_other_edges = []
        for e in range(total_edges):
            c6_indices = edge_to_c6s_arrays[e]
            if len(c6_indices) == 0:
                edge_other_edges.append((c6_indices, np.empty((0, 5), dtype=np.int32)))
            else:
                # Get all 6 edges for these C6s
                all_edges = c6_array[c6_indices]  # shape (k, 6)
                # Find position of e in each row
                # e is in each row exactly once
                pos = (all_edges == e).nonzero()[1]  # column index of e in each row
                # Build other edges: for each row, take all columns except pos
                # Use advanced indexing
                rows = np.arange(len(c6_indices))
                cols = np.arange(6)
                mask = np.ones((len(c6_indices), 6), dtype=bool)
                mask[rows, pos] = False
                other = all_edges[mask].reshape(len(c6_indices), 5)
                edge_other_edges.append((c6_indices, other))

        # Simulated annealing
        colors = np.random.randint(0, 4, size=total_edges, dtype=np.int32)

        # Initial cost
        def compute_cost(col):
            # For each C6, check if all 6 edges have same color
            c = col[c6_array]  # shape (num_c6s, 6)
            # monochromatic if all equal to first element
            mono = np.all(c == c[:, 0:1], axis=1)
            return int(np.sum(mono))

        cost = compute_cost(colors)
        best_cost = cost
        best_colors = colors.copy()

        if cost == 0:
            return colors

        # Temperature schedule
        T_start = 2.0
        T_end = 0.01
        # We'll do as many iterations as possible

        np_rng = np.random.default_rng(42)

        iter_count = 0
        while time.time() < max_time:
            # Batch: pick a batch of random edges to try
            # For speed, process edges one at a time but with vectorized delta

            # Pick random edge
            e = int(np_rng.integers(0, total_edges))
            old_color = int(colors[e])

            c6_indices, other_edges = edge_other_edges[e]
            k = len(c6_indices)

            if k == 0:
                # No C6s contain this edge, any color change is neutral
                new_color = int(np_rng.integers(0, 4))
                if new_color != old_color:
                    colors[e] = new_color
                continue

            # For each candidate new color, compute delta
            # delta = (# C6s that become mono new_color) - (# C6s that were mono old_color)
            # A C6 becomes mono new_color iff all 5 other edges are new_color
            # A C6 was mono old_color iff all 5 other edges are old_color

            # other_edges shape: (k, 5)
            # For new_color c: mono_new = all(other_edges == c, axis=1)
            # For old_color: mono_old = all(other_edges == old_color, axis=1)

            mono_old = np.all(other_edges == old_color, axis=1)
            num_mono_old = int(np.sum(mono_old))

            best_delta = 0
            best_new_color = -1
            for new_color in range(4):
                if new_color == old_color:
                    continue
                mono_new = np.all(other_edges == new_color, axis=1)
                num_mono_new = int(np.sum(mono_new))
                delta = num_mono_new - num_mono_old
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color

            # Accept best move or random move with probability based on temperature
            # Compute temperature
            progress = min(1.0, iter_count / 200000.0)
            T = T_start * (T_end / T_start) ** progress

            if best_new_color != -1:
                # Accept best move always if improving or neutral
                if best_delta <= 0:
                    colors[e] = best_new_color
                    cost += best_delta
                    if cost < best_cost:
                        best_cost = cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return colors
                else:
                    # Accept with probability exp(-delta/T)
                    if np_rng.random() < np.exp(-best_delta / T):
                        colors[e] = best_new_color
                        cost += best_delta
            else:
                # No improving move; do a random perturbation
                new_color = int(np_rng.integers(0, 4))
                if new_color != old_color:
                    mono_new = np.all(other_edges == new_color, axis=1)
                    delta = int(np.sum(mono_new)) - num_mono_old
                    colors[e] = new_color
                    cost += delta
                    if cost < best_cost:
                        best_cost = cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return colors

            iter_count += 1

            # Occasionally restart if stuck
            if iter_count % 50000 == 0 and best_cost > 0:
                # Check if we're stuck
                if cost > best_cost * 1.5 + 10:
                    # Reset to best and add noise
                    colors = best_colors.copy()
                    cost = best_cost
                    # Add random noise
                    num_flips = max(1, total_edges // 20)
                    flip_indices = np_rng.choice(total_edges, size=num_flips, replace=False)
                    for fi in flip_indices:
                        colors[fi] = int(np_rng.integers(0, 4))
                    cost = compute_cost(colors)
                    if cost < best_cost:
                        best_cost = cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return colors

        # Return best found if it's valid, else None
        if best_cost == 0:
            return best_colors
        return None

    # Try n=17 first with most of the time
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol = solve(17, remaining - 2.0)
        if sol is not None:
            return {'n': 17, 'colors': sol.tolist()}

    # Try n=16
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol = solve(16, remaining - 2.0)
        if sol is not None:
            return {'n': 16, 'colors': sol.tolist()}

    # Try n=15
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol = solve(15, remaining - 2.0)
        if sol is not None:
            return {'n': 15, 'colors': sol.tolist()}

    # Fallback: n=9 with a simple valid coloring
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 4)
    return {'n': n, 'colors': colors}
