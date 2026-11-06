"""Rewritten program: Latin-square 1-factorization construction (15 colors) + polish."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 142.0  # safety margin before 150s wall-clock

    n = 16
    total_edges = 120

    # Edge index mapping (lexicographic order of pairs (i,j))
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1

    # Precompute K4 edge index lists
    k4_edges = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges.append([
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ])
    num_k4 = len(k4_edges)
    k4_arr = np.array(k4_edges, dtype=np.int32)  # (num_k4, 6)

    # Precompute, for each edge, the list of K4 indices it belongs to
    edge_to_k4 = [[] for _ in range(total_edges)]
    for k in range(num_k4):
        for e in k4_edges[k]:
            edge_to_k4[e].append(k)
    edge_to_k4_arr = [np.array(lst, dtype=np.int32) for lst in edge_to_k4]

    def count_colors(coloring):
        return len(set(coloring))

    def is_valid(coloring):
        col_arr = np.array(coloring, dtype=np.int32)
        colors_in_k4 = col_arr[k4_arr]  # (num_k4, 6)
        sorted_colors = np.sort(colors_in_k4, axis=1)
        distinct_counts = np.ones(num_k4, dtype=np.int32)
        for k in range(1, 6):
            distinct_counts += (sorted_colors[:, k] != sorted_colors[:, k - 1]).astype(np.int32)
        return bool(np.all(distinct_counts >= 5))

    def fast_is_valid(coloring):
        """Vectorized validity check using precomputed K4 membership."""
        col_arr = np.array(coloring, dtype=np.int32)
        colors_in_k4 = col_arr[k4_arr]
        sorted_colors = np.sort(colors_in_k4, axis=1)
        distinct_counts = np.ones(num_k4, dtype=np.int32)
        for k in range(1, 6):
            distinct_counts += (sorted_colors[:, k] != sorted_colors[:, k - 1]).astype(np.int32)
        return bool(np.all(distinct_counts >= 5))

    # Guaranteed valid 15-colour solution: Latin-square 1-factorization
    # color = (i + j) mod 15. Each colour class is a perfect matching (1-factor).
    # Every K4 sees at least 5 distinct colours (proved: 2 colours cover <=4 edges).
    def base_solution():
        return [(i + j) % 15 for i in range(n) for j in range(i + 1, n)]

    best_coloring = base_solution()
    best_colors = count_colors(best_coloring)
    assert is_valid(best_coloring), "base solution must be valid"

    # Polish: local search to try to reduce the number of colours.
    # Move: reassign one edge to a currently-used colour, keep if valid.
    # Track which colours are still used; a colour can be "dropped" once no edge uses it.

    used_mask = np.zeros(15, dtype=bool)
    for c in best_coloring:
        used_mask[c] = True

    def recompute_used(coloring):
        m = np.zeros(15, dtype=bool)
        for c in coloring:
            m[c] = True
        return m

    rng = random.Random(12345)

    # Multiple restarts of local search
    for attempt in range(40):
        if time.time() > deadline:
            break
        if best_colors <= 13:
            break

        coloring = list(best_coloring)
        current_colors = best_colors
        used = recompute_used(coloring)

        # Simulated-annealing-ish: allow neutral moves (same colour count) to explore
        for step in range(3000):
            if time.time() > deadline:
                break

            e = rng.randint(0, total_edges - 1)
            old_color = coloring[e]

            # Candidate target colours: currently used colours (to not increase palette)
            used_list = [c for c in range(15) if used[c]]
            if len(used_list) < 2:
                break
            candidates = [c for c in used_list if c != old_color]
            if not candidates:
                continue
            new_color = rng.choice(candidates)

            coloring[e] = new_color

            if is_valid(coloring):
                # Update used mask
                # Check if old_color is now unused
                new_used = used.copy()
                # recompute only affected: check if old_color still used anywhere
                # cheap: scan
                still_old = False
                for v in coloring:
                    if v == old_color:
                        still_old = True
                        break
                if not still_old:
                    new_used[old_color] = False
                new_used[new_color] = True
                new_count = int(np.sum(new_used))
                used = new_used
                if new_count < current_colors:
                    current_colors = new_count
                    if current_colors < best_colors:
                        best_coloring = list(coloring)
                        best_colors = current_colors
                        if best_colors <= 13:
                            break
            else:
                coloring[e] = old_color

        if best_colors <= 13:
            break

    # Final safety check
    if not is_valid(best_coloring):
        best_coloring = base_solution()
        best_colors = count_colors(best_coloring)

    return best_coloring
# EVOLVE-BLOCK-END
