import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit

    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        e1 = edge_idx[(a, b)]
        e2 = edge_idx[(a, c)]
        e3 = edge_idx[(a, d)]
        e4 = edge_idx[(b, c)]
        e5 = edge_idx[(b, d)]
        e6 = edge_idx[(c, d)]
        k4s.append((e1, e2, e3, e4, e5, e6))

    k4_edges_np = np.array(k4s, dtype=np.int32)  # shape (1820, 6)

    def count_violations(colours):
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np]
        sorted_colors = np.sort(k4_colors, axis=1)
        unique_mask = np.ones((1820, 6), dtype=bool)
        unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
        counts = np.sum(unique_mask, axis=1)
        return int(np.sum(counts < 5))

    def is_valid(colours):
        return count_violations(colours) == 0

    # Start with seed (i+j) % 17
    current = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]

    if not is_valid(current):
        for _ in range(100):
            if time.time() > deadline:
                break
            rand = [random.randint(0, 16) for _ in range(120)]
            if is_valid(rand):
                current = rand
                break

    def local_search(colours, K, max_steps=5000):
        """Target violated K4s and try all colour swaps for affected edges."""
        colours = list(colours)
        v = count_violations(colours)
        if v == 0:
            return colours, 0
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if v == 0:
                break
            # Recompute violated K4s
            c_np = np.asarray(colours, dtype=np.int32)
            k4_colors = c_np[k4_edges_np]
            sorted_colors = np.sort(k4_colors, axis=1)
            unique_mask = np.ones((1820, 6), dtype=bool)
            unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
            counts = np.sum(unique_mask, axis=1)
            violated = np.where(counts < 5)[0]
            if len(violated) == 0:
                v = 0
                break
            # Pick a random violated K4
            k4_idx = violated[random.randint(0, len(violated) - 1)]
            k4_edge_idxs = k4_edges_np[k4_idx]
            # Try all swaps: for each edge in K4, try each of K colors
            best_new_v = v
            best_edge = -1
            best_color = -1
            for eidx in k4_edge_idxs:
                old_c = colours[eidx]
                for new_c in range(K):
                    if new_c == old_c:
                        continue
                    colours[eidx] = new_c
                    new_v = count_violations(colours)
                    if new_v < best_new_v:
                        best_new_v = new_v
                        best_edge = eidx
                        best_color = new_c
                    colours[eidx] = old_c
            if best_edge >= 0:
                colours[best_edge] = best_color
                v = best_new_v
            else:
                # No improvement in this K4; random perturbation
                eidx = random.randint(0, 119)
                colours[eidx] = random.randint(0, K - 1)
                v = count_violations(colours)
        return colours, v

    for target_max_colors in range(16, 12, -1):
        if time.time() > deadline:
            break

        if len(set(current)) <= target_max_colors:
            continue

        current_set = set(current)
        colors_list = sorted(current_set)

        if len(colors_list) <= target_max_colors:
            continue

        merged = False
        for i in range(len(colors_list)):
            for j in range(i + 1, len(colors_list)):
                if time.time() > deadline:
                    break
                c1 = colors_list[i]
                c2 = colors_list[j]
                new_colours = [c2 if c == c1 else c for c in current]
                if is_valid(new_colours):
                    current = new_colours
                    merged = True
                    break
            if merged:
                break

        if not merged:
            K = target_max_colors
            mapping = {}
            next_id = 0
            init_colours = []
            for c in current:
                if c not in mapping:
                    if next_id >= K:
                        break
                    mapping[c] = next_id
                    next_id += 1
                init_colours.append(mapping[c])

            if len(init_colours) < 120:
                continue

            if is_valid(init_colours):
                current = init_colours
                continue

            # Local search on init_colours
            best_colours, best_violations = local_search(init_colours, K)
            if best_violations == 0:
                current = best_colours
                continue

            # Random restarts with local search
            success = False
            for restart in range(30):
                if time.time() > deadline:
                    break
                rand_colours = [random.randint(0, K - 1) for _ in range(120)]
                rand_colours, v = local_search(rand_colours, K)
                if v == 0:
                    current = rand_colours
                    success = True
                    break

            if not success:
                break

    if not is_valid(current):
        seed = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        if is_valid(seed):
            current = seed

    return current
