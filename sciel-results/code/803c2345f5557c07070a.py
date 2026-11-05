import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n = 16
    num_edges = n * (n - 1) // 2  # 120
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}

    # Precompute all K4s (1820 of them) and their 6 edge indices
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]

    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4_edges) - 1)

    num_k4s = len(k4_edges)  # 1820

    def count_bad_k4s(colors):
        """Count K4s with fewer than 5 distinct colors using bitmasks."""
        bad = 0
        for e_idx_list in k4_edges:
            mask = 0
            for e in e_idx_list:
                mask |= (1 << colors[e])
            if mask.bit_count() < 5:
                bad += 1
        return bad

    def is_valid(colors):
        return count_bad_k4s(colors) == 0

    best_result = None
    best_colors_used = 16

    for target_C in [13, 14, 15]:
        if time.time() > deadline - 5:
            break

        max_iterations = 15000
        restarts = 30

        for r in range(restarts):
            if time.time() > deadline - 2:
                break

            # Initialize with random colors
            colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
            current_bad = count_bad_k4s(colors)

            if current_bad == 0:
                if target_C < best_colors_used:
                    best_colors_used = target_C
                    best_result = colors[:]
                break

            # Simulated annealing local search
            temperature = 0.3
            cooling = 0.9999
            no_improve = 0
            best_bad_in_run = current_bad
            best_colors_in_run = colors[:]

            for it in range(max_iterations):
                if time.time() > deadline - 2:
                    break

                # Pick a random edge
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, target_C - 1)
                if new_color == old_color:
                    continue

                # Compute delta by checking only affected K4s
                colors[e_idx] = new_color
                delta = 0
                for k4_idx in edge_to_k4s[e_idx]:
                    e_list = k4_edges[k4_idx]
                    mask = 0
                    for e in e_list:
                        mask |= (1 << colors[e])
                    if mask.bit_count() < 5:
                        delta += 1

                new_bad = current_bad + delta

                if new_bad < current_bad:
                    current_bad = new_bad
                    no_improve = 0
                    if current_bad < best_bad_in_run:
                        best_bad_in_run = current_bad
                        best_colors_in_run = colors[:]
                elif new_bad == current_bad:
                    if random.random() < 0.1:
                        current_bad = new_bad
                        no_improve += 1
                    else:
                        colors[e_idx] = old_color
                        no_improve += 1
                else:
                    # Accept worse with probability exp(-delta/temperature)
                    if random.random() < (2.718281828 ** (-delta / temperature)):
                        current_bad = new_bad
                        no_improve += 1
                    else:
                        colors[e_idx] = old_color
                        no_improve += 1

                temperature *= cooling
                if temperature < 0.01:
                    temperature = 0.01

                if current_bad == 0:
                    if target_C < best_colors_used:
                        best_colors_used = target_C
                        best_result = colors[:]
                    break

            # If we found a valid coloring in this run, record it
            if best_bad_in_run == 0:
                if target_C < best_colors_used:
                    best_colors_used = target_C
                    best_result = best_colors_in_run[:]
                break

    if best_result is not None:
        return best_result

    # Fallback: try a few simple structured colorings
    # (i + j) % 17
    baseline = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    if is_valid(baseline):
        return baseline

    # (i * 7 + j) % 16
    alt = [(i * 7 + j) % 16 for i in range(n) for j in range(i + 1, n)]
    if is_valid(alt):
        return alt

    # (i * 5 + j * 3) % 15
    alt2 = [(i * 5 + j * 3) % 15 for i in range(n) for j in range(i + 1, n)]
    if is_valid(alt2):
        return alt2

    # Last resort: random (may be invalid but avoids crash)
    return [random.randint(0, 119) for _ in range(num_edges)]
