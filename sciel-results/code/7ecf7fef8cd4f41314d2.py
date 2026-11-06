"""Edge-colourings of K16: minimize colours so every K4 sees >=5 colours."""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx

    # Precompute all K4s and their edge index lists
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        e_list = [edge_idx[(a, b)] for a, b in combinations(k4, 2)]
        k4_edges.append(np.array(e_list, dtype=np.int32))
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)  # shape (1820, 6)

    # For each edge, precompute which K4s contain it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e in e_list:
            edge_to_k4s[e].append(k4_idx)
    edge_to_k4s_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in edge_to_k4s]

    def count_distinct(c):
        return len(np.unique(c))

    def check_violations(c):
        """Return number of K4s with < 5 distinct colors."""
        # For each K4, count distinct colors among its 6 edges
        # c[k4_edges_arr] gives shape (1820, 6)
        colors_in_k4 = c[k4_edges_arr]  # (1820, 6)
        # Count distinct per row
        # A row has < 5 distinct if it has <= 4 distinct values
        # Use: distinct = 6 - (number of positions equal to first) won't work generally
        # Better: for each row, check if all 6 are same, 5 same+1, 4+2, 3+3, 3+2+1, etc.
        # Simplest: use np.unique per row - but that's slow.
        # Alternative: a row has >= 5 distinct iff it's NOT the case that
        # at most 4 distinct values appear.
        # Let's use: sort each row and count distinct.
        sorted_c = np.sort(colors_in_k4, axis=1)
        # distinct count = 1 + sum(sorted_c[:,1:] != sorted_c[:,:-1])
        distinct = 1 + np.sum(sorted_c[:, 1:] != sorted_c[:, :-1], axis=1)
        return np.sum(distinct < 5)

    def is_valid(c):
        return check_violations(c) == 0

    # Initial construction: use a base with fewer colors
    # Strategy: assign colors based on a 13-color scheme
    # Use modular construction but with 13 colors
    # Try: color = (i + j) % 13
    colors = np.array([(i + j) % 13 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)

    # Check if valid
    if not is_valid(colors):
        # Try (i + j) % 14
        colors = np.array([(i + j) % 14 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    if not is_valid(colors):
        # Try (i + j) % 15
        colors = np.array([(i + j) % 15 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    if not is_valid(colors):
        # Try (i + j) % 16
        colors = np.array([(i + j) % 16 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    if not is_valid(colors):
        # Fallback: (i + j) % 17
        colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)

    best_colors = colors.copy()
    best_count = count_distinct(best_colors)

    # Aggressive color merging: try to merge pairs of colors
    def try_merge(c, color_a, color_b):
        """Try to merge color_b into color_a."""
        if color_a == color_b:
            return None
        new_c = c.copy()
        new_c[new_c == color_b] = color_a
        if is_valid(new_c):
            return new_c
        return None

    # Iterative merging phase
    for iteration in range(500):
        if time.time() > deadline - 30:
            break

        current_colors = np.unique(best_colors)
        if len(current_colors) <= 13:
            break

        # Count frequency
        color_freq = np.bincount(best_colors, minlength=120)
        used_colors = np.where(color_freq > 0)[0]

        # Sort by frequency (rarest first)
        sorted_colors = used_colors[np.argsort(color_freq[used_colors])]

        merged = False
        for target in sorted_colors:
            if time.time() > deadline - 30:
                break
            if color_freq[target] == 0:
                continue
            for donor in sorted_colors:
                if donor == target:
                    continue
                if color_freq[donor] == 0:
                    continue
                new_c = try_merge(best_colors, donor, target)
                if new_c is not None:
                    best_colors = new_c
                    best_count = count_distinct(best_colors)
                    merged = True
                    break
            if merged:
                break

        if not merged:
            # Try random edge reassignment to enable future merges
            existing = np.unique(best_colors)
            if len(existing) > 0:
                num_reassign = random.randint(1, 8)
                new_c = best_colors.copy()
                for _ in range(num_reassign):
                    edge_pos = random.randint(0, num_edges - 1)
                    new_c[edge_pos] = random.choice(existing)
                if is_valid(new_c):
                    best_colors = new_c
                    best_count = count_distinct(best_colors)

    # Simulated annealing phase to reduce colors further
    def annealing(c, start_temp, end_temp, steps):
        current = c.copy()
        current_count = count_distinct(current)
        best = current.copy()
        best_count = current_count

        for step in range(steps):
            if time.time() > deadline - 10:
                break
            temp = start_temp * (end_temp / start_temp) ** (step / steps)

            # Pick a random edge and change its color
            edge_pos = random.randint(0, num_edges - 1)
            old_color = current[edge_pos]
            # Pick a random new color from existing colors or a new one
            existing = np.unique(current)
            if random.random() < 0.7:
                new_color = random.choice(existing)
            else:
                # Try a color not currently used (to potentially introduce then merge)
                max_color = int(existing.max()) + 1 if len(existing) > 0 else 0
                new_color = random.randint(0, max_color + 1)

            current[edge_pos] = new_color
            new_count = count_distinct(current)
            violations = check_violations(current)

            # Energy: prefer fewer colors, penalize violations heavily
            energy = new_count + 100 * violations
            best_energy = best_count + 100 * check_violations(best)
            old_energy = current_count + 100 * check_violations(current)

            if energy < old_energy or (violations == 0 and random.random() < np.exp(-(energy - old_energy) / max(temp, 1e-10))):
                current_count = new_count
                if violations == 0 and new_count < best_count:
                    best = current.copy()
                    best_count = new_count
            else:
                # Reject
                current[edge_pos] = old_color

        return best, best_count

    # Run annealing if we have time
    if time.time() < deadline - 20 and best_count > 13:
        best_colors, best_count = annealing(best_colors, 5.0, 0.01, 50000)

    # Final merge pass
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        current_colors = np.unique(best_colors)
        if len(current_colors) <= 13:
            break
        color_freq = np.bincount(best_colors, minlength=120)
        used_colors = np.where(color_freq > 0)[0]
        sorted_colors = used_colors[np.argsort(color_freq[used_colors])]
        merged = False
        for target in sorted_colors:
            if time.time() > deadline - 5:
                break
            if color_freq[target] == 0:
                continue
            for donor in sorted_colors:
                if donor == target or color_freq[donor] == 0:
                    continue
                new_c = try_merge(best_colors, donor, target)
                if new_c is not None:
                    best_colors = new_c
                    best_count = count_distinct(best_colors)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break

    # Final validity check
    if not is_valid(best_colors):
        # Fallback
        best_colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)

    return best_colors.tolist()
