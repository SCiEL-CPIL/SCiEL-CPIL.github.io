import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    num_edges = len(edges)

    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
    num_k4s = len(k4s)

    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)

    def compact(colors):
        unique = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(unique)}
        return [mapping[c] for c in colors], len(unique)

    # Try simple algebraic constructions first
    best_colors = None
    best_count = float('inf')

    # (i+j) % p
    for p in range(13, 22):
        colors = [(i + j) % p for (i, j) in edges]
        valid = True
        for k4 in k4s:
            if len(set(colors[ei] for ei in k4)) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break

    # (i*j) % p
    for p in [13, 17, 19, 23, 29, 31]:
        colors = [(i * j) % p for (i, j) in edges]
        valid = True
        for k4 in k4s:
            if len(set(colors[ei] for ei in k4)) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break

    # (i^2 + j) % p
    for p in [13, 15, 17, 19]:
        colors = [(i * i + j) % p for (i, j) in edges]
        valid = True
        for k4 in k4s:
            if len(set(colors[ei] for ei in k4)) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break

    # (i + 2*j) % p
    for p in [13, 15, 17, 19]:
        colors = [(i + 2 * j) % p for (i, j) in edges]
        valid = True
        for k4 in k4s:
            if len(set(colors[ei] for ei in k4)) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break

    # (j - i) % p
    for p in [13, 15, 17, 19]:
        colors = [(j - i) % p for (i, j) in edges]
        valid = True
        for k4 in k4s:
            if len(set(colors[ei] for ei in k4)) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break

    if best_colors is not None and best_count <= 13:
        return compact(best_colors)[0]

    # Use local search with O(1) penalty updates via per-K4 color count tracking
    if best_colors is not None:
        colors = best_colors[:]
    else:
        colors = [(i + j) % 17 for (i, j) in edges]

    # Determine max color value for array sizing
    max_color_val = max(colors)
    max_k = max(17, max_color_val + 1)

    # For each K4, track color counts and distinct color count
    # k4_color_counts[k4_idx] is a list of [color, count] pairs (or dict)
    # We'll use a list of dicts for O(1) lookup
    k4_color_counts = []
    k4_distinct = []
    current_penalty = 0

    for k4_idx in range(num_k4s):
        e_list = k4s[k4_idx]
        counts = {}
        for ei in e_list:
            c = colors[ei]
            counts[c] = counts.get(c, 0) + 1
        k4_color_counts.append(counts)
        distinct = len(counts)
        k4_distinct.append(distinct)
        if distinct < 5:
            current_penalty += 1

    best_count = len(set(colors))
    best_solution = colors[:]

    T = 3.0
    cooling = 0.9997
    min_T = 0.005

    edge_k4s_list = edge_k4s

    # O(1) penalty delta: for each affected K4, check if distinct count crosses 5
    # When changing edge e from old_c to new_c in K4:
    #   - decrement count[old_c], if it hits 0 remove key -> distinct -1
    #   - increment count[new_c], if it was 0 add key -> distinct +1
    #   - net change in distinct is in {-1, 0, +1}
    #   - penalty changes only if distinct crosses 5

    iterations = 0
    while time.time() < deadline - 10.0 and T > min_T:
        e_idx = random.randint(0, num_edges - 1)
        old_c = colors[e_idx]
        new_c = random.randint(0, max_k - 1)
        if new_c == old_c:
            continue

        # Compute penalty delta for affected K4s in O(1) per K4
        delta_penalty = 0
        affected_k4s = edge_k4s_list[e_idx]
        for k4_idx in affected_k4s:
            counts = k4_color_counts[k4_idx]
            old_distinct = k4_distinct[k4_idx]
            
            # Simulate: remove old_c, add new_c
            old_count = counts.get(old_c, 0)
            new_count_before = counts.get(new_c, 0)
            
            new_distinct = old_distinct
            if old_count == 1:
                new_distinct -= 1
            if new_count_before == 0:
                new_distinct += 1
            
            if old_distinct < 5 and new_distinct >= 5:
                delta_penalty -= 1
            elif old_distinct >= 5 and new_distinct < 5:
                delta_penalty += 1

        if delta_penalty <= 0 or random.random() < math.exp(-delta_penalty / T):
            # Accept: update state
            colors[e_idx] = new_c
            current_penalty += delta_penalty
            iterations += 1

            for k4_idx in affected_k4s:
                counts = k4_color_counts[k4_idx]
                old_count = counts.get(old_c, 0)
                new_count_before = counts.get(new_c, 0)
                
                # Update counts
                if old_count == 1:
                    del counts[old_c]
                else:
                    counts[old_c] = old_count - 1
                if new_count_before == 0:
                    counts[new_c] = 1
                else:
                    counts[new_c] = new_count_before + 1
                
                k4_distinct[k4_idx] = len(counts)

            # Check if we improved the number of distinct colors used overall
            if iterations % 50 == 0:
                nc = len(set(colors))
                if nc < best_count:
                    best_count = nc
                    best_solution = colors[:]
                    if best_count < max_k:
                        max_k = best_count
        else:
            pass  # colors unchanged

        T *= cooling

    # Final polish: try to reduce colors used
    remaining_time = deadline - time.time()
    if remaining_time > 3.0 and best_count > 13:
        polish_colors = best_solution[:]
        polish_best_count = best_count

        # Rebuild K4 data structures for polish
        polish_k4_color_counts = []
        polish_k4_distinct = []
        for k4_idx in range(num_k4s):
            e_list = k4s[k4_idx]
            counts = {}
            for ei in e_list:
                c = polish_colors[ei]
                counts[c] = counts.get(c, 0) + 1
            polish_k4_color_counts.append(counts)
            polish_k4_distinct.append(len(counts))

        for _ in range(int(min(remaining_time * 500, 50000))):
            if time.time() > deadline - 1.5:
                break
            if len(set(polish_colors)) <= 13:
                break
            e_idx = random.randint(0, num_edges - 1)
            old_c = polish_colors[e_idx]
            unique_colors = set(polish_colors)
            other_colors = [c for c in unique_colors if c != old_c]
            if not other_colors:
                break
            new_c = random.choice(other_colors)

            valid = True
            for k4_idx in edge_k4s_list[e_idx]:
                counts = polish_k4_color_counts[k4_idx]
                old_distinct = polish_k4_distinct[k4_idx]
                old_count = counts.get(old_c, 0)
                new_count_before = counts.get(new_c, 0)
                
                new_distinct = old_distinct
                if old_count == 1:
                    new_distinct -= 1
                if new_count_before == 0:
                    new_distinct += 1
                
                if new_distinct < 5:
                    valid = False
                    break

            if valid:
                polish_colors[e_idx] = new_c
                for k4_idx in edge_k4s_list[e_idx]:
                    counts = polish_k4_color_counts[k4_idx]
                    old_count = counts.get(old_c, 0)
                    new_count_before = counts.get(new_c, 0)
                    
                    if old_count == 1:
                        del counts[old_c]
                    else:
                        counts[old_c] = old_count - 1
                    if new_count_before == 0:
                        counts[new_c] = 1
                    else:
                        counts[new_c] = new_count_before + 1
                    polish_k4_distinct[k4_idx] = len(counts)
                
                nc = len(set(polish_colors))
                if nc < polish_best_count:
                    polish_best_count = nc
                    best_solution = polish_colors[:]
                    best_count = nc

    return compact(best_solution)[0]
