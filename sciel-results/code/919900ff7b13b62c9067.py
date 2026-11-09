import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 142.0

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve(n, d, seed, budget):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build conflict adjacency: edges i,j conflict if share vertex or joined by an edge
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict sets as bitsets (use int for speed)
        conflict = [0] * m
        for i in range(m):
            ui, vi = edges[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    mask |= (1 << j)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        mask |= (1 << j)
            conflict[i] = mask

        # Precompute for fast induced-matching check of a subset represented by a mask
        # We'll check a colour class (list of edge indices) is an induced matching by:
        #   for each pair in class, they must not conflict.
        # Since class size is small, pairwise check is fine.

        def is_induced_matching(indices):
            for a in range(len(indices)):
                ia = indices[a]
                for b in range(a + 1, len(indices)):
                    ib = indices[b]
                    if (conflict[ia] >> ib) & 1:
                        return False
            return True

        # Initial greedy: repeatedly extract a maximal induced matching (greedy by random order)
        def greedy_initial():
            remaining = set(range(m))
            classes = []
            while remaining:
                order = list(remaining)
                random.shuffle(order)
                chosen = []
                for i in order:
                    if i not in remaining:
                        continue
                    # Check if i can be added to chosen
                    ok = True
                    for j in chosen:
                        if (conflict[i] >> j) & 1:
                            ok = False
                            break
                    if ok:
                        chosen.append(i)
                        remaining.remove(i)
                if not chosen:
                    # Should not happen, but safety
                    chosen = [list(remaining)[0]]
                    remaining.remove(chosen[0])
                classes.append(chosen)
            return classes

        # Simulated annealing over colour assignments
        # State: color_of_edge (list of ints), num_classes = max+1
        # Move: recolor one edge to a new color (possibly new class)
        # Objective: minimize number of distinct colors used

        best_classes = None
        best_num = float('inf')

        # Try multiple greedy restarts
        rng = random.Random(seed)
        for restart in range(12):
            random.seed(seed + restart * 1000)
            classes = greedy_initial()
            num = len(classes)
            if num < best_num:
                best_num = num
                best_classes = [c[:] for c in classes]
            if time.time() > deadline - 10.0:
                break

        if best_classes is None:
            # Fallback: trivial coloring
            classes = [[i] for i in range(m)]
            best_classes = classes
            best_num = m

        # Simulated annealing refinement
        if best_num > 1 and time.time() < deadline - 5.0:
            # Convert classes to color array
            color_of = [-1] * m
            for c, cls in enumerate(best_classes):
                for i in cls:
                    color_of[i] = c
            cur_num = best_num

            T = 0.5
            T_min = 0.01
            cooling = 0.9995
            max_iter = 50000

            for it in range(max_iter):
                if time.time() > deadline - 5.0:
                    break
                if T < T_min:
                    break

                # Pick a random edge
                i = rng.randrange(m)
                old_c = color_of[i]

                # Try to recolor to a random existing color or a new one
                existing_colors = list(range(cur_num))
                if rng.random() < 0.3 and cur_num > 1:
                    # Try to eliminate a color: pick a color to remove, recolor all its edges
                    target_color = rng.choice(existing_colors)
                    if target_color == old_c:
                        continue
                    # Check if i can take target_color
                    if (conflict[i] >> i) & 1:  # self, skip
                        continue
                    # Check conflicts with edges of target_color
                    ok = True
                    for j in range(m):
                        if color_of[j] == target_color and j != i:
                            if (conflict[i] >> j) & 1:
                                ok = False
                                break
                    if not ok:
                        continue
                    new_c = target_color
                else:
                    # Random recolor to existing or new
                    if rng.random() < 0.7:
                        new_c = rng.choice(existing_colors)
                    else:
                        new_c = cur_num  # new color
                    # Check validity
                    ok = True
                    for j in range(m):
                        if color_of[j] == new_c and j != i:
                            if (conflict[i] >> j) & 1:
                                ok = False
                                break
                    if not ok:
                        continue

                # Apply move
                color_of[i] = new_c

                # Compute new number of colors
                used = set(color_of)
                new_num = len(used)

                delta = new_num - cur_num
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    cur_num = new_num
                    if cur_num < best_num:
                        best_num = cur_num
                        # Rebuild classes
                        best_classes = [[] for _ in range(best_num)]
                        for i in range(m):
                            best_classes[color_of[i]].append(i)
                else:
                    color_of[i] = old_c

                T *= cooling

            # Final rebuild
            used = set(color_of)
            final_num = len(used)
            if final_num < best_num:
                best_num = final_num
            final_classes = [[] for _ in range(final_num)]
            for i in range(m):
                final_classes[color_of[i]].append(i)
            if final_num < best_num:
                best_classes = final_classes
                best_num = final_num

        # Build final color array
        final_color = [-1] * m
        for c, cls in enumerate(best_classes):
            for i in cls:
                final_color[i] = c
        # Renumber colors to be contiguous 0..K-1
        mapping = {}
        next_c = 0
        for i in range(m):
            c = final_color[i]
            if c not in mapping:
                mapping[c] = next_c
                next_c += 1
            final_color[i] = mapping[c]

        return final_color

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining < 5.0:
            # Fallback trivial
            edges = instance(n, d, seed)
            results.append([i for i in range(len(edges))])
            continue
        budget = min(remaining - 2.0, 45.0)
        if len(results) == 2:
            budget = remaining - 2.0
        res = solve(n, d, seed, budget)
        results.append(res)

    return results
