import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Precompute adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict sets
        conflict_sets = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
                else:
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)

        # Greedy DSATUR initial coloring
        def greedy_dsatur():
            colors = [-1] * m
            uncolored = set(range(m))
            while uncolored:
                best_i, best_score = None, (-1, -1)
                for i in uncolored:
                    neighbor_colors = set()
                    for j in conflict_sets[i]:
                        if colors[j] != -1:
                            neighbor_colors.add(colors[j])
                    sat = len(neighbor_colors)
                    deg = sum(1 for j in conflict_sets[i] if j in uncolored)
                    score = (sat, deg)
                    if score > best_score:
                        best_score = score
                        best_i = i
                i = best_i
                taken = set()
                for j in conflict_sets[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.remove(i)
            return colors

        init_colors = greedy_dsatur()
        best_colors = init_colors[:]
        best_num_colors = max(init_colors) + 1 if init_colors else 0

        # Simulated Annealing with proper incremental updates
        rng = random.Random(seed + 999)
        current = init_colors[:]
        max_c = max(current) if current else 0

        # Maintain color_counts and per-color conflict counts
        color_counts = defaultdict(int)
        for c in current:
            color_counts[c] += 1

        def count_conflicts_in_color(c):
            es = [i for i in range(m) if current[i] == c]
            cnt = 0
            for idx in range(len(es)):
                for jdx in range(idx + 1, len(es)):
                    if es[jdx] in conflict_sets[es[idx]]:
                        cnt += 1
            return cnt

        conflict_counts = {}
        for c in color_counts:
            conflict_counts[c] = count_conflicts_in_color(c)
        total_conf = sum(conflict_counts.values())

        def energy():
            return total_conf * 1000 + max_c + 1

        T = 50.0
        cooling = 0.9995
        best_energy = energy()
        best_colors = current[:]
        best_max = max_c

        steps = 0
        max_steps = 3000000
        while time.time() < time_limit and steps < max_steps:
            steps += 1
            i = rng.randrange(m)
            old_c = current[i]
            new_c = rng.randint(0, max_c + 1)
            if new_c == old_c:
                continue

            # Delta conflicts
            rem = sum(1 for j in conflict_sets[i] if current[j] == old_c)
            add = sum(1 for j in conflict_sets[i] if current[j] == new_c)
            delta_conf = add - rem

            # Update color counts
            color_counts[old_c] -= 1
            color_counts[new_c] += 1

            # New max color
            new_max = max_c
            if color_counts[old_c] == 0 and old_c == max_c:
                new_max = -1
                for cc, cnt in color_counts.items():
                    if cnt > 0 and cc > new_max:
                        new_max = cc
            if new_c > new_max:
                new_max = new_c

            new_total = total_conf + delta_conf
            new_energy = new_total * 1000 + new_max + 1
            delta_e = new_energy - best_energy

            if delta_e <= 0 or rng.random() < pow(2.718281828, -delta_e / T):
                current[i] = new_c
                total_conf = new_total
                max_c = new_max
                # Update conflict counts
                conflict_counts[old_c] -= rem
                conflict_counts[new_c] = conflict_counts.get(new_c, 0) + add
                if color_counts[old_c] == 0:
                    del conflict_counts[old_c]
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_colors = current[:]
                    best_max = new_max
            else:
                color_counts[old_c] += 1
                color_counts[new_c] -= 1

            T *= cooling

        # Verify and return
        if total_conf == 0:
            return best_colors
        return init_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colors = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colors.items() if set(f) & near}
                c = 0
                while c in taken:
                    c += 1
                colors[e] = c
            results.append([colors[e] for e in edges])
            continue
        time_limit = max(1.0, remaining - 1.0)
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)
    return results
