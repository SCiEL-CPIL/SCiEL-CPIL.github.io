import time
import math
import numpy as np
from random import Random

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # Build conflict graph on edges (two edges conflict if they share a vertex
        # or are joined by an edge in the original graph)
        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                # Share a vertex?
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    # Joined by an edge?
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)

        # Build adjacency matrix for vectorized operations
        adj_mat = np.zeros((m, m), dtype=np.int8)
        for i in range(m):
            for j in neigh[i]:
                adj_mat[i, j] = 1

        deg_arr = np.array([len(x) for x in neigh], dtype=np.int32)
        max_deg = int(deg_arr.max()) if m > 0 else 0

        # DSATUR initial coloring
        def dsatur():
            color = [-1] * m
            color_sets = [set() for _ in range(m)]
            uncolored = set(range(m))
            while uncolored:
                best = max(uncolored, key=lambda v: (len(color_sets[v]), deg_arr[v]))
                forbidden = set()
                for u in neigh[best]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                c = 0
                while c in forbidden:
                    c += 1
                color[best] = c
                for u in neigh[best]:
                    if color[u] != -1:
                        color_sets[u].add(c)
                uncolored.discard(best)
            return color

        color0 = dsatur()
        K0 = max(color0) + 1

        # Vectorized conflict count: number of pairs (i,j) with i<j, adj_mat[i,j]=1, color[i]==color[j]
        def count_conflicts_vec(color_arr):
            # For each pair of adjacent vertices, check if they have the same color
            # Using broadcasting: for each row i, count neighbors j>i with same color
            # More efficient: use the adjacency matrix and color equality
            # conflict[i,j] = adj_mat[i,j] * (color[i] == color[j]) for i<j
            # Sum over all i<j
            eq = (color_arr[:, None] == color_arr[None, :])
            # Only count upper triangle to avoid double counting
            mask = np.triu(np.ones((m, m), dtype=bool), k=1)
            conflicts = np.sum(adj_mat * eq * mask)
            return int(conflicts)

        # Delta conflict when changing vertex v from old_c to new_c
        def delta_conflicts(color_arr, v, old_c, new_c):
            # Neighbors of v that have old_c: removing v's old_c removes conflicts with them
            # Neighbors of v that have new_c: adding v's new_c adds conflicts with them
            nb = adj_mat[v]
            old_count = np.sum(nb * (color_arr == old_c))
            new_count = np.sum(nb * (color_arr == new_c))
            return new_count - old_count

        # Simulated annealing for a given K
        def anneal(color_arr, K, rng, time_limit):
            color_arr = color_arr.copy()
            current_conflicts = count_conflicts_vec(color_arr)
            best_color = color_arr.copy()
            best_conflicts = current_conflicts

            T0 = max(1.0, current_conflicts * 0.5)
            T_end = 0.01
            # Estimate steps we can do
            start_time = time.time()
            step_time = 0.0001  # rough estimate per step

            while time.time() < time_limit:
                # Compute temperature based on elapsed fraction
                elapsed = time.time() - start_time
                # Use exponential cooling
                alpha = 0.9995
                T = max(T_end, T0 * (alpha ** int(elapsed / 0.001)))

                if current_conflicts == 0:
                    break

                # Pick a random vertex
                v = rng.randrange(m)
                old_c = color_arr[v]

                # Pick a random new color
                new_c = rng.randrange(K)
                if new_c == old_c:
                    continue

                delta = delta_conflicts(color_arr, v, old_c, new_c)

                # Metropolis acceptance
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    color_arr[v] = new_c
                    current_conflicts += delta

                    if current_conflicts < best_conflicts:
                        best_conflicts = current_conflicts
                        best_color = color_arr.copy()

            return best_color, best_conflicts

        # Main search: start with DSATUR, then use simulated annealing to reduce colors
        best_color = np.array(color0, dtype=np.int32)
        best_K = K0
        rng = Random(seed * 7919 + 42)

        # Try to reduce K from best_K down to a lower bound
        lower_bound = max(2, int(math.ceil(math.sqrt(m))))  # rough lower bound

        for target_K in range(best_K, lower_bound - 1, -1):
            if time.time() > deadline - 2.0:
                break

            # Multiple restarts with simulated annealing
            found = False
            max_restarts = 20
            for attempt in range(max_restarts):
                if time.time() > deadline - 1.0:
                    break

                if attempt == 0:
                    # Start from current best, clamp to target_K colors
                    trial = np.minimum(best_color, target_K - 1)
                elif attempt < 5:
                    # Perturb from best
                    trial = np.minimum(best_color, target_K - 1)
                    n_perturb = max(1, m // 10)
                    idx_perturb = rng.sample(range(m), n_perturb)
                    for i in idx_perturb:
                        trial[i] = rng.randrange(target_K)
                else:
                    # Random start
                    trial = np.array([rng.randrange(target_K) for _ in range(m)], dtype=np.int32)

                # Run simulated annealing
                remaining = deadline - 1.0 - time.time()
                if remaining < 0.5:
                    break
                time_slice = min(remaining / (max_restarts - attempt) * 0.8, 5.0)
                result, result_conflicts = anneal(trial, target_K, rng, time.time() + time_slice)

                if result_conflicts == 0:
                    best_color = result
                    best_K = target_K
                    found = True
                    break

            if not found:
                continue

        return compress(best_color.tolist())

    try:
        start = time.time()
        total_budget = 140.0  # leave 10s margin for 150s kill
        per_instance = total_budget / len(specs)

        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = start + total_budget - time.time()
            share = min(per_instance, remaining / (len(specs) - idx))
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))

        if time.time() > start + 145.0:
            return out

        return out
    except Exception:
        # Fallback: simple greedy
        out = []
        for spec in specs:
            n, d, seed = spec
            edges = instance(n, d, seed)
            if not edges:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj.get(x, set())}
                taken = set()
                for f, c in colour.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                colour[e] = c
            out.append([colour[e] for e in edges])
        return out
