import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_c6_cycles(n):
        """Build list of all C6 cycles as frozensets of edge indices."""
        cycles = []
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = set()
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.add(edge_index(u, v, n))
                cycles.append(cycle_edges)
        return cycles

    def build_c4_cycles(n):
        """Build list of all C4 cycles as frozensets of edge indices."""
        cycles = []
        verts = list(range(n))
        for combo in itertools.combinations(verts, 4):
            a, b, c, d = combo
            # Three ways to form a C4 from 4 vertices:
            # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
            for (v1, v2, v3, v4) in [(a, b, c, d), (a, b, d, c), (a, c, b, d)]:
                edges = {
                    edge_index(v1, v2, n),
                    edge_index(v2, v3, n),
                    edge_index(v3, v4, n),
                    edge_index(v4, v1, n)
                }
                cycles.append(edges)
        return cycles

    def count_mono_cycles(colors, cycles):
        """Count number of monochromatic cycles."""
        count = 0
        for ce in cycles:
            c0 = colors[list(ce)[0]]
            if all(colors[e] == c0 for e in ce):
                count += 1
        return count

    def build_edge_to_cycles(cycles, m):
        """Build mapping from edge index to list of cycle indices."""
        edge_to_cycles = [[] for _ in range(m)]
        for ci, ce in enumerate(cycles):
            for e in ce:
                edge_to_cycles[e].append(ci)
        return edge_to_cycles

    def spectral_init(n, seed_offset=0):
        """Generate initial coloring using spectral embedding."""
        rng = np.random.default_rng(42 + seed_offset)

        deg = min(4, n - 1)
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            candidates = [j for j in range(n) if j != i]
            neighbors = rng.choice(candidates, size=deg, replace=False)
            for j in neighbors:
                A[i, j] = 1.0

        D = np.diag(A.sum(axis=1))
        L = D - A

        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)[1:5]
            V = eigvecs[:, idx]
        except Exception:
            return [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]

        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                vi = V[i, :]
                vj = V[j, :]
                diff = vi - vj
                signs = np.sign(diff)
                s1 = 1 if signs[0] >= 0 else 0
                s2 = 1 if signs[1] >= 0 else 0
                color = s1 * 2 + s2
                colors.append(int(color))

        return colors

    def random_init(n, seed_offset=0):
        random.seed(42 + seed_offset)
        return [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]

    def vectorized_local_search(n, time_budget, seed_offset=0, use_spectral=False):
        """
        Vectorized local search minimizing total C4 + C6 monochromatic cycles.
        Uses the insight that C4-free bipartite graphs have girth >= 6,
        so minimizing C4 cycles helps ensure no C6.
        """
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)

        # Build cycle structures
        c4_cycles = build_c4_cycles(n)
        c6_cycles = build_c6_cycles(n)

        # Combine with weights: C6 is the hard constraint, C4 is a proxy
        # Weight C6 heavily, C4 lightly
        all_cycles = [(ce, 10) for ce in c6_cycles] + [(ce, 1) for ce in c4_cycles]
        num_cycles = len(all_cycles)

        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        edge_to_cycles = build_edge_to_cycles([ce for ce, w in all_cycles], m)

        # Initialize
        if use_spectral:
            colors = spectral_init(n, seed_offset)
        else:
            colors = random_init(n, seed_offset)

        # Compute initial objective: weighted sum of monochromatic cycles
        def compute_objective(colors):
            obj = 0
            for ci, (ce, w) in enumerate(all_cycles):
                c0 = colors[list(ce)[0]]
                if all(colors[e] == c0 for e in ce):
                    obj += w
            return obj

        objective = compute_objective(colors)
        c6_count = count_mono_cycles(colors, c6_cycles)

        if c6_count == 0:
            return {'n': n, 'colors': colors}

        best_colors = colors[:]
        best_objective = objective
        best_c6 = c6_count

        T_start = 2.0
        T_end = 0.005
        start_iter_time = time.time()
        max_iter = 500000

        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break

            e = random.randint(0, m - 1)
            c_old = colors[e]
            c_new = random.randint(0, 3)
            while c_new == c_old:
                c_new = random.randint(0, 3)

            # Compute delta in objective
            delta_obj = 0
            delta_c6 = 0
            for ci in edge_to_cycles[e]:
                ce, w = all_cycles[ci]
                is_c6 = w == 10
                cnt_old = 0
                cnt_new = 0
                for ed in ce:
                    if ed == e:
                        continue
                    if colors[ed] == c_old:
                        cnt_old += 1
                    elif colors[ed] == c_new:
                        cnt_new += 1

                size = len(ce)
                before_mono = 1 if cnt_old == size - 1 else 0
                after_mono = 1 if cnt_new == size - 1 else 0
                delta_obj += w * (after_mono - before_mono)
                if is_c6:
                    delta_c6 += (after_mono - before_mono)

            new_objective = objective + delta_obj
            new_c6 = c6_count + delta_c6

            if delta_obj <= 0:
                accept = True
            else:
                progress = i / max_iter
                T = T_start * (T_end / T_start) ** progress
                if delta_obj / T > 80:
                    accept = False
                else:
                    prob = math.exp(-delta_obj / T)
                    accept = random.random() < prob

            if accept:
                colors[e] = c_new
                objective = new_objective
                c6_count = new_c6

                if c6_count < best_c6:
                    best_c6 = c6_count
                    best_objective = objective
                    best_colors = colors[:]
                    if best_c6 == 0:
                        return {'n': n, 'colors': best_colors}
                elif best_c6 == 0 and objective < best_objective:
                    best_objective = objective
                    best_colors = colors[:]

        if best_c6 == 0:
            return {'n': n, 'colors': best_colors}
        return None

    def search_n(n, time_budget, seed_offset=0):
        """Try multiple initialization strategies and return best result."""
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        c6_cycles = build_c6_cycles(n)

        if len(c6_cycles) == 0:
            return {'n': n, 'colors': [0] * m}

        # Try different initialization strategies
        strategies = [
            ('spectral', True),
            ('random', False),
            ('spectral2', True),
        ]

        best = None
        best_c6 = float('inf')

        for si, (name, use_spec) in enumerate(strategies):
            if time.time() > deadline - 5:
                break
            tb = time_budget / len(strategies)
            result = vectorized_local_search(n, tb, seed_offset=seed_offset + si * 1000, use_spectral=use_spec)
            if result:
                # Verify C6 count
                c6_count = count_mono_cycles(result['colors'], c6_cycles)
                if c6_count == 0:
                    return result
                elif c6_count < best_c6:
                    best_c6 = c6_count
                    best = result

        return best

    # Try from largest n down
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    time_alloc = {17: 28, 16: 22, 15: 16, 14: 12, 13: 10, 12: 8, 11: 7, 10: 5, 9: 3, 8: 3, 7: 2, 6: 2}

    for n in targets:
        if time.time() > deadline - 8:
            break

        tb = time_alloc.get(n, 5)
        result = search_n(n, tb, seed_offset=n * 100)
        if result:
            best_result = result
            return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
