import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        """Compute edge index for (i,j) in lexicographic order for K_n."""
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_cycles(n):
        """Build list of all C6 cycles as lists of edge indices."""
        cycles = []
        verts = list(range(n))
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_index(u, v, n))
                cycles.append(cycle_edges)
        return cycles

    def total_c6(colors, n, cycles):
        """Count all monochromatic C6s."""
        total = 0
        for ce in cycles:
            c0 = colors[ce[0]]
            if all(colors[e] == c0 for e in ce):
                total += 1
        return total

    def search_n(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)

        # Build cycles
        cycles = build_cycles(n)
        num_cycles = len(cycles)

        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}

        # Build edge-to-cycles mapping
        edge_to_cycles = [[] for _ in range(m)]
        for ci, ce in enumerate(cycles):
            for e in ce:
                edge_to_cycles[e].append(ci)

        # Initial coloring: random
        colors = [random.randint(0, 3) for _ in range(m)]
        bad_count = total_c6(colors, n, cycles)

        if bad_count == 0:
            return {'n': n, 'colors': colors}

        best_colors = colors[:]
        best_bad = bad_count

        # Local search with simulated annealing
        T_start = 1.0
        T_end = 0.01
        max_iter = 500000
        start_iter_time = time.time()

        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break

            # Pick random edge
            e = random.randint(0, m - 1)
            c_old = colors[e]

            # Pick new color (uniform from other 3)
            c_new = random.randint(0, 3)
            while c_new == c_old:
                c_new = random.randint(0, 3)

            # Calculate delta efficiently
            delta = 0
            for ci in edge_to_cycles[e]:
                ce = cycles[ci]
                cnt_old = 0
                cnt_new = 0
                for ed in ce:
                    if ed == e:
                        continue
                    if colors[ed] == c_old:
                        cnt_old += 1
                    elif colors[ed] == c_new:
                        cnt_new += 1

                before_mono = 1 if cnt_old == 5 else 0
                after_mono = 1 if cnt_new == 5 else 0

                delta += (after_mono - before_mono)

            # Acceptance
            if delta <= 0:
                accept = True
            else:
                progress = i / max_iter
                T = T_start * (T_end / T_start) ** progress
                if delta / T > 50:
                    accept = False
                else:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob

            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors[:]
                    if best_bad == 0:
                        return {'n': n, 'colors': best_colors}

        # Return best found if it's valid
        if best_bad == 0:
            return {'n': n, 'colors': best_colors}
        return None

    # Try targets from high to low
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]

    # Allocate time: prioritize higher n
    time_alloc = {17: 30, 16: 25, 15: 20, 14: 15, 13: 10, 12: 8, 11: 7, 10: 5, 9: 3, 8: 3, 7: 2, 6: 2}

    for n in targets:
        if time.time() > deadline - 8:
            break

        tb = time_alloc.get(n, 5)
        # Try more restarts for harder n
        num_restarts = 4 if n >= 15 else 2
        restart_budget = tb / num_restarts

        for r in range(num_restarts):
            if time.time() > deadline - 8:
                break
            result = search_n(n, restart_budget, seed_offset=r * 1000 + n)
            if result:
                best_result = result
                # Found valid coloring for n, return immediately since we go high to low
                return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
