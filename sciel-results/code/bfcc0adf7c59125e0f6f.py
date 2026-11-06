import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s for safety

    def try_solve(n):
        if n < 6:
            return [0] * (n*(n-1)//2)

        vertices = list(range(n))
        total_edges = n*(n-1)//2

        # Build edge index map
        edge_idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_idx_map[(i, j)] = idx
                idx += 1

        # Pre-compute all C6s as lists of 6 edge indices
        # For each 6-vertex subset, there are 5 distinct Hamiltonian cycles
        c6_edge_indices = []
        for comb in combinations(vertices, 6):
            v0, v1, v2, v3, v4, v5 = comb
            # The 5 distinct C6 patterns (up to rotation/reflection)
            patterns = [
                [(v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)],
                [(v0,v1), (v1,v2), (v2,v4), (v4,v3), (v3,v5), (v5,v0)],
                [(v0,v1), (v1,v3), (v3,v2), (v2,v4), (v4,v5), (v5,v0)],
                [(v0,v1), (v1,v3), (v3,v4), (v4,v2), (v2,v5), (v5,v0)],
                [(v0,v1), (v1,v4), (v4,v2), (v2,v3), (v3,v5), (v5,v0)]
            ]
            for pat in patterns:
                e_indices = []
                for (u, v) in pat:
                    key = (u, v) if u < v else (v, u)
                    e_indices.append(edge_idx_map[key])
                c6_edge_indices.append(e_indices)

        num_c6s = len(c6_edge_indices)

        # Map each edge to the list of C6 indices it participates in
        edge_to_c6s = [[] for _ in range(total_edges)]
        for i, e_indices in enumerate(c6_edge_indices):
            for e in e_indices:
                edge_to_c6s[e].append(i)

        # Initialize with a structured random coloring
        colors = [random.randint(0, 3) for _ in range(total_edges)]

        # Calculate initial total_bad
        def compute_total_bad(colors):
            total_bad = 0
            for e_indices in c6_edge_indices:
                c0 = colors[e_indices[0]]
                is_mono = True
                for e in e_indices[1:]:
                    if colors[e] != c0:
                        is_mono = False
                        break
                if is_mono:
                    total_bad += 1
            return total_bad

        total_bad = compute_total_bad(colors)

        if total_bad == 0:
            return colors

        # Local search with simulated annealing-like acceptance
        import math
        T = 1.0
        cooling = 0.9995

        while time.time() < deadline:
            if total_bad == 0:
                return colors

            # Pick a random edge
            e_idx = random.randint(0, total_edges - 1)
            old_color = colors[e_idx]

            # Try all 3 other colors, pick best delta
            best_delta = 0
            best_new_color = -1

            for new_color in range(4):
                if new_color == old_color:
                    continue

                delta = 0
                for c6_idx in edge_to_c6s[e_idx]:
                    e_indices = c6_edge_indices[c6_idx]

                    # Check if C6 becomes monochromatic new_color
                    # We assume e_idx is new_color. Check other 5.
                    is_mono_new = True
                    for e in e_indices:
                        if e == e_idx:
                            continue
                        if colors[e] != new_color:
                            is_mono_new = False
                            break
                    if is_mono_new:
                        delta += 1

                    # Check if C6 stops being monochromatic old_color
                    is_mono_old = True
                    for e in e_indices:
                        if e == e_idx:
                            continue
                        if colors[e] != old_color:
                            is_mono_old = False
                            break
                    if is_mono_old:
                        delta -= 1

                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color

            # Acceptance criterion
            if best_new_color != -1 and (best_delta < 0 or random.random() < math.exp(-best_delta / T)):
                colors[e_idx] = best_new_color
                total_bad += best_delta
            else:
                # No improvement or not accepted, try a random move
                new_color = random.choice([c for c in range(4) if c != old_color])
                delta = 0
                for c6_idx in edge_to_c6s[e_idx]:
                    e_indices = c6_edge_indices[c6_idx]
                    is_mono_new = True
                    for e in e_indices:
                        if e == e_idx:
                            continue
                        if colors[e] != new_color:
                            is_mono_new = False
                            break
                    if is_mono_new:
                        delta += 1
                    is_mono_old = True
                    for e in e_indices:
                        if e == e_idx:
                            continue
                        if colors[e] != old_color:
                            is_mono_old = False
                            break
                    if is_mono_old:
                        delta -= 1
                colors[e_idx] = new_color
                total_bad += delta

            T *= cooling

            if total_bad == 0:
                return colors

        return None

    # Try n=17
    solution = try_solve(17)
    if solution:
        return {'n': 17, 'colors': solution}

    # If 17 fails, try 16
    solution = try_solve(16)
    if solution:
        return {'n': 16, 'colors': solution}

    # If 16 fails, try 15
    solution = try_solve(15)
    if solution:
        return {'n': 15, 'colors': solution}

    # Fallback to a small valid coloring
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
