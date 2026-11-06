import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_structure(n):
        m = n * (n - 1) // 2
        edge_lookup = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_lookup[i][j] = idx
                edge_lookup[j][i] = idx
                idx += 1

        cycles_list = []
        verts = list(range(n))

        # Generate all C6 cycles (unlabeled, canonical form)
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
                    cycle_edges.append(edge_lookup[u][v])
                cycles_list.append(cycle_edges)

        N = len(cycles_list)
        if N == 0:
            return [], [], 0

        # Map edge to list of cycle indices
        edge_to_cycles = [[] for _ in range(m)]
        for i in range(N):
            for e in cycles_list[i]:
                edge_to_cycles[e].append(i)

        return cycles_list, edge_to_cycles, N

    def spectral_init(n, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        m = n * (n - 1) // 2
        # Structured initialization: c(i,j) = (i - j) % 4
        colors = np.zeros(m, dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                colors[idx] = (i - j) % 4
                idx += 1
        return colors

    def random_init(n, seed=42):
        np.random.seed(seed)
        m = n * (n - 1) // 2
        return np.random.randint(0, 4, m, dtype=np.int8)

    def search_n(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)

        cycles_list, edge_to_cycles, N = build_structure(n)

        if N == 0:
            colors = np.zeros(m, dtype=np.int8)
            return {'n': n, 'colors': colors.tolist()}

        # Try multiple initializations
        best_result = None
        inits = [
            ('spectral', spectral_init(n, seed=seed_offset)),
            ('random1', random_init(n, seed=seed_offset + 100)),
            ('random2', random_init(n, seed=seed_offset + 200)),
        ]

        for init_name, colors in inits:
            if time.time() > deadline - 5:
                break

            # Initialize counts
            cnt = np.zeros((N, 4), dtype=np.int32)
            is_bad = np.zeros(N, dtype=bool)
            bad_count = 0

            for c_idx in range(N):
                ce = cycles_list[c_idx]
                for c in range(4):
                    cnt[c_idx, c] = np.sum(colors[ce] == c)
                if np.any(cnt[c_idx] == 6):
                    is_bad[c_idx] = True
                    bad_count += 1

            if bad_count == 0:
                return {'n': n, 'colors': colors.tolist()}

            best_colors = colors.copy()
            best_bad = bad_count

            # Simulated Annealing with restarts
            T_start = 2.0
            T_end = 0.01
            start_iter_time = time.time()
            max_iters = int(time_budget * 5000)  # Estimate iterations

            iter_count = 0
            no_improve_count = 0
            max_no_improve = 5000

            while time.time() < start_iter_time + time_budget:
                if time.time() > deadline - 5:
                    break

                e = np.random.randint(0, m)
                c_old = colors[e]

                # Pick new color
                c_new = np.random.randint(0, 4)
                while c_new == c_old:
                    c_new = np.random.randint(0, 4)

                cycs = edge_to_cycles[e]

                # Calculate Delta
                delta = 0
                old_states = []

                for c_idx in cycs:
                    was_bad = is_bad[c_idx]

                    old_cnt_old = cnt[c_idx, c_old]
                    old_cnt_new = cnt[c_idx, c_new]

                    cnt[c_idx, c_old] -= 1
                    cnt[c_idx, c_new] += 1

                    is_bad_now = False
                    if cnt[c_idx, c_old] == 6:
                        is_bad_now = True
                    elif cnt[c_idx, c_new] == 6:
                        is_bad_now = True

                    if was_bad and not is_bad_now:
                        delta -= 1
                    elif not was_bad and is_bad_now:
                        delta += 1

                    is_bad[c_idx] = is_bad_now
                    old_states.append((c_idx, was_bad, old_cnt_old, old_cnt_new))

                # Acceptance
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    elapsed = time.time() - start_iter_time
                    progress = min(1.0, elapsed / time_budget)
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
                        best_colors = colors.copy()
                        no_improve_count = 0
                        if best_bad == 0:
                            return {'n': n, 'colors': colors.tolist()}
                    else:
                        no_improve_count += 1
                else:
                    for c_idx, was_bad, old_cnt_old, old_cnt_new in old_states:
                        is_bad[c_idx] = was_bad
                        cnt[c_idx, c_old] = old_cnt_old
                        cnt[c_idx, c_new] = old_cnt_new
                    no_improve_count += 1

                iter_count += 1

                # Restart if stuck
                if no_improve_count > max_no_improve:
                    # Perturbation: flip random edges
                    for _ in range(10):
                        e2 = np.random.randint(0, m)
                        c_old2 = colors[e2]
                        c_new2 = np.random.randint(0, 4)
                        while c_new2 == c_old2:
                            c_new2 = np.random.randint(0, 4)

                        cycs2 = edge_to_cycles[e2]
                        for c_idx in cycs2:
                            cnt[c_idx, c_old2] -= 1
                            cnt[c_idx, c_new2] += 1
                            is_bad[c_idx] = (cnt[c_idx, c_old2] == 6) or (cnt[c_idx, c_new2] == 6)
                        colors[e2] = c_new2

                    # Recount
                    bad_count = np.sum(is_bad)
                    no_improve_count = 0

            if best_bad == 0:
                return {'n': n, 'colors': best_colors.tolist()}
            elif best_result is None or best_bad < best_result[0]:
                best_result = (best_bad, best_colors)

        if best_result is not None and best_result[0] == 0:
            return {'n': n, 'colors': best_result[1].tolist()}
        return None

    # Try targets in order
    targets = [17, 16, 15]

    for n in targets:
        if time.time() > deadline - 10:
            break
        tb = 40 if n == 17 else 20
        result = search_n(n, tb, seed_offset=n)
        if result:
            return result

    # Fallback: return a valid small coloring
    n = 5
    m = n * (n - 1) // 2
    colors = [0] * m
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
