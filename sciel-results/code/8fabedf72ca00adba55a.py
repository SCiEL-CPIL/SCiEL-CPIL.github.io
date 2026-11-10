import numpy as np
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def construct():
    start_time = time.time()
    DEADLINE = 84.0

    # Build edge index
    edge_idx = {}
    edge_list = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1

    # Precompute all 4-cycles (3 per 4-vertex subset)
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]

    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    e1 = (edge_idx[(a, b)], edge_idx[(b, c)], edge_idx[(c, d)], edge_idx[(d, a)])
                    cycles.append(e1)
                    for e in e1:
                        edge_to_cycles[e].append(len(cycles) - 1)
                    e2 = (edge_idx[(a, b)], edge_idx[(b, d)], edge_idx[(d, c)], edge_idx[(c, a)])
                    cycles.append(e2)
                    for e in e2:
                        edge_to_cycles[e].append(len(cycles) - 1)
                    e3 = (edge_idx[(a, c)], edge_idx[(c, b)], edge_idx[(b, d)], edge_idx[(d, a)])
                    cycles.append(e3)
                    for e in e3:
                        edge_to_cycles[e].append(len(cycles) - 1)

    NUM_CYCLES = len(cycles)
    cycles_arr = np.array(cycles, dtype=np.int32)

    def is_viol(c0, c1, c2, c3):
        return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3

    def get_violations(colors):
        v = 0
        for ci in range(NUM_CYCLES):
            cyc = cycles_arr[ci]
            if is_viol(int(colors[cyc[0]]), int(colors[cyc[1]]), int(colors[cyc[2]]), int(colors[cyc[3]])):
                v += 1
        return v

    def get_distinct(colors):
        return len(set(int(x) for x in colors))

    # Baseline: color edge (i,j) with i -> 15 distinct colors, valid
    baseline = np.array([i for i, j in edge_list], dtype=np.int32)

    best_score = get_distinct(baseline) * 1000 - get_violations(baseline)
    best_colors = baseline.copy()

    def anneal(initial, max_iters, t0, t1, deadline):
        colors = initial.copy()
        cur_v = get_violations(colors)
        cur_d = get_distinct(colors)
        cur_score = cur_d * 1000 - cur_v
        best_local = cur_score
        best_local_colors = colors.copy()

        color_freq = [0] * 120
        for c in colors:
            color_freq[int(c)] += 1

        for it in range(max_iters):
            if time.time() > deadline:
                break

            e = random.randint(0, NUM_EDGES - 1)
            old_c = int(colors[e])
            new_c = random.randint(0, 119)
            if new_c == old_c:
                continue

            delta_v = 0
            for ci in edge_to_cycles[e]:
                cyc = cycles_arr[ci]
                e0, e1, e2, e3 = int(cyc[0]), int(cyc[1]), int(cyc[2]), int(cyc[3])
                c0, c1, c2, c3 = int(colors[e0]), int(colors[e1]), int(colors[e2]), int(colors[e3])

                if e == e0:
                    nc0, nc1, nc2, nc3 = new_c, c1, c2, c3
                elif e == e1:
                    nc0, nc1, nc2, nc3 = c0, new_c, c2, c3
                elif e == e2:
                    nc0, nc1, nc2, nc3 = c0, c1, new_c, c3
                else:
                    nc0, nc1, nc2, nc3 = c0, c1, c2, new_c

                ov = is_viol(c0, c1, c2, c3)
                nv = is_viol(nc0, nc1, nc2, nc3)
                if ov and not nv:
                    delta_v -= 1
                elif not ov and nv:
                    delta_v += 1

            delta_d = 0
            if color_freq[old_c] == 1:
                delta_d -= 1
            if color_freq[new_c] == 0:
                delta_d += 1

            delta = delta_d * 1000 - delta_v
            T = t0 * (t1 / t0) ** (it / max(max_iters, 1))

            if delta > 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                colors[e] = new_c
                color_freq[old_c] -= 1
                color_freq[new_c] += 1
                cur_v += delta_v
                cur_d += delta_d
                cur_score += delta
                if cur_score > best_local:
                    best_local = cur_score
                    best_local_colors = colors.copy()

        return best_local, best_local_colors

    # Multi-start
    max_iters = 8000
    t0 = 40.0
    t1 = 0.2

    starts = []
    # Start 0: baseline
    starts.append(baseline.copy())
    # Start 1: baseline + perturb
    p = baseline.copy()
    for _ in range(5):
        e = random.randint(0, NUM_EDGES - 1)
        p[e] = random.randint(0, 119)
    starts.append(p)
    # Start 2: (i+j) mod 16
    starts.append(np.array([(i + j) % 16 for i, j in edge_list], dtype=np.int32))
    # Start 3: min(i,j) mod 8
    starts.append(np.array([min(i, j) % 8 for i, j in edge_list], dtype=np.int32))
    # Start 4: random
    starts.append(np.random.randint(0, 120, size=NUM_EDGES).astype(np.int32))
    # Start 5: (i*3+j) mod 16
    starts.append(np.array([(i * 3 + j) % 16 for i, j in edge_list], dtype=np.int32))
    # Start 6: (i+j) mod 32
    starts.append(np.array([(i + j) % 32 for i, j in edge_list], dtype=np.int32))
    # Start 7: (i*7+j*3) mod 24
    starts.append(np.array([(i * 7 + j * 3) % 24 for i, j in edge_list], dtype=np.int32))

    for s, init in enumerate(starts):
        if time.time() > DEADLINE:
            break
        ls, lc = anneal(init, max_iters, t0, t1, DEADLINE)
        if ls > best_score:
            best_score = ls
            best_colors = lc.copy()

    # Final validation and fix
    if best_colors is not None:
        v = get_violations(best_colors)
        if v == 0:
            return [int(x) for x in best_colors]
        else:
            colors = best_colors.copy()
            for _ in range(300):
                if time.time() > DEADLINE:
                    break
                v = get_violations(colors)
                if v == 0:
                    break
                for ci in range(NUM_CYCLES):
                    cyc = cycles_arr[ci]
                    c0 = int(colors[cyc[0]])
                    c1 = int(colors[cyc[1]])
                    c2 = int(colors[cyc[2]])
                    c3 = int(colors[cyc[3]])
                    if is_viol(c0, c1, c2, c3):
                        e = int(cyc[random.randint(0, 3)])
                        others = [int(cyc[k]) for k in range(4) if int(cyc[k]) != e]
                        colors[e] = colors[random.choice(others)]
                        break
            v = get_violations(colors)
            if v == 0:
                return [int(x) for x in colors]
            return [0] * NUM_EDGES
    return [0] * NUM_EDGES
