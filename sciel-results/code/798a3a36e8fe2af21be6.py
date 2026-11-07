import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 84  # safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    c1_idx, c2_idx, vert_idx = [], [], []
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            c1_idx.append(i)
        elif u >= m and v >= m:
            c2_idx.append(i)
        else:
            vert_idx.append(i)

    c1_from = [min(u, v) for (u, v) in [edges[i] for i in c1_idx]]
    c2_from = [min(u, v) - m for (u, v) in [edges[i] for i in c2_idx]]
    vert_top = [u if u < m else v - m for (u, v) in [edges[i] for i in vert_idx]]
    vert_pos = [0] * m
    for k, top in enumerate(vert_top):
        vert_pos[top] = k
    vert_order = np.zeros(m, dtype=int)
    vert_order[vert_pos] = np.arange(m)

    eu = np.array([u for u, v in edges], dtype=int)
    ev = np.array([v for u, v in edges], dtype=int)

    def full_radius(signs):
        A = np.zeros((n_vertices, n_vertices))
        A[eu, ev] = signs
        A[ev, eu] = signs
        return float(np.max(np.abs(np.linalg.eigvalsh(A))))

    def prod_cycle(s):
        p = 1
        for x in s:
            p *= x
        return p

    def connected(c1s, c2s, vsign):
        return (prod_cycle(c1s) * prod_cycle(c2s) * vsign[0] * vsign[m - 1]) == -1

    def build_signs(a, b, vsign):
        signs = np.ones(n_edges, dtype=int)
        for j, idx in enumerate(c1_idx):
            signs[idx] = 1 if (a >> j) & 1 else -1
        for j, idx in enumerate(c2_idx):
            signs[idx] = 1 if (b >> j) & 1 else -1
        for k in range(m):
            signs[vert_idx[vert_order[k]]] = vsign[k]
        return signs

    n_c = 1 << m
    c1s_list, c2s_list, prod1, prod2 = [], [], [], []
    for code in range(n_c):
        s = np.array([1 if (code >> j) & 1 else -1 for j in range(m)], dtype=int)
        c1s_list.append(s)
        c2s_list.append(s)
        prod1.append(int(np.prod(s)))
        prod2.append(int(np.prod(s)))

    best_radius = float('inf')
    best_a, best_b, best_v = 0, 0, None

    # Exhaustive over cycle pairs; pick a default vertical signing per pair.
    for a in range(n_c):
        if time.time() - start_time > DEADLINE - 4:
            break
        c1s = c1s_list[a]
        for b in range(n_c):
            c2s = c2s_list[b]
            need = -1 // (prod1[a] * prod2[b])
            vsign = np.ones(m, dtype=int)
            if need == -1:
                vsign[m - 1] = -1
            signs = build_signs(a, b, vsign)
            r = full_radius(signs)
            if r < best_radius:
                best_radius = r
                best_a, best_b, best_v = a, b, vsign.copy()

    best_signs = build_signs(best_a, best_b, best_v)

    # Vertical polish: local search on vertical signs keeping connectivity.
    if time.time() - start_time < DEADLINE - 2:
        cur = best_signs.copy()
        cur_r = best_radius
        improved = True
        it = 0
        while improved and it < 10 and (time.time() - start_time < DEADLINE - 1):
            improved = False
            it += 1
            for k in range(m):
                eidx = vert_idx[vert_order[k]]
                cur[eidx] *= -1
                if not connected(c1s_list[best_a], c2s_list[best_b],
                                 np.array([cur[vert_idx[vert_order[q]]] for q in range(m)])):
                    cur[eidx] *= -1
                    continue
                r = full_radius(cur)
                if r < cur_r - 1e-12:
                    cur_r = r
                    improved = True
                else:
                    cur[eidx] *= -1
        if cur_r < best_radius:
            best_radius = cur_r
            best_signs = cur

    # Final safety: ensure validity
    if time.time() - start_time < DEADLINE - 0.5:
        if not connected(c1s_list[best_a], c2s_list[best_b],
                         np.array([best_signs[vert_idx[vert_order[q]]] for q in range(m)])):
            # fix by flipping a vertical
            for k in range(m):
                eidx = vert_idx[vert_order[k]]
                best_signs[eidx] *= -1
                if connected(c1s_list[best_a], c2s_list[best_b],
                             np.array([best_signs[vert_idx[vert_order[q]]] for q in range(m)])):
                    break

    return best_signs
