import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}

    # Precompute all K4s
    k4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
                    edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]]
        k4s.append(k4_edges)
    K = len(k4s)

    # For each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4_edges in enumerate(k4s):
        for e in k4_edges:
            edge_to_k4s[e].append(k4_idx)

    # For each K4, position of each edge (for fast lookup)
    k4_edge_pos = []
    for k4_edges in k4s:
        pos = {}
        for p, e in enumerate(k4_edges):
            pos[e] = p
        k4_edge_pos.append(pos)

    # Incremental state: for each K4, maintain colour counts
    # k4_colour_counts[k4_idx] = list of 6 colours (one per edge position)
    def compute_k4_state(colors):
        k4_cc = []
        for k4_edges in k4s:
            k4_cc.append([colors[e] for e in k4_edges])
        return k4_cc

    def k4_distinct(k4_cc_row):
        # Count distinct colours in a 6-element list
        seen = [False] * 120
        cnt = 0
        for c in k4_cc_row:
            if not seen[c]:
                seen[c] = True
                cnt += 1
        return cnt

    def total_violations(k4_cc):
        v = 0
        for row in k4_cc:
            d = k4_distinct(row)
            if d < 5:
                v += (5 - d)
        return v

    def palette_size(colors):
        return len(set(colors))

    # --- Initial solutions ---
    rng_init = random.Random(42)
    inits = [
        [(i + j) % 17 for (i, j) in edges],          # valid, 16 colours
        [(i * 3 + j) % 15 for (i, j) in edges],
        [(i * 7 + j) % 13 for (i, j) in edges],
    ]
    for _ in range(8):
        p = rng_init.randint(13, 16)
        inits.append([rng_init.randint(0, p - 1) for _ in range(E)])

    best_colors = None
    best_palette = 999
    best_viol = 0

    for init in inits:
        k4_cc = compute_k4_state(init)
        v = total_violations(k4_cc)
        p = palette_size(init)
        if v == 0 and p < best_palette:
            best_palette = p
            best_colors = init[:]
            best_viol = 0
            if p <= 13:
                break

    if best_colors is None:
        # Fallback: additive scheme (always valid)
        best_colors = [(i + j) % 17 for (i, j) in edges]
        best_palette = 16

    # --- Simulated Annealing ---
    current = best_colors[:]
    k4_cc = compute_k4_state(current)
    current_viol = total_violations(k4_cc)
    current_pal = palette_size(current)

    # Colour counts for palette tracking
    colour_counts = [0] * 120
    for c in current:
        colour_counts[c] += 1

    def delta_palette(c_old, c_new):
        d = 0
        colour_counts[c_old] -= 1
        colour_counts[c_new] += 1
        if colour_counts[c_old] == 0:
            d -= 1
        if colour_counts[c_new] == 1:
            d += 1
        return d

    def undo_delta_palette(c_new, c_old):
        d = 0
        colour_counts[c_new] -= 1
        colour_counts[c_old] += 1
        if colour_counts[c_new] == 0:
            d -= 1
        if colour_counts[c_old] == 1:
            d += 1
        return d

    rng = random.Random(12345)
    T = 15.0
    T_min = 0.05
    cooling = 0.9997

    # Max colour to try (encourage small palette)
    max_colour_try = 16

    while time.time() < deadline - 3:
        if T < T_min:
            T = T_min

        e = rng.randint(0, E - 1)
        c_old = current[e]
        c_new = rng.randint(0, max_colour_try - 1)
        if c_new == c_old:
            continue

        # Compute delta violations using incremental K4 state
        delta_viol = 0
        affected = edge_to_k4s[e]
        for k4_idx in affected:
            row = k4_cc[k4_idx]
            pos = k4_edge_pos[k4_idx][e]
            old_c = row[pos]

            # Old distinct count
            old_seen = [False] * 120
            old_cnt = 0
            for c in row:
                if not old_seen[c]:
                    old_seen[c] = True
                    old_cnt += 1

            # New distinct count (change one element)
            new_cnt = old_cnt
            if old_seen[old_c] and row.count(old_c) == 1:
                # old_c disappears
                new_cnt -= 1
            if not old_seen[c_new]:
                # c_new is new
                new_cnt += 1

            old_v = max(0, 5 - old_cnt)
            new_v = max(0, 5 - new_cnt)
            delta_viol += (new_v - old_v)

        delta_pal = delta_palette(c_old, c_new)
        delta_cost = delta_viol * 100 + delta_pal

        accept = False
        if delta_cost <= 0:
            accept = True
        else:
            if delta_cost > 60 * T:
                accept = False
            else:
                ratio = delta_cost / T
                if ratio > 500:
                    accept = False
                else:
                    # Use log comparison to avoid underflow
                    import math
                    if math.log(rng.random()) < -ratio:
                        accept = True

        if accept:
            # Apply move
            current[e] = c_new
            for k4_idx in affected:
                row = k4_cc[k4_idx]
                pos = k4_edge_pos[k4_idx][e]
                row[pos] = c_new
            current_viol += delta_viol
            current_pal += delta_pal

            if current_viol == 0 and current_pal < best_palette:
                best_palette = current_pal
                best_colors = current[:]
                if best_palette <= 13:
                    break
        else:
            # Undo palette change
            undo_delta_palette(c_new, c_old)

        T *= cooling

    # --- Final polish: try to merge colours in best valid solution ---
    if best_colors is not None and best_palette > 13:
        best_k4_cc = compute_k4_state(best_colors)
        best_viol = total_violations(best_k4_cc)
        if best_viol == 0:
            # Try pairwise colour merges
            used = sorted(set(best_colors))
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for src in used[:]:
                    for tgt in used[:]:
                        if src == tgt:
                            continue
                        test = best_colors[:]
                        for idx in range(E):
                            if test[idx] == src:
                                test[idx] = tgt
                        new_pal = palette_size(test)
                        if new_pal >= best_palette:
                            continue
                        # Validate
                        test_k4_cc = compute_k4_state(test)
                        test_viol = total_violations(test_k4_cc)
                        if test_viol == 0:
                            best_colors = test
                            best_palette = new_pal
                            used = sorted(set(best_colors))
                            improved = True
                            break
                    if improved:
                        break

    return best_colors
