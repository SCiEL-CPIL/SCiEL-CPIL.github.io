import random
import time
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 88.0  # safety margin before the 90s kill

    # Precompute all nonconstant 4-term APs (0-indexed) and per-position index lists.
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)

    def count_violations(coloring):
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations

    def hamming_weight_init():
        # Map 1-based position to popcount of its binary representation (0..7).
        coloring = [0] * N
        for i in range(N):
            coloring[i] = bin(i + 1).count("1")
        return coloring

    def greedy_init(k):
        # Assign positions left-to-right, always picking the smallest color that
        # does not create a violation in any 4-AP ending at the current position.
        coloring = [-1] * N
        coloring[0] = 0
        for i in range(1, N):
            chosen = -1
            for c in range(k):
                ok = True
                # i is the last term: i-3d, i-2d, i-d, i
                max_d4 = i // 3
                for d in range(1, max_d4 + 1):
                    a0 = i - 3 * d
                    a1 = i - 2 * d
                    a2 = i - d
                    if coloring[a0] < 0 or coloring[a1] < 0 or coloring[a2] < 0:
                        ok = False
                        break
                    if len({coloring[a0], coloring[a1], coloring[a2], c}) < 3:
                        ok = False
                        break
                if ok:
                    chosen = c
                    break
            if chosen < 0:
                chosen = 0  # fallback, will be repaired by local search
            coloring[i] = chosen
        return coloring

    def try_recolor_all(coloring, old_color, k):
        # Try to recolor every position currently holding old_color to some color
        # in 0..k-1 (other than old_color) while preserving validity.
        positions = [i for i in range(N) if coloring[i] == old_color]
        if not positions:
            return False, None
        working = coloring[:]
        order = sorted(positions, key=lambda i: len(ap_indices[i]))
        for idx in order:
            found = False
            for c in range(k):
                if c == old_color:
                    continue
                ok = True
                for ap_id in ap_indices[idx]:
                    ap = aps[ap_id]
                    c0 = working[ap[0]]
                    c1 = working[ap[1]]
                    c2 = working[ap[2]]
                    c3 = working[ap[3]]
                    c0 = c if ap[0] == idx else c0
                    c1 = c if ap[1] == idx else c1
                    c2 = c if ap[2] == idx else c2
                    c3 = c if ap[3] == idx else c3
                    if len({c0, c1, c2, c3}) < 3:
                        ok = False
                        break
                if ok:
                    working[idx] = c
                    found = True
                    break
            if not found:
                return False, None
        return True, working

    def merge_colors(coloring, k):
        # Repeatedly try to eliminate one color by recoloring all its occurrences.
        coloring = coloring[:]
        current_k = k
        changed = True
        while changed:
            if time.time() > deadline - 12.0:
                break
            changed = False
            colors_used = sorted(set(coloring))
            # Try least-used color first.
            for c in sorted(colors_used, key=lambda cc: coloring.count(cc)):
                if time.time() > deadline - 12.0:
                    break
                ok, new_coloring = try_recolor_all(coloring, c, current_k)
                if ok:
                    coloring = new_coloring
                    # Recompute effective k as max label + 1 (labels may be non-consecutive)
                    current_k = max(coloring) + 1
                    changed = True
                    break
        return coloring, len(set(coloring))

    def local_search(coloring, k, max_steps, T0=5.0, cooling=0.9995):
        current = coloring[:]
        current_violations = count_violations(current)
        best = current[:]
        best_violations = current_violations
        T = T0
        steps = 0
        while current_violations > 0 and steps < max_steps and time.time() < deadline - 3.0:
            steps += 1
            idx = random.randint(0, N - 1)
            old_color = current[idx]
            new_color = random.randint(0, k - 1)
            if old_color == new_color:
                continue
            delta = 0
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                c0 = current[ap[0]]
                c1 = current[ap[1]]
                c2 = current[ap[2]]
                c3 = current[ap[3]]
                old_set = len({c0, c1, c2, c3})
                n0 = new_color if ap[0] == idx else c0
                n1 = new_color if ap[1] == idx else c1
                n2 = new_color if ap[2] == idx else c2
                n3 = new_color if ap[3] == idx else c3
                new_set = len({n0, n1, n2, n3})
                if old_set < 3 and new_set >= 3:
                    delta -= 1
                elif old_set >= 3 and new_set < 3:
                    delta += 1
            if delta < 0:
                current[idx] = new_color
                current_violations += delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best = current[:]
            else:
                if delta == 0:
                    accept = random.random() < 0.5
                else:
                    if T > 0:
                        accept = random.random() < np.exp(-delta / T)
                    else:
                        accept = False
                if accept:
                    current[idx] = new_color
                    current_violations += delta
            T *= cooling
        return best if best_violations == 0 else (current if current_violations == 0 else best)

    # --- Phase 1: Deterministic structural initializations ---
    candidates = []

    # Hamming-weight initialization (0..7)
    hw = hamming_weight_init()
    hw_v = count_violations(hw)
    candidates.append((hw_v, hw, max(hw) + 1))

    # Greedy initializations for a range of k
    for k in range(3, 13):
        if time.time() > deadline - 40.0:
            break
        g = greedy_init(k)
        g_v = count_violations(g)
        candidates.append((g_v, g, k))

    # Random initializations for a few k
    for k in range(3, 9):
        if time.time() > deadline - 40.0:
            break
        for _ in range(3):
            r = [random.randint(0, k - 1) for _ in range(N)]
            r_v = count_violations(r)
            candidates.append((r_v, r, k))

    # Sort by violations then by k
    candidates.sort(key=lambda x: (x[0], x[2]))

    # --- Phase 2: For each candidate, repair to valid, then merge colors ---
    best_result = None
    best_count = N + 1
    for v, coloring, k in candidates:
        if time.time() > deadline - 15.0:
            break
        if v == 0:
            valid = coloring
        else:
            valid = local_search(coloring, k, max_steps=200000)
            if count_violations(valid) != 0:
                continue
        # Merge colors to reduce distinct label count
        merged, count = merge_colors(valid, max(valid) + 1)
        if count_violations(merged) != 0:
            merged = valid
            count = len(set(valid))
        if count < best_count:
            best_count = count
            best_result = merged
            if count <= 3:
                break

    if best_result is not None:
        return best_result

    # --- Phase 3: Fallback full search over k with local search ---
    for k in range(3, 129):
        if time.time() > deadline - 5.0:
            break
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
            coloring = [random.randint(0, k - 1) for _ in range(N)]
            repaired = local_search(coloring, k, max_steps=150000)
            if count_violations(repaired) == 0:
                merged, count = merge_colors(repaired, k)
                if count_violations(merged) == 0:
                    return merged
                return repaired

    # Final fallback: N distinct colors guarantees validity.
    return list(range(N))
