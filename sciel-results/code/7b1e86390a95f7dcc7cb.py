"""Strong edge colouring: DSATUR heuristic + bitmask conflict + intensive local search."""
import time
import math
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0

    def gen_instance(n, d, seed):
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

    def build_conflict_bitmasks(edges):
        m = len(edges)
        v2e = {}
        for i, (u, v) in enumerate(edges):
            v2e.setdefault(u, []).append(i)
            v2e.setdefault(v, []).append(i)

        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i

        conflict = [0] * m

        for u, lst in v2e.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    conflict[lst[a]] |= 1 << lst[b]
                    conflict[lst[b]] |= 1 << lst[a]

        for i, (u, v) in enumerate(edges):
            lu = v2e.get(u, [])
            lv = v2e.get(v, [])
            for a in lu:
                if a == i:
                    continue
                for b in lv:
                    if b == i or b == a:
                        continue
                    conflict[a] |= 1 << b
                    conflict[b] |= 1 << a

        return conflict

    def dsatur_coloring(m, conflict, degrees):
        """DSATUR: pick vertex with max saturation, then max degree."""
        colour = [-1] * m
        sat = [0] * m  # bitmask of colours used by neighbors
        uncolored = list(range(m))
        used = 0

        # Precompute degree in conflict graph
        conf_deg = [bin(conflict[i]).count('1') for i in range(m)]

        for _ in range(m):
            if not uncolored:
                break
            # Find vertex with max saturation, break ties by conflict degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                s = bin(sat[v]).count('1')
                d = conf_deg[v]
                if s > best_sat or (s == best_sat and d > best_deg):
                    best_sat = s
                    best_deg = d
                    best_v = v

            if best_v == -1:
                break

            # Find smallest available colour
            placed = False
            for c in range(used):
                if not (sat[best_v] >> c) & 1:
                    colour[best_v] = c
                    placed = True
                    break
            if not placed:
                colour[best_v] = used
                used += 1

            # Update saturation of neighbors
            for u in uncolored:
                if u == best_v:
                    continue
                if (conflict[best_v] >> u) & 1:
                    sat[u] |= 1 << colour[best_v]

            uncolored.remove(best_v)

        return colour, used

    def local_search(edges, conflict, colour, used, rng, time_budget=5.0):
        """Intensive local search with simulated annealing: merge, move, swap."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

        classes = {}
        class_masks = {}
        for e in range(m):
            c = col_of[e]
            if c >= 0:
                if c not in classes:
                    classes[c] = []
                    class_masks[c] = 0
                classes[c].append(e)
                class_masks[c] |= 1 << e

        def get_num_colors():
            return len(classes)

        # Phase 1: Greedy improvement (merge, move, swap)
        for _ in range(5000):
            if time.time() - start > time_budget:
                break
            improved = False
            maxc = max(classes.keys()) if classes else -1
            if maxc < 0:
                break

            # 1) Try merging colour classes (highest into lower)
            for c in range(maxc, -1, -1):
                if c not in classes:
                    continue
                cls_c = classes[c]
                mask_c = class_masks[c]
                for d in range(c):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    ok = True
                    for e in cls_c:
                        if conflict[e] & mask_d:
                            ok = False
                            break
                    if ok:
                        for e in cls_c:
                            col_of[e] = d
                        classes[d].extend(cls_c)
                        class_masks[d] |= mask_c
                        del classes[c]
                        del class_masks[c]
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # 2) Try moving a single edge to a lower colour
            for e in range(m):
                ce = col_of[e]
                if ce <= 0:
                    continue
                best_d = None
                for d in range(ce):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    if conflict[e] & mask_d == 0:
                        best_d = d
                        break
                if best_d is not None:
                    col_of[e] = best_d
                    classes[best_d].append(e)
                    class_masks[best_d] |= 1 << e
                    classes[ce].remove(e)
                    class_masks[ce] &= ~(1 << e)
                    if not classes[ce]:
                        del classes[ce]
                        del class_masks[ce]
                    improved = True
                    break

            if improved:
                continue

            # 3) Try 1-swap
            for _ in range(30):
                e = rng.randrange(m)
                ce = col_of[e]
                if ce <= 0:
                    continue
                lower_colors = [d for d in range(ce) if d in classes and classes[d]]
                if not lower_colors:
                    continue
                d = rng.choice(lower_colors)
                f = rng.choice(classes[d])

                mask_d_without_f = class_masks[d] & ~(1 << f)
                if conflict[e] & mask_d_without_f:
                    continue
                mask_ce_without_e = class_masks[ce] & ~(1 << e)
                if conflict[f] & mask_ce_without_e:
                    continue

                col_of[e] = d
                col_of[f] = ce
                classes[d].remove(f)
                classes[d].append(e)
                class_masks[d] = (class_masks[d] & ~(1 << f)) | (1 << e)
                classes[ce].remove(e)
                classes[ce].append(f)
                class_masks[ce] = (class_masks[ce] & ~(1 << e)) | (1 << f)
                improved = True
                break

            if not improved:
                break

        # Phase 2: Simulated annealing with incremental state
        best_col = col_of[:]
        best_nc = get_num_colors()
        current_nc = best_nc
        T_start = max(1.0, current_nc * 0.1)
        T_end = 0.01
        sa_start = time.time()
        sa_budget = max(1.0, time_budget - (sa_start - start))

        for it in range(20000):
            if time.time() - sa_start > sa_budget:
                break
            T = T_start * (T_end / T_start) ** (it / 20000.0)

            # Random move: pick an edge and try to move it to a random different color
            e = rng.randrange(m)
            ce = col_of[e]
            # Pick a target color (could be existing or new)
            if classes:
                target_colors = list(classes.keys())
                if len(target_colors) > 1:
                    target_colors.remove(ce)
                if target_colors:
                    d = rng.choice(target_colors)
                else:
                    d = ce + 1
            else:
                d = 0

            if d == ce:
                continue

            # Check if move is valid
            mask_d = class_masks.get(d, 0)
            if conflict[e] & mask_d:
                continue

            # Perform move
            old_d = ce
            col_of[e] = d
            classes[old_d].remove(e)
            class_masks[old_d] &= ~(1 << e)
            if not classes[old_d]:
                del classes[old_d]
                del class_masks[old_d]

            if d not in classes:
                classes[d] = []
                class_masks[d] = 0
            classes[d].append(e)
            class_masks[d] |= 1 << e

            new_nc = get_num_colors()
            delta = new_nc - current_nc

            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                current_nc = new_nc
                if new_nc < best_nc:
                    best_nc = new_nc
                    best_col = col_of[:]
            else:
                # Revert
                col_of[e] = old_d
                classes[d].remove(e)
                class_masks[d] &= ~(1 << e)
                if not classes[d]:
                    del classes[d]
                    del class_masks[d]

                if old_d not in classes:
                    classes[old_d] = []
                    class_masks[old_d] = 0
                classes[old_d].append(e)
                class_masks[old_d] |= 1 << e
                current_nc = get_num_colors()

        # Final greedy polish
        col_of = best_col[:]
        for e in range(m):
            c = col_of[e]
            if c >= 0:
                if c not in classes:
                    classes[c] = []
                    class_masks[c] = 0
                classes[c].append(e)
                class_masks[c] |= 1 << e

        for _ in range(2000):
            if time.time() - start > time_budget:
                break
            improved = False
            maxc = max(classes.keys()) if classes else -1
            if maxc < 0:
                break

            for c in range(maxc, -1, -1):
                if c not in classes:
                    continue
                cls_c = classes[c]
                mask_c = class_masks[c]
                for d in range(c):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    ok = True
                    for e in cls_c:
                        if conflict[e] & mask_d:
                            ok = False
                            break
                    if ok:
                        for e in cls_c:
                            col_of[e] = d
                        classes[d].extend(cls_c)
                        class_masks[d] |= mask_c
                        del classes[c]
                        del class_masks[c]
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            for e in range(m):
                ce = col_of[e]
                if ce <= 0:
                    continue
                best_d = None
                for d in range(ce):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    if conflict[e] & mask_d == 0:
                        best_d = d
                        break
                if best_d is not None:
                    col_of[e] = best_d
                    classes[best_d].append(e)
                    class_masks[best_d] |= 1 << e
                    classes[ce].remove(e)
                    class_masks[ce] &= ~(1 << e)
                    if not classes[ce]:
                        del classes[ce]
                        del class_masks[ce]
                    improved = True
                    break

            if not improved:
                break

        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    def two_swap_search(edges, conflict, colour, used, rng, time_budget=5.0):
        """Try 2-swap moves to enable merges with incremental tracking."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

        classes = {}
        class_masks = {}
        for e in range(m):
            c = col_of[e]
            if c >= 0:
                if c not in classes:
                    classes[c] = []
                    class_masks[c] = 0
                classes[c].append(e)
                class_masks[c] |= 1 << e

        best_col = col_of[:]
        best_nc = len(classes)

        for _ in range(10000):
            if time.time() - start > time_budget:
                break
            maxc = max(classes.keys()) if classes else -1
            if maxc < 1:
                break

            a = rng.randrange(maxc + 1)
            b = rng.randrange(a) if a > 0 else 0
            if a == b or a not in classes or b not in classes:
                continue
            cls_a = classes[a]
            cls_b = classes[b]
            if not cls_a or not cls_b:
                continue
            e = rng.choice(cls_a)
            f = rng.choice(cls_b)

            mask_b_without_f = class_masks[b] & ~(1 << f)
            if conflict[e] & mask_b_without_f:
                continue
            mask_a_without_e = class_masks[a] & ~(1 << e)
            if conflict[f] & mask_a_without_e:
                continue

            col_of[e] = b
            col_of[f] = a
            classes[b].remove(f)
            classes[b].append(e)
            class_masks[b] = (class_masks[b] & ~(1 << f)) | (1 << e)
            classes[a].remove(e)
            classes[a].append(f)
            class_masks[a] = (class_masks[a] & ~(1 << e)) | (1 << f)

            # Try to merge after swap
            merged = False
            for c in range(maxc, -1, -1):
                if c not in classes:
                    continue
                cls_c = classes[c]
                mask_c = class_masks[c]
                for d in range(c):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    ok = True
                    for edge_c in cls_c:
                        if conflict[edge_c] & mask_d:
                            ok = False
                            break
                    if ok:
                        for edge_c in cls_c:
                            col_of[edge_c] = d
                        classes[d].extend(cls_c)
                        class_masks[d] |= mask_c
                        del classes[c]
                        del class_masks[c]
                        merged = True
                        break
                if c not in classes:
                    break

            if merged:
                cur_nc = len(classes)
                if cur_nc < best_nc:
                    best_nc = cur_nc
                    best_col = col_of[:]

        present = sorted(set(best_col))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            best_col[e] = remap[best_col[e]]
        return best_col, len(present)

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = gen_instance(n, d, seed)
        conflict = build_conflict_bitmasks(edges)
        m = len(edges)
        best_colour = None
        best_used = m
        rng = Random(42)

        # DSATUR as primary heuristic
        degrees = [bin(conflict[i]).count('1') for i in range(m)]
        colour, used = dsatur_coloring(m, conflict, degrees)
        best_colour = colour[:]
        best_used = used

        # Local compression on DSATUR result (with SA)
        best_colour, best_used = local_search(edges, conflict, best_colour, best_used, rng, time_budget=25.0)

        # Two-swap search for further improvement
        if time.time() < DEADLINE - 15.0:
            best_colour, best_used = two_swap_search(edges, conflict, best_colour, best_used, rng, time_budget=15.0)

        # Additional DSATUR restarts with different tie-breaking if time permits
        if time.time() < DEADLINE - 10.0:
            for r in range(30):
                if time.time() > DEADLINE - 5.0:
                    break
                # Perturb: shuffle edges slightly to get different DSATUR ordering
                order = list(range(m))
                rng.shuffle(order)
                # Remap conflict for shuffled order
                perm = [0] * m
                for new_i, old_i in enumerate(order):
                    perm[old_i] = new_i
                new_conflict = [0] * m
                for i in range(m):
                    for j in range(m):
                        if (conflict[i] >> j) & 1:
                            new_conflict[perm[i]] |= 1 << perm[j]
                col2, used2 = dsatur_coloring(m, new_conflict, [bin(new_conflict[i]).count('1') for i in range(m)])
                # Map back to original edge indices
                col_orig = [0] * m
                for new_i, old_i in enumerate(order):
                    col_orig[old_i] = col2[new_i]
                if used2 <= best_used:
                    col_orig, used2 = local_search(edges, conflict, col_orig, used2, rng, time_budget=3.0)
                    if used2 < best_used:
                        best_colour = col_orig[:]
                        best_used = used2

        if best_colour is None:
            best_colour, best_used = dsatur_coloring(m, conflict, degrees)

        out.append(best_colour)

    return out
