"""Strong edge colouring: DSATUR + CS backtracking (MRV+FC) + intensive local search."""
import time
import math
from random import Random
from collections import defaultdict


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

    def dsatur_coloring(m, conflict):
        """DSATUR: pick vertex with max saturation, then max degree."""
        colour = [-1] * m
        sat = [0] * m
        uncolored = list(range(m))
        used = 0

        conf_deg = [bin(conflict[i]).count('1') for i in range(m)]

        for _ in range(m):
            if not uncolored:
                break
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

            placed = False
            for c in range(used):
                if not (sat[best_v] >> c) & 1:
                    colour[best_v] = c
                    placed = True
                    break
            if not placed:
                colour[best_v] = used
                used += 1

            for u in uncolored:
                if u == best_v:
                    continue
                if (conflict[best_v] >> u) & 1:
                    sat[u] |= 1 << colour[best_v]

            uncolored.remove(best_v)

        return colour, used

    def cs_backtracking(m, conflict, max_colors, deadline):
        """Constraint satisfaction backtracking with MRV and forward checking."""
        colour = [-1] * m
        color_classes = [set() for _ in range(max_colors)]
        forbidden = [0] * m

        conf_deg = [bin(conflict[i]).count('1') for i in range(m)]
        edge_order = sorted(range(m), key=lambda i: -conf_deg[i])

        def is_valid_color(edge, color):
            if (forbidden[edge] >> color) & 1:
                return False
            for other in color_classes[color]:
                if conflict[edge] & (1 << other):
                    return False
            return True

        def assign(edge, color):
            colour[edge] = color
            color_classes[color].add(edge)
            for nb in range(m):
                if nb == edge:
                    continue
                if (conflict[edge] >> nb) & 1 and colour[nb] == -1:
                    forbidden[nb] |= (1 << color)

        def unassign(edge, color):
            colour[edge] = -1
            color_classes[color].discard(edge)
            for nb in range(m):
                if nb == edge:
                    continue
                if (conflict[edge] >> nb) & 1 and colour[nb] == -1:
                    forbidden[nb] &= ~(1 << color)

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == m:
                return True

            edge = edge_order[idx]

            for color in range(max_colors):
                if is_valid_color(edge, color):
                    assign(edge, color)
                    if backtrack(idx + 1):
                        return True
                    unassign(edge, color)

            return False

        if backtrack(0):
            used = 0
            for c in range(max_colors):
                if color_classes[c]:
                    used = c + 1
            return colour, used
        return None, None

    def local_search(edges, conflict, colour, used, rng, time_budget=5.0):
        """Intensive local search: merge, move, swap, 2-swap."""
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

        def try_merge():
            maxc = max(classes.keys()) if classes else -1
            if maxc < 0:
                return False
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
                        return True
            return False

        for _ in range(10000):
            if time.time() - start > time_budget:
                break
            improved = False

            if try_merge():
                continue

            edges_list = list(range(m))
            rng.shuffle(edges_list)
            for e in edges_list:
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

        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    def two_swap_search(edges, conflict, colour, used, rng, time_budget=5.0):
        """Try 2-swap moves to enable merges."""
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

        for _ in range(5000):
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
                        break
                if c not in classes:
                    break

        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

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
        colour, used = dsatur_coloring(m, conflict)
        best_colour = colour[:]
        best_used = used

        # Local compression on DSATUR result
        best_colour, best_used = local_search(edges, conflict, best_colour, best_used, rng, time_budget=15.0)

        # Try CS backtracking with tight bound
        if time.time() < DEADLINE - 20.0:
            cs_deadline = time.time() + 8.0
            cs_col, cs_used = cs_backtracking(m, conflict, best_used, cs_deadline)
            if cs_col is not None and cs_used < best_used:
                best_colour, best_used = cs_col, cs_used
                best_colour, best_used = local_search(edges, conflict, best_colour, best_used, rng, time_budget=5.0)

        # Two-swap search for further improvement
        if time.time() < DEADLINE - 10.0:
            best_colour, best_used = two_swap_search(edges, conflict, best_colour, best_used, rng, time_budget=10.0)

        # Additional DSATUR restarts with different tie-breaking if time permits
        if time.time() < DEADLINE - 5.0:
            for r in range(30):
                if time.time() > DEADLINE - 3.0:
                    break
                order = list(range(m))
                rng.shuffle(order)
                perm = [0] * m
                for new_i, old_i in enumerate(order):
                    perm[old_i] = new_i
                new_conflict = [0] * m
                for i in range(m):
                    for j in range(m):
                        if (conflict[i] >> j) & 1:
                            new_conflict[perm[i]] |= 1 << perm[j]
                col2, used2 = dsatur_coloring(m, new_conflict)
                col_orig = [0] * m
                for new_i, old_i in enumerate(order):
                    col_orig[old_i] = col2[new_i]
                if used2 <= best_used:
                    col_orig, used2 = local_search(edges, conflict, col_orig, used2, rng, time_budget=2.0)
                    if used2 < best_used:
                        best_colour = col_orig[:]
                        best_used = used2

        if best_colour is None:
            best_colour, best_used = dsatur_coloring(m, conflict)

        out.append(best_colour)

    return out
