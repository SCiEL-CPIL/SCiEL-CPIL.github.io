"""Strong edge colouring: DSATUR + bitmask-accelerated greedy + local search with annealing."""
import time
import math
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin under the 150 s kill

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
        """Build conflict bitmask for each edge using Python ints as bitsets."""
        m = len(edges)
        v2e = {}
        for i, (u, v) in enumerate(edges):
            v2e.setdefault(u, []).append(i)
            v2e.setdefault(v, []).append(i)

        conflict = [0] * m

        # 1) Share a vertex
        for u, lst in v2e.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    conflict[lst[a]] |= 1 << lst[b]
                    conflict[lst[b]] |= 1 << lst[a]

        # 2) Joined by an edge
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

    def greedy_first_fit(edges, conflict, order):
        """Greedy first-fit using conflict bitmasks. Returns (colour_list, num_colours)."""
        m = len(edges)
        colour = [-1] * m
        used = 0
        col_masks = []

        for e in order:
            ce = 1 << e
            placed = False
            for c in range(used):
                if col_masks[c] & conflict[e] == 0:
                    colour[e] = c
                    col_masks[c] |= ce
                    placed = True
                    break
            if not placed:
                colour[e] = used
                col_masks.append(ce)
                used += 1
        return colour, used

    def dsatur_order(edges, conflict, m, rng=None):
        """
        DSATUR-style ordering: at each step pick the uncoloured edge with the
        highest saturation (number of distinct colours among its coloured neighbours),
        tie-break by degree in the conflict graph, then randomly if rng is given.
        Returns a list of edge indices in the order they should be coloured.
        """
        coloured = [False] * m
        order = []
        # degree in conflict graph
        deg = [bin(conflict[i]).count('1') for i in range(m)]

        # colour_seen[e] = set of colours used by coloured neighbours of e
        colour_seen = [set() for _ in range(m)]

        for _ in range(m):
            best_e = -1
            best_sat = -1
            best_deg = -1
            candidates = []
            for e in range(m):
                if coloured[e]:
                    continue
                sat = len(colour_seen[e])
                if sat > best_sat:
                    best_sat = sat
                    best_deg = deg[e]
                    best_e = e
                    candidates = [e]
                elif sat == best_sat:
                    if deg[e] > best_deg:
                        best_deg = deg[e]
                        best_e = e
                        candidates = [e]
                    elif deg[e] == best_deg:
                        candidates.append(e)
            if not candidates:
                break
            if rng is not None and len(candidates) > 1:
                best_e = rng.choice(candidates)
            else:
                best_e = candidates[0]
            order.append(best_e)
            coloured[best_e] = True
            # update colour_seen for uncoloured neighbours
            # we don't know the colour yet; we'll update after assignment in greedy.
            # For ordering purposes, we just mark as coloured.
        return order

    def dsatur_greedy(edges, conflict, m, rng=None):
        """
        DSATUR-guided greedy: dynamically pick the next edge to colour based on
        saturation, then assign the lowest available colour.
        Returns (colour_list, num_colours).
        """
        colour = [-1] * m
        col_masks = []  # bitmask of edges in each colour
        used = 0
        coloured = [False] * m
        colour_seen = [set() for _ in range(m)]
        deg = [bin(conflict[i]).count('1') for i in range(m)]

        for _ in range(m):
            # pick edge with highest saturation, tie-break by degree
            best_e = -1
            best_sat = -1
            best_deg = -1
            candidates = []
            for e in range(m):
                if coloured[e]:
                    continue
                sat = len(colour_seen[e])
                if sat > best_sat:
                    best_sat = sat
                    best_deg = deg[e]
                    best_e = e
                    candidates = [e]
                elif sat == best_sat:
                    if deg[e] > best_deg:
                        best_deg = deg[e]
                        best_e = e
                        candidates = [e]
                    elif deg[e] == best_deg:
                        candidates.append(e)
            if not candidates:
                break
            if rng is not None and len(candidates) > 1:
                e = rng.choice(candidates)
            else:
                e = candidates[0]

            # assign lowest available colour
            ce = 1 << e
            placed = False
            for c in range(used):
                if col_masks[c] & conflict[e] == 0:
                    colour[e] = c
                    col_masks[c] |= ce
                    placed = True
                    break
            if not placed:
                colour[e] = used
                col_masks.append(ce)
                used += 1

            coloured[e] = True
            # update colour_seen for uncoloured neighbours
            c_assigned = colour[e]
            # neighbours of e in conflict graph
            cf = conflict[e]
            # iterate over set bits in cf
            cf_temp = cf
            while cf_temp:
                lsb = cf_temp & (-cf_temp)
                f = lsb.bit_length() - 1
                cf_temp ^= lsb
                if not coloured[f]:
                    colour_seen[f].add(c_assigned)

        return colour, used

    def local_search(edges, conflict, colour, used, rng, time_budget=3.0):
        """Local search: try edge moves, swaps, and class merges. Returns (colour, used)."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

        def build_classes():
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
            return classes, class_masks

        classes, class_masks = build_classes()

        for _ in range(2000):
            if time.time() - start > time_budget:
                break
            improved = False

            # Rebuild classes if they might be stale (after a move)
            # We track this with a dirty flag, but for simplicity rebuild each iteration
            # since moves are infrequent relative to the budget
            classes, class_masks = build_classes()
            maxc = max(classes.keys()) if classes else 0

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

            # 3) Try 1-swap
            if rng.random() < 0.7:
                for _ in range(20):
                    e = rng.randrange(m)
                    ce = col_of[e]
                    if ce <= 0:
                        continue

                    lower_colors = [d for d in range(ce) if d in classes]
                    if not lower_colors:
                        continue
                    d = rng.choice(lower_colors)

                    if not classes[d]:
                        continue
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

    def two_swap_search(edges, conflict, colour, used, rng, time_budget=3.0):
        """Try 2-swap moves: swap edges between two colour classes to enable merges."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

        for _ in range(800):
            if time.time() - start > time_budget:
                break
            improved = False

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

            maxc = max(classes.keys()) if classes else 0

            if maxc < 1:
                break

            a = rng.randrange(maxc + 1)
            b = rng.randrange(a) if a > 0 else 0
            if a == b or a not in classes or b not in classes:
                continue

            cls_a = classes[a]
            cls_b = classes[b]
            mask_a = class_masks[a]
            mask_b = class_masks[b]

            if not cls_a or not cls_b:
                continue
            e = rng.choice(cls_a)
            f = rng.choice(cls_b)

            mask_b_without_f = mask_b & ~(1 << f)
            if conflict[e] & mask_b_without_f:
                continue

            mask_a_without_e = mask_a & ~(1 << e)
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

            improved = True

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
                        improved = True
                        break
                if improved:
                    break

        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    def simulated_annealing(edges, conflict, colour, used, rng, time_budget=5.0):
        """SA-style perturbation and local search."""
        m = len(edges)
        best_colour = colour[:]
        best_used = used
        cur_colour = colour[:]
        cur_used = used
        start = time.time()
        T = 10.0

        while time.time() - start < time_budget:
            if rng.random() < 0.5:
                e = rng.randrange(m)
                old_c = cur_colour[e]
                new_c = rng.randrange(used + 1)
                if new_c == old_c:
                    continue
                cur_colour[e] = new_c

                # Check validity using conflict bitmask
                valid = True
                # Find all edges with the same new color that conflict with e
                # We need to check if any edge with color new_c conflicts with e
                # Build a mask of edges with color new_c
                same_color_mask = 0
                for f in range(m):
                    if cur_colour[f] == new_c and f != e:
                        same_color_mask |= 1 << f
                if conflict[e] & same_color_mask:
                    valid = False

                if not valid:
                    cur_colour[e] = old_c
                    continue
            else:
                e1 = rng.randrange(m)
                e2 = rng.randrange(m)
                if e1 == e2:
                    continue
                c1 = cur_colour[e1]
                c2 = cur_colour[e2]
                if c1 == c2:
                    continue

                cur_colour[e1] = c2
                cur_colour[e2] = c1

                # Check validity
                valid = True
                # Check e1 with color c2
                same_color_mask_c2 = 0
                for f in range(m):
                    if cur_colour[f] == c2 and f != e1:
                        same_color_mask_c2 |= 1 << f
                if conflict[e1] & same_color_mask_c2:
                    valid = False

                if valid:
                    # Check e2 with color c1
                    same_color_mask_c1 = 0
                    for f in range(m):
                        if cur_colour[f] == c1 and f != e2:
                            same_color_mask_c1 |= 1 << f
                    if conflict[e2] & same_color_mask_c1:
                        valid = False

                if not valid:
                    cur_colour[e1] = c1
                    cur_colour[e2] = c2
                    continue

            present = set(cur_colour)
            cur_used = len(present)

            if cur_used < best_used:
                best_colour = cur_colour[:]
                best_used = cur_used

            cur_colour, cur_used = local_search(edges, conflict, cur_colour, cur_used, rng, time_budget=0.5)

            if cur_used < best_used:
                best_colour = cur_colour[:]
                best_used = cur_used

            T *= 0.95

        return best_colour, best_used

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = gen_instance(*spec)
        conflict = build_conflict_bitmasks(edges)
        m = len(edges)
        best_colour = None
        best_used = m
        rng = Random(42)

        # Phase 1: DSATUR greedy (deterministic and randomized)
        # Deterministic DSATUR
        colour_ds, used_ds = dsatur_greedy(edges, conflict, m, rng=None)
        if used_ds < best_used:
            best_colour = colour_ds[:]
            best_used = used_ds

        # Randomized DSATUR restarts
        n_ds_restarts = 40
        for r in range(n_ds_restarts):
            if time.time() > DEADLINE - 20.0:
                break
            colour_ds, used_ds = dsatur_greedy(edges, conflict, m, rng=rng)
            if used_ds < best_used:
                best_colour = colour_ds[:]
                best_used = used_ds

        # Phase 2: Random-order greedy restarts
        n_restarts = 80
        for r in range(n_restarts):
            if time.time() > DEADLINE - 20.0:
                break
            order = list(range(m))
            rng.shuffle(order)
            colour, used = greedy_first_fit(edges, conflict, order)
            if used < best_used:
                best_colour = colour[:]
                best_used = used

        # Local compression on best
        if best_colour is not None:
            best_colour, best_used = local_search(edges, conflict, best_colour, best_used, rng, time_budget=8.0)

        # Two-swap search for further improvement
        if time.time() < DEADLINE - 25.0 and best_colour is not None:
            best_colour, best_used = two_swap_search(edges, conflict, best_colour, best_used, rng, time_budget=8.0)

        # Simulated annealing for further improvement
        if time.time() < DEADLINE - 25.0 and best_colour is not None:
            best_colour, best_used = simulated_annealing(edges, conflict, best_colour, best_used, rng, time_budget=12.0)

        # More targeted restarts if time permits
        if time.time() < DEADLINE - 30.0:
            for r in range(60):
                if time.time() > DEADLINE - 20.0:
                    break
                # Mix DSATUR and random-order
                if r % 2 == 0:
                    colour, used = dsatur_greedy(edges, conflict, m, rng=rng)
                else:
                    order = list(range(m))
                    rng.shuffle(order)
                    colour, used = greedy_first_fit(edges, conflict, order)
                if used <= best_used:
                    colour, used = local_search(edges, conflict, colour, used, rng, time_budget=2.0)
                    if used < best_used:
                        best_colour = colour[:]
                        best_used = used

        # Final polish: try to eliminate color classes one by one
        if best_colour is not None and time.time() < DEADLINE - 10.0:
            polish_start = time.time()
            polish_deadline = min(DEADLINE - 5.0, polish_start + 10.0)
            for attempt in range(50):
                if time.time() > polish_deadline:
                    break
                # Try to merge each color class into a lower one
                classes = {}
                class_masks = {}
                for e in range(m):
                    c = best_colour[e]
                    if c >= 0:
                        if c not in classes:
                            classes[c] = []
                            class_masks[c] = 0
                        classes[c].append(e)
                        class_masks[c] |= 1 << e

                if not classes:
                    break

                maxc = max(classes.keys())
                merged = False
                for c in range(maxc, 0, -1):
                    if c not in classes or not classes[c]:
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
                                best_colour[e] = d
                            merged = True
                            break
                    if merged:
                        break

                if not merged:
                    break

            # Remap colors
            present = sorted(set(best_colour))
            remap = {old: new for new, old in enumerate(present)}
            for e in range(m):
                best_colour[e] = remap[best_colour[e]]
            best_used = len(present)

        if best_colour is None:
            best_colour, best_used = greedy_first_fit(edges, conflict, list(range(m)))

        out.append(best_colour)

    return out
