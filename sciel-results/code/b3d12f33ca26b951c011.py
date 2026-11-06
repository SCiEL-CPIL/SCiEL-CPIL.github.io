import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin vs 150s kill

    n = 20
    num_colors = 20

    # ---- Precompute 4-cycles (edge sets) and incidence ----
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    for u, v, w, x in itertools.combinations(range(n), 4):
        cycles = [
            [(u, v), (v, w), (w, x), (x, u)],
            [(u, v), (v, x), (x, w), (w, u)],
            [(u, w), (w, v), (v, x), (x, u)],
        ]
        for cycle in cycles:
            norm = tuple(sorted((min(e), max(e)) for e in cycle))
            if len(set(norm)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm)
                for e in norm:
                    edge_to_cycles[e].append(idx)

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # ---- Incremental cycle bookkeeping ----
    # present_count[c] = how many of the 4 edges of cycle c are currently in the graph
    # color_count[c]   = {colour: count} over the present edges of cycle c
    present_count = [0] * len(all_cycles)
    color_count = [defaultdict(int) for _ in range(len(all_cycles))]

    def add_edge(e, c):
        for idx in edge_to_cycles[e]:
            present_count[idx] += 1
            color_count[idx][c] += 1

    def remove_edge(e, c):
        for idx in edge_to_cycles[e]:
            present_count[idx] -= 1
            d = color_count[idx]
            d[c] -= 1
            if d[c] == 0:
                del d[c]

    def valid_color(e, c, edge_colors):
        """True iff adding edge e with colour c keeps every 4-cycle through e rainbow."""
        for idx in edge_to_cycles[e]:
            if present_count[idx] == 3:          # the other 3 edges are present
                others = color_count[idx]
                if c in others:                  # colour clash
                    return False
                if len(others) < 3:              # the 3 others are not all distinct
                    return False
        return True

    def find_color(e, edge_colors, prefer=0.6):
        """Pick a valid colour for e, preferring less-conflicting colours (soft pressure)."""
        # colours appearing in any (nearly) complete cycle through e
        pressure = defaultdict(int)
        for idx in edge_to_cycles[e]:
            if present_count[idx] >= 3:
                for c, k in color_count[idx].items():
                    pressure[c] += k
        valid = []
        for c in range(num_colors):
            if valid_color(e, c, edge_colors):
                valid.append((pressure.get(c, 0), c))
        if not valid:
            return None
        valid.sort()
        if valid[0][0] == 0:                     # a fully free colour exists
            free = [c for p, c in valid if p == 0]
            return random.choice(free)
        top_k = min(3, len(valid))
        _, c = random.choice(valid[:top_k])
        return c

    # ---- Phase 1: dense random init + repair (fast) ----
    edge_colors = {}
    for e in all_edges:
        if random.random() < 0.85:
            c = random.randint(0, num_colors - 1)
            edge_colors[e] = c
            add_edge(e, c)

    # Repair: drop edges that sit in a non-rainbow 4-cycle until clean.
    for _ in range(4000):
        if time.time() > deadline - 60.0:
            break
        bad = None
        for idx, pc in enumerate(present_count):
            if pc == 4 and len(color_count[idx]) < 4:
                bad = idx
                break
        if bad is None:
            break
        # remove the edge in this cycle with the most incident cycles (biggest relief)
        e_rm = max(all_cycles[bad], key=lambda e: len(edge_to_cycles[e]))
        c_rm = edge_colors[e_rm]
        del edge_colors[e_rm]
        remove_edge(e_rm, c_rm)

    best_count = len(edge_colors)
    best_state = dict(edge_colors)
    missing_set = set(e for e in all_edges if e not in edge_colors)

    def save_if_better():
        nonlocal best_count, best_state
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_state = dict(edge_colors)

    # ---- Phase 2: Lagrangian-guided local search ----
    lam_color = [0.0] * num_colors
    lag_interval = 200
    it = 0
    while time.time() < deadline:
        it += 1

        # Periodic Lagrangian subgradient update (cheap: single pass over cycles)
        if it % lag_interval == 0:
            tight = [0] * num_colors
            for idx, pc in enumerate(present_count):
                if pc == 4:
                    for c in color_count[idx]:
                        tight[c] += 1
            target = len(all_cycles) / num_colors
            for c in range(num_colors):
                lam_color[c] = max(0.0, min(5.0, lam_color[c] + 0.1 * (tight[c] - target)))

        # Move 1: add a missing edge (sample & pick the most "free" candidate)
        if missing_set:
            sample = random.sample(list(missing_set), min(20, len(missing_set)))
            cand = []
            for e in sample:
                c = find_color(e, edge_colors)
                if c is not None:
                    # bonus for edges in few tight cycles (more flexible)
                    tight_cnt = sum(1 for i in edge_to_cycles[e] if present_count[i] == 3)
                    cand.append((tight_cnt, e, c))
            if cand:
                cand.sort()
                _, e, c = cand[0]
                edge_colors[e] = c
                add_edge(e, c)
                missing_set.discard(e)
                save_if_better()
                continue

        # Move 2: recolor an edge (free up a colour for a missing edge)
        if edge_colors and random.random() < 0.5:
            e_rc = random.choice(list(edge_colors.keys()))
            old_c = edge_colors[e_rc]
            new_c = find_color(e_rc, edge_colors, prefer=0.2)
            if new_c is not None and new_c != old_c:
                remove_edge(e_rc, old_c)
                edge_colors[e_rc] = new_c
                add_edge(e_rc, new_c)

        # Move 3: swap (remove one, add another)
        if edge_colors and missing_set and random.random() < 0.3:
            e_rm = random.choice(list(edge_colors.keys()))
            e_add = random.choice(list(missing_set))
            c_rm = edge_colors[e_rm]
            del edge_colors[e_rm]
            remove_edge(e_rm, c_rm)
            missing_set.add(e_rm)
            c_add = find_color(e_add, edge_colors)
            if c_add is not None:
                edge_colors[e_add] = c_add
                add_edge(e_add, c_add)
                missing_set.discard(e_add)
                save_if_better()
                c_back = find_color(e_rm, edge_colors)
                if c_back is not None:
                    edge_colors[e_rm] = c_back
                    add_edge(e_rm, c_back)
                    missing_set.discard(e_rm)
                    save_if_better()
            else:
                c_back = find_color(e_rm, edge_colors)
                if c_back is not None:
                    edge_colors[e_rm] = c_back
                    add_edge(e_rm, c_back)
                    missing_set.discard(e_rm)
                else:
                    edge_colors[e_rm] = c_rm
                    add_edge(e_rm, c_rm)
                    missing_set.discard(e_rm)

        # Move 4: double swap (remove 2, add 2)
        if len(edge_colors) >= 2 and len(missing_set) >= 2 and random.random() < 0.15:
            e1, e2 = random.sample(list(edge_colors.keys()), 2)
            e3, e4 = random.sample(list(missing_set), 2)
            c1, c2 = edge_colors[e1], edge_colors[e2]
            del edge_colors[e1]; del edge_colors[e2]
            remove_edge(e1, c1); remove_edge(e2, c2)
            missing_set.add(e1); missing_set.add(e2)
            c3 = find_color(e3, edge_colors)
            c4 = find_color(e4, edge_colors) if c3 is not None else None
            if c3 is not None and c4 is not None:
                edge_colors[e3] = c3; edge_colors[e4] = c4
                add_edge(e3, c3); add_edge(e4, c4)
                missing_set.discard(e3); missing_set.discard(e4)
                save_if_better()
                for e, _ in ((e1, c1), (e2, c2)):
                    cb = find_color(e, edge_colors)
                    if cb is not None:
                        edge_colors[e] = cb
                        add_edge(e, cb)
                        missing_set.discard(e)
                        save_if_better()
            # restore removed edges
            for e, c in ((e1, c1), (e2, c2)):
                if e not in edge_colors:
                    edge_colors[e] = c
                    add_edge(e, c)
                    missing_set.discard(e)

    # ---- Restore best, then high-precision final polish ----
    # Rebuild bookkeeping from best_state (cheap: O(edges * incidences))
    for e in edge_colors:
        del edge_colors[e]
    for idx in range(len(all_cycles)):
        present_count[idx] = 0
        color_count[idx].clear()
    for e, c in best_state.items():
        edge_colors[e] = c
        add_edge(e, c)
    missing_set = set(e for e in all_edges if e not in edge_colors)

    # Exhaustive sweep: keep trying every missing edge until none fit.
    progress = True
    while progress and time.time() < deadline:
        progress = False
        for e in list(missing_set):
            c = find_color(e, edge_colors)
            if c is not None:
                edge_colors[e] = c
                add_edge(e, c)
                missing_set.discard(e)
                progress = True
                save_if_better()

    result = [[u, v, c] for (u, v), c in edge_colors.items()]
    return result
