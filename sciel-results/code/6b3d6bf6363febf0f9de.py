"""Strong edge colouring: bitmask-accelerated greedy + local colour compression."""
import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under the 150 s kill

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
        # share a vertex
        for u, lst in v2e.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    conflict[lst[a]] |= (1 << lst[b])
                    conflict[lst[b]] |= (1 << lst[a])
        # joined by an edge
        for i, (u, v) in enumerate(edges):
            lu = v2e.get(u, [])
            lv = v2e.get(v, [])
            for a in lu:
                if a == i:
                    continue
                for b in lv:
                    if b == i or b == a:
                        continue
                    conflict[a] |= (1 << b)
                    conflict[b] |= (1 << a)
        return conflict

    def greedy_first_fit(edges, conflict, rng, order=None, adj=None):
        m = len(edges)
        if order is None:
            order = list(range(m))
        colour = [-1] * m
        used = 0
        col_forbidden = []  # list of sets: vertices blocked for this colour
        for e in order:
            u, v = edges[e]
            placed = False
            for c in range(used):
                if u not in col_forbidden[c] and v not in col_forbidden[c]:
                    colour[e] = c
                    col_forbidden[c].add(u)
                    col_forbidden[c].add(v)
                    for nb in adj.get(u, ()):
                        col_forbidden[c].add(nb)
                    for nb in adj.get(v, ()):
                        col_forbidden[c].add(nb)
                    placed = True
                    break
            if not placed:
                colour[e] = used
                s = set()
                s.add(u)
                s.add(v)
                for nb in adj.get(u, ()):
                    s.add(nb)
                for nb in adj.get(v, ()):
                    s.add(nb)
                col_forbidden.append(s)
                used += 1
        return colour, used

    def local_compress(edges, conflict, colour, used, rng, time_budget=5.0):
        """Local compression using bitmask conflict checks."""
        start = time.time()
        m = len(edges)
        col_of = colour[:]
        for _ in range(200):
            if time.time() - start > time_budget:
                break
            improved = False
            # build colour classes and their bitmask union
            classes = {}
            for e in range(m):
                c = col_of[e]
                if c < 0:
                    continue
                classes.setdefault(c, []).append(e)
            maxc = max(classes.keys()) if classes else 0
            # try merging class c into class d (c > d)
            for c in range(maxc, -1, -1):
                if c not in classes:
                    continue
                for d in range(c):
                    if d not in classes:
                        continue
                    # check no conflicts between classes[c] and classes[d]
                    ok = True
                    for e in classes[c]:
                        # if any edge in classes[d] conflicts with e
                        mask_d = 0
                        for f in classes[d]:
                            mask_d |= (1 << f)
                        if conflict[e] & mask_d:
                            ok = False
                            break
                    if ok:
                        for e in classes[c]:
                            col_of[e] = d
                        del classes[c]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # try moving single edges to lower colours
                for e in range(m):
                    ce = col_of[e]
                    for d in range(ce):
                        if d not in classes:
                            continue
                        mask_d = 0
                        for f in classes[d]:
                            mask_d |= (1 << f)
                        if not (conflict[e] & mask_d):
                            col_of[e] = d
                            classes[d].append(e)
                            classes[ce].remove(e)
                            if not classes[ce]:
                                del classes[ce]
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                # try 1-swap: move e to colour d, move f from d to colour ce
                for e in range(m):
                    ce = col_of[e]
                    for d in range(ce):
                        if d not in classes:
                            continue
                        for f in classes[d]:
                            # e to d: e must not conflict with any in classes[d] except f
                            mask_d_except_f = 0
                            for g in classes[d]:
                                if g != f:
                                    mask_d_except_f |= (1 << g)
                            if conflict[e] & mask_d_except_f:
                                continue
                            # f to ce: f must not conflict with any in classes[ce] except e
                            mask_ce_except_e = 0
                            for g in classes[ce]:
                                if g != e:
                                    mask_ce_except_e |= (1 << g)
                            if conflict[f] & mask_ce_except_e:
                                continue
                            # perform swap
                            col_of[e] = d
                            col_of[f] = ce
                            classes[d].remove(f)
                            classes[d].append(e)
                            classes[ce].remove(e)
                            classes[ce].append(f)
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        # renumber colours compactly
        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = gen_instance(*spec)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        conflict = build_conflict_bitmasks(edges)
        m = len(edges)
        best_colour = None
        best_used = m
        rng = Random(42)
        # multiple randomized restarts
        n_restarts = 120
        for r in range(n_restarts):
            if time.time() > DEADLINE - 10.0:
                break
            order = list(range(m))
            rng.shuffle(order)
            colour, used = greedy_first_fit(edges, conflict, rng, order, adj)
            if used < best_used:
                best_colour = colour[:]
                best_used = used
        # local compression on best
        if best_colour is not None:
            best_colour, best_used = local_compress(edges, conflict, best_colour, best_used, rng)
        # if we still have time, do a few more targeted restarts + compress
        if time.time() < DEADLINE - 15.0:
            for r in range(80):
                if time.time() > DEADLINE - 10.0:
                    break
                order = list(range(m))
                rng.shuffle(order)
                colour, used = greedy_first_fit(edges, conflict, rng, order, adj)
                if used <= best_used:
                    colour, used = local_compress(edges, conflict, colour, used, rng)
                    if used < best_used:
                        best_colour = colour[:]
                        best_used = used
        if best_colour is None:
            best_colour, best_used = greedy_first_fit(edges, conflict, rng, None, adj)
        out.append(best_colour)
    return out
# EVOLVE-BLOCK-END
