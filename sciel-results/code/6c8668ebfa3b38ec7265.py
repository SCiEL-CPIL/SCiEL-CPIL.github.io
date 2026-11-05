"""Strong edge colouring via conflict-graph + maximal induced matching + merge."""
import time

DEADLINE = 140.0  # safety margin under 150s kill

def construct():
    start = time.time()
    from random import Random

    def instance(n, d, seed):
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

    def build_conflict_graph(edges):
        # conflict[i][j] = True if edges i,j cannot share a colour
        m = len(edges)
        # adjacency of original graph
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        # map vertex -> list of incident edge indices
        incident = {}
        for i, (u, v) in enumerate(edges):
            incident.setdefault(u, []).append(i)
            incident.setdefault(v, []).append(i)
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            # edges sharing u or v
            for j in incident[u]:
                if j != i:
                    conflict[i].add(j)
            for j in incident[v]:
                if j != i:
                    conflict[i].add(j)
            # edges joined by an edge: for each neighbour w of u (w != v),
            #   edges (w, x) conflict with (u,v)
            for w in adj.get(u, ()):
                if w == v:
                    continue
                for j in incident.get(w, ()):
                    if j != i:
                        conflict[i].add(j)
            for w in adj.get(v, ()):
                if w == u:
                    continue
                for j in incident.get(w, ()):
                    if j != i:
                        conflict[i].add(j)
        return conflict

    def greedy_max_induced_matching(edges, conflict, available):
        # available: set of edge indices not yet coloured
        # return a maximal independent set (induced matching) of available
        chosen = []
        chosen_set = set()
        # order by degree in conflict (within available) for better packing
        order = sorted(available, key=lambda i: len(conflict[i] & available))
        for i in order:
            if i not in available:
                continue
            if not (conflict[i] & chosen_set):
                chosen.append(i)
                chosen_set.add(i)
                # remove its neighbours from available
                for j in conflict[i]:
                    available.discard(j)
        return chosen

    def color_one_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        conflict = build_conflict_graph(edges)

        colour = [-1] * m
        remaining = set(range(m))
        next_colour = 0
        while remaining:
            if time.time() > DEADLINE:
                break
            cm = greedy_max_induced_matching(edges, conflict, set(remaining))
            if not cm:
                # fallback: single edge
                cm = [next(iter(remaining))]
            for i in cm:
                colour[i] = next_colour
                remaining.discard(i)
            next_colour += 1
        # ensure all coloured (should be)
        for i in range(m):
            if colour[i] < 0:
                colour[i] = next_colour
                next_colour += 1

        # Merge pass: try to merge colour classes to reduce count.
        # Re-label to 0..k-1 first
        present = sorted(set(colour))
        remap = {c: i for i, c in enumerate(present)}
        colour = [remap[c] for c in colour]
        k = len(present)

        # Build class lists
        classes = [[] for _ in range(k)]
        for i, c in enumerate(colour):
            classes[c].append(i)

        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for a in range(k):
                if not classes[a]:
                    continue
                for b in range(a + 1, k):
                    if not classes[b]:
                        continue
                    # can we merge a and b? i.e., no conflict between any edge in a and any in b
                    ok = True
                    for i in classes[a]:
                        if conflict[i] & set(classes[b]):
                            ok = False
                            break
                    if not ok:
                        continue
                    # merge b into a
                    for i in classes[b]:
                        colour[i] = a
                    classes[a].extend(classes[b])
                    classes[b] = []
                    changed = True
                    break
                if changed:
                    break
            # compact labels
            present = sorted(set(colour))
            if len(present) < k:
                remap = {c: i for i, c in enumerate(present)}
                colour = [remap[c] for c in colour]
                k = len(present)
                classes = [[] for _ in range(k)]
                for i, c in enumerate(colour):
                    classes[c].append(i)
        return colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 5:
            # not enough time; do a quick greedy first-fit for remaining
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
            continue
        out.append(color_one_instance(*spec))
    return out
