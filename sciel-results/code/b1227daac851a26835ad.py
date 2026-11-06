"""Strong edge colouring: DSATUR greedy + local-search refinement."""
# EVOLVE-BLOCK-START
import time
from random import Random

_DEADLINE = 115.0

def _bounded_degree_graph(n, d, seed):
    rng = Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(8 * n * d):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        e = (u, v) if u < v else (v, u)
        if e in edges:
            continue
        if degree[u] < d and degree[v] < d:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)

def _build_conflict(edges, adj):
    m = len(edges)
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    conflict = [0] * m
    for i, (u, v) in enumerate(edges):
        mask = 0
        for w in adj.get(u, ()):
            j = edge_index.get((u, w))
            if j is not None and j != i:
                mask |= (1 << j)
        for w in adj.get(v, ()):
            j = edge_index.get((v, w))
            if j is not None and j != i:
                mask |= (1 << j)
        for x in adj.get(v, ()):
            j = edge_index.get((u, x))
            if j is not None and j != i:
                mask |= (1 << j)
        for x in adj.get(u, ()):
            j = edge_index.get((v, x))
            if j is not None and j != i:
                mask |= (1 << j)
        conflict[i] = mask
    return conflict

def _dsatur_fast(edges, conflict, adj, rng):
    m = len(edges)
    color = [-1] * m
    color_mask = [0] * m
    ncolors = 0
    deg = [len(adj.get(i, ())) for i in range(len(adj))]
    edge_deg = [deg[u] + deg[v] for u, v in edges]

    uncolored = set(range(m))
    sat = [0] * m

    while uncolored:
        best = -1
        best_sat = -1
        best_deg = -1
        for i in uncolored:
            s = bin(color_mask[i]).count('1')
            if s > best_sat or (s == best_sat and edge_deg[i] > best_deg):
                best = i
                best_sat = s
                best_deg = edge_deg[i]
        c = 0
        while (color_mask[best] >> c) & 1:
            c += 1
        color[best] = c
        if c + 1 > ncolors:
            ncolors = c + 1
        uncolored.discard(best)
        cm = conflict[best]
        x = cm
        while x:
            lb = x & (-x)
            j = lb.bit_length() - 1
            color_mask[j] |= (1 << c)
            x -= lb
    return color, ncolors

def _local_search(color, conflict, m, rng, time_budget):
    deadline = time.time() + time_budget
    # Build colour -> set of edges
    def build_cmap(color):
        cmap = {}
        for i, c in enumerate(color):
            cmap.setdefault(c, set()).add(i)
        return cmap

    cmap = build_cmap(color)
    ncolors = max(color) + 1
    while ncolors > 1 and time.time() < deadline:
        target = ncolors - 1
        if target not in cmap or not cmap[target]:
            ncolors -= 1
            continue
        edges_in = list(cmap[target])
        rng.shuffle(edges_in)
        moved_any = False
        for e in edges_in:
            if time.time() > deadline:
                break
            moved = False
            for c in range(target):
                if c not in cmap or not cmap[c]:
                    # check if e can take c: no conflict with any edge of colour c
                    # but cmap[c] is empty so it's fine
                    # remove e from target, add to c
                    cmap[target].discard(e)
                    color[e] = c
                    cmap.setdefault(c, set()).add(e)
                    moved = True
                    break
                # check conflict
                cx = conflict[e]
                ok = True
                for k in cmap[c]:
                    if (cx >> k) & 1:
                        ok = False
                        break
                if ok:
                    cmap[target].discard(e)
                    color[e] = c
                    cmap[c].add(e)
                    moved = True
                    break
            if moved:
                moved_any = True
        if not moved_any:
            break
        # rebuild cmap for correctness
        cmap = build_cmap(color)
        ncolors = max(color) + 1
    return color, ncolors

def construct():
    start = time.time()
    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    budgets = (25.0, 30.0, 40.0)
    for idx, (n, d, seed) in enumerate(specs):
        elapsed = time.time() - start
        remaining_total = _DEADLINE - elapsed
        if remaining_total < 8.0:
            # quick greedy
            edges = _bounded_degree_graph(n, d, seed)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            m = len(edges)
            color = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj.get(x, set())}
                taken = {c for f, c in color.items() if set(f) & near}
                color[e] = next(c for c in range(m) if c not in taken)
            out.append([color[e] for e in edges])
            continue
        edges = _bounded_degree_graph(n, d, seed)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        m = len(edges)
        conflict = _build_conflict(edges, adj)
        rng = Random(seed + 7)
        best_color = None
        best_nc = m + 1
        # Run DSATUR multiple times
        max_trials = 8
        for trial in range(max_trials):
            if time.time() - start > _DEADLINE - 10.0:
                break
            c, nc = _dsatur_fast(edges, conflict, adj, rng)
            if nc < best_nc:
                best_nc = nc
                best_color = c[:]
        if best_color is None:
            c, nc = _dsatur_fast(edges, conflict, adj, rng)
            best_color = c
            best_nc = nc
        # Local search
        ls_budget = min(budgets[idx] - (time.time() - start), _DEADLINE - (time.time() - start) - 8.0)
        if ls_budget > 0.5:
            best_color, best_nc = _local_search(best_color, conflict, m, rng, ls_budget)
        out.append(best_color[:])
    return out
# EVOLVE-BLOCK-END
