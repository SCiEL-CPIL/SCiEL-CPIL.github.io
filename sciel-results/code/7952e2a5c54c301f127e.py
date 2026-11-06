import time

_N = 20
_EDGE = _N * (_N - 1) // 2  # 190
_DEADLINE = 142.0


def _build():
    # Edge index mapping
    idx = {}
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            idx[(u, v)] = len(edges)
            edges.append((u, v))
    for (u, v), i in idx.items():
        if (v, u) not in idx:
            idx[(v, u)] = i

    # conflict adjacency: edge e conflicts with edge f iff they share a vertex
    # and have 2 other common neighbors (adding both makes a K2,3)
    adj = [set() for _ in range(_EDGE)]
    for i in range(_EDGE):
        u, v = edges[i]
        for j in range(i + 1, _EDGE):
            a, b = edges[j]
            if (a == u and b == v) or (a == v and b == u):
                continue
            if a == u or a == v or b == u or b == v:
                # exactly one shared vertex
                if a == u or a == v:
                    w = a
                else:
                    w = b
                # common neighbors of the other pair
                other1 = v if w == u else u
                other2 = b if w == a else a
                cn = 0
                for x in range(_N):
                    if x == w or x == other1 or x == other2:
                        continue
                    if (w, x) in idx and (other1, x) in idx and (other2, x) in idx:
                        cn += 1
                if cn >= 2:
                    adj[i].add(j)
                    adj[j].add(i)
    adj = [sorted(s) for s in adj]
    return edges, adj


_EDGES, _ADJ = _build()
_ADJ_SETS = [set(a) for a in _ADJ]
_NEIGHBOR_COUNTS = [len(a) for a in _ADJ]


def _greedy_lb():
    """Randomized greedy maximal independent set -> edge count."""
    avail = set(range(_EDGE))
    chosen = []
    while avail:
        # pick a random vertex, add it, remove it and its neighbors
        v = next(iter(avail))
        chosen.append(v)
        avail.discard(v)
        for w in _ADJ[v]:
            avail.discard(w)
    return len(chosen)


def _tomita(adj, edges, adj_sets, n):
    """Exact MCIS via Tomita branch-and-bound with greedy coloring bound."""
    best = [0]
    t0 = time.time()

    def deadline():
        return (time.time() - t0) > _DEADLINE

    def color_sort(cand):
        # greedy coloring of the induced subgraph on cand; returns (ordered, colors)
        uncolored = list(cand)
        color_class = []
        order = []
        colors = {}
        while uncolored:
            cc = []
            uncolored2 = []
            for v in uncolored:
                if all(v not in adj_sets[u] for u in cc):
                    cc.append(v)
                else:
                    uncolored2.append(v)
            color_class.append(cc)
            uncolored = uncolored2
        # order vertices by increasing color class
        for c in range(len(color_class) - 1, -1, -1):
            for v in color_class[c]:
                order.append(v)
                colors[v] = c + 1
        return order, colors, len(color_class)

    def search(cand, chosen_count):
        if deadline():
            return
        if not cand:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        order, colors, ncolor = color_sort(cand)
        # upper bound
        if chosen_count + ncolor <= best[0]:
            return
        # branch on vertices in order
        i = 0
        while i < len(order):
            v = order[i]
            if colors[v] + chosen_count <= best[0]:
                break
            # take v
            new_cand = [x for x in order[:i] if x in adj_sets[v]]
            search(new_cand, chosen_count + 1)
            i += 1

    full = list(range(_EDGE))
    order, colors, ncolor = color_sort(full)
    search(order, 0)
    return best[0]


def construct():
    t0 = time.time()
    # strong lower bound from many random greedy trials
    lb = 0
    for _ in range(400):
        if (time.time() - t0) > _DEADLINE - 30.0:
            break
        val = _greedy_lb()
        if val > lb:
            lb = val
    # exact solve if we have time; else return best heuristic
    if (time.time() - t0) < _DEADLINE - 5.0:
        try:
            exact = _tomita(_ADJ, _EDGES, _ADJ_SETS, _EDGE)
            if exact > lb:
                lb = exact
        except Exception:
            pass
    # build edge list from a greedy independent set of size lb (recompute to be safe)
    # find a concrete independent set of size lb via greedy, fallback to best found
    # re-run greedy to get an actual set
    best_set = None
    for _ in range(200):
        if (time.time() - t0) > _DEADLINE - 2.0:
            break
        avail = set(range(_EDGE))
        chosen = []
        while avail:
            v = next(iter(avail))
            chosen.append(v)
            avail.discard(v)
            for w in _ADJ[v]:
                avail.discard(w)
        if best_set is None or len(chosen) > len(best_set):
            best_set = chosen
    result = [[_EDGES[i][0], _EDGES[i][1]] for i in best_set]
    return result
