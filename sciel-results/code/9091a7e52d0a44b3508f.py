"""Dense edge-coloured graph on 20 vertices, every 4-cycle rainbow.
Greedy randomized construction with efficient 4-cycle validity checks."""
import time
import random

N = 20
C = 20
DEADLINE = 142.0  # safety margin under 150s limit


def _validate(edges):
    """Return True iff every 4-cycle is rainbow (4 distinct colours)."""
    adj = {}
    for u, v, c in edges:
        adj.setdefault((u, v), c)
        adj.setdefault((v, u), c)

    def has_edge(a, b):
        return (a, b) in adj

    # Build neighbour lists
    neigh = {i: [] for i in range(N)}
    for u, v, c in edges:
        neigh[u].append(v)
        neigh[v].append(u)

    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            # common neighbours of u and v
            common = [w for w in neigh[u] if has_edge(v, w) and w != u]
            for x in common:
                for y in common:
                    if x >= y:
                        continue
                    # 4-cycle u - x - v - y - u
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
    return True


def _check_add(adj, edge_set, u, v, c):
    """Check whether adding edge (u,v) with colour c keeps every 4-cycle rainbow.
    adj: dict (a,b)->c for ordered pairs. edge_set: set of frozenset pairs."""
    # Common neighbours of u and v (excluding u,v themselves)
    nu = set(adj.keys())
    # neighbours of u
    ns_u = set()
    for (a, b) in adj:
        if a == u:
            ns_u.add(b)
        elif b == u:
            ns_u.add(a)
    ns_v = set()
    for (a, b) in adj:
        if a == v:
            ns_v.add(b)
        elif b == v:
            ns_v.add(a)
    common = ns_u & ns_v
    common.discard(u)
    common.discard(v)
    for x in common:
        for y in common:
            if x >= y:
                continue
            # 4-cycle u - x - v - y - u
            cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
            if len(cols) != 4:
                return False
    # Also the new edge must not itself create a 4-cycle where two edges share colour c
    # i.e. a 4-cycle containing the new edge (u,v): u - x - v - y - u uses new edge (u,v)
    # Wait, that's already covered? No. The new edge (u,v) participates in cycles u-x-v-y-u
    # where the path u-x-v uses new edge? Let's handle: cycles that include edge (u,v).
    # A 4-cycle containing (u,v): u - v - x - y - u, with x adjacent to v, y adjacent to u,
    # x != u, y != v, and x-y edge exists, x != y, and x,y not creating u/v repeats.
    # Edges: (u,v)=c, (v,x), (x,y), (y,u). Need all 4 distinct.
    for x in ns_v:
        if x == u:
            continue
        for y in ns_u:
            if y == v or y == x:
                continue
            if not has_edge_pair(adj, x, y):
                continue
            cols = {c, adj[(v, x)], adj[(x, y)], adj[(y, u)]}
            if len(cols) != 4:
                return False
    return True


def has_edge_pair(adj, a, b):
    return (a, b) in adj or (b, a) in adj


def _make_adj(edges):
    adj = {}
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
    return adj


def construct():
    start = time.time()
    rng = random.Random(12345)

    best_edges = None
    best_len = 0

    # Candidate edge list (all unordered pairs)
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    def try_greedy(order_seed, time_budget):
        nonlocal best_edges, best_len
        r = random.Random(order_seed)
        order = all_pairs[:]
        r.shuffle(order)

        edges = []
        adj = {}
        # Maintain neighbour sets for speed
        neigh = [set() for _ in range(N)]
        color_of = {}  # (u,v) with u<v -> c

        def add_edge(u, v, c):
            edges.append([u, v, c])
            adj[(u, v)] = c
            adj[(v, u)] = c
            neigh[u].add(v)
            neigh[v].add(u)
            color_of[(u, v) if u < v else (v, u)] = c

        def can_add(u, v, c):
            # common neighbours of u and v
            common = neigh[u] & neigh[v]
            common.discard(u)
            common.discard(v)
            common_list = list(common)
            # cycles not using the new edge: u - x - v - y - u where x,y common neighbours
            for i in range(len(common_list)):
                x = common_list[i]
                for y in common_list[i + 1:]:
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
            # cycles using the new edge (u,v): u - v - x - y - u
            # x in neigh[v] (x != u), y in neigh[u] (y != v), x-y edge, x != y
            for x in neigh[v]:
                if x == u:
                    continue
                for y in neigh[u]:
                    if y == v or y == x:
                        continue
                    if x < y:
                        key = (x, y)
                    else:
                        key = (y, x)
                    if key not in color_of:
                        continue
                    cols = {c, adj[(v, x)], adj[(x, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
            return True

        for (u, v) in order:
            if time.time() > time_budget:
                break
            # try colours in random order
            cols = list(range(C))
            r.shuffle(cols)
            placed = False
            for c in cols:
                if can_add(u, v, c):
                    add_edge(u, v, c)
                    placed = True
                    break
            # if no colour works, skip this edge
        return edges

    # Run multiple greedy attempts with different seeds, tracking best
    seed_counter = 1000
    while time.time() < DEADLINE:
        budget = min(time.time() + 8.0, DEADLINE)
        edges = try_greedy(seed_counter, budget)
        seed_counter += 1
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges
        # early exit if we reached a very dense graph
        if best_len >= 150:
            break

    if best_edges is None:
        # fallback: star
        return [[0, v, 0] for v in range(1, N)]
    return best_edges


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
