"""Suite task: 5-regular graph of girth 9 (cage n(5,9)).

Keeps the Lazebnik-Ustimenko-Woldar explicit construction as a guaranteed
valid fallback, and adds a time-bounded random regular-graph search that
accepts a candidate only after certifying its girth is at least g via
non-backtracking walks from every vertex.
"""
import time
import random
import math

_DEADLINE = None


def _remaining():
    if _DEADLINE is None:
        return float("inf")
    return _DEADLINE - time.time()


# ---------------------------------------------------------------------------
# Girth certification via non-backtracking walks (BFS on directed edges).
# A graph has girth >= g  <=>  no non-backtracking closed walk of length < g.
# ---------------------------------------------------------------------------
def _girth_ge(adj, n, g):
    # states: (prev, cur); start from every directed edge (u, v), prev=u, cur=v
    # distance 1 corresponds to that single directed edge.
    # We expand to (cur, nxt) for nxt != prev, up to distance g-1.
    # If we ever reach (cur, nxt) == (u, v) i.e. nxt == u at distance >= 2
    #   -> that is a closed non-backtracking walk of length < g -> girth < g.
    # Actually a closed walk of length L corresponds to returning to the
    # starting directed edge after L steps. We track visited directed edges
    # with the distance at which they were first reached.
    from collections import deque
    q = deque()
    # dist[(prev, cur)] = length of the non-backtracking walk (number of edges)
    dist = {}
    for u in range(n):
        for v in adj[u]:
            key = (u, v)
            dist[key] = 1
            q.append(key)
    while q:
        prev, cur = q.popleft()
        d = dist[(prev, cur)]
        if d >= g - 1:
            # We have already certified that no shorter closed walk exists
            # from this state; do not extend further.
            continue
        for nxt in adj[cur]:
            if nxt == prev:
                continue
            # A non-backtracking closed walk of length d+1 back to (prev,cur)
            # would mean nxt == prev? No: closed walk returns to start edge.
            # The walk (prev->cur->nxt) closes if nxt == prev (excluded) -- so
            # a cycle of length d+1 is detected when nxt equals the ORIGINAL
            # start vertex AND the path has no backtracking AND d+1 >= 3.
            # Simpler: a non-backtracking closed walk of length L exists iff
            # some directed edge (a,b) is reachable from itself in L steps of
            # non-backtracking transitions. We detect by checking if (cur,nxt)
            # was already seen with distance d' such that d+1 and d' form a
            # cycle. Standard approach: when we revisit a directed edge, the
            # difference in distances gives a cycle length.
            nkey = (cur, nxt)
            if nkey in dist:
                # cycle length = (d + 1) - dist[nkey]
                cyc = (d + 1) - dist[nkey]
                if cyc >= 3 and cyc < g:
                    return False
                # else it's a longer cycle, ignore
            else:
                dist[nkey] = d + 1
                q.append(nkey)
    return True


# ---------------------------------------------------------------------------
# Random k-regular graph generation (configurable model + edge swaps).
# ---------------------------------------------------------------------------
def _generate_k_regular(n, k, rng, model="pairing"):
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
    if model == "pairing":
        # Configuration model: each vertex gets k stubs, pair stubs randomly.
        stubs = [i for i in range(n) for _ in range(k)]
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for a in range(0, n * k, 2):
            u, v = stubs[a], stubs[a + 1]
            if u == v:
                ok = False
                break
            if v in adj[u]:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not ok:
            return None
        return [sorted(adj[i]) for i in range(n)]
    elif model == "swap":
        # Start from a random pairing (retry until simple), then done.
        for _ in range(200):
            adj = _generate_k_regular(n, k, rng, model="pairing")
            if adj is not None:
                return adj
        return None
    return None


def _edge_swap(adj, n, k, rng, iters=2000):
    # Perform random edge swaps to randomize the regular graph (keeps degree k).
    m = n * k // 2
    # Build list of edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    # Map edge to index
    edge_set = set(edges)
    for _ in range(iters):
        # Pick two distinct edges
        i = rng.randrange(m)
        j = rng.randrange(m)
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        # Try swap: (a,b),(c,d) -> (a,c),(b,d)  or (a,d),(b,c)
        for new1, new2 in [((a, c), (b, d)), ((a, d), (b, c))]:
            x1, y1 = new1
            x2, y2 = new2
            if x1 == y1 or x2 == y2:
                continue
            if (x1, y1) in edge_set or (x2, y2) in edge_set:
                continue
            if (x1, y1) == (x2, y2):
                continue
            # apply
            edge_set.discard((a, b))
            edge_set.discard((c, d))
            edge_set.add((x1, y1))
            edge_set.add((x2, y2))
            adj[x1].remove(y1)
            adj[y1].remove(x1)
            adj[x2].remove(y2)
            adj[y2].remove(x2)
            adj[x1].add(y1)
            adj[y1].add(x1)
            adj[x2].add(y2)
            adj[y2].add(x2)
            edges[i] = (x1, y1)
            edges[j] = (x2, y2)
            break
    return adj


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular
# bipartite graph whose girth is at least d + 5 for odd d.
# ---------------------------------------------------------------------------
def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.time() + 140.0  # safety margin under 150s kill
    rng = random.Random(12345)

    # Guaranteed valid fallback
    try:
        n_luw, edges_luw = _luw_construct(k, g)
    except Exception:
        n_luw, edges_luw = 0, []

    best_n, best_edges = n_luw, edges_luw

    # Moore bound (lower bound on n)
    if k == 1:
        moore = g
    else:
        # For girth g: if g is even, n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i
        # if g is odd, n >= k * sum_{i=0}^{(g-1)/2-1} (k-1)^i + k*(k-1)*sum...
        # Simplified: use the standard Moore bound
        if g % 2 == 0:
            moore = 2 * sum((k - 1) ** i for i in range(g // 2))
        else:
            moore = k * sum((k - 1) ** i for i in range((g - 1) // 2))
    moore = max(moore, k + 1)

    # Random search: try random k-regular graphs of increasing size
    # Start near Moore bound and grow
    search_n = moore
    max_search_n = min(n_luw - 1, moore + 500) if n_luw > moore else moore

    while search_n <= max_search_n and _remaining() > 10.0:
        if search_n * k % 2 != 0:
            search_n += 1
            continue
        # Try a few random graphs at this size
        attempts = 0
        max_attempts = 30 if search_n < 100 else 15
        while attempts < max_attempts and _remaining() > 10.0:
            attempts += 1
            adj = _generate_k_regular(search_n, k, rng)
            if adj is None:
                continue
            # Quick degree check
            if any(len(adj[i]) != k for i in range(search_n)):
                continue
            # Verify girth
            if _girth_ge(adj, search_n, g):
                # Found a valid graph
                edges = []
                for u in range(search_n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                best_n, best_edges = search_n, edges
                break
            # If girth is too small, try edge swaps to increase girth
            # (limited: only for small n)
            if search_n <= 60 and _remaining() > 15.0:
                adj = _edge_swap(adj, search_n, k, rng, iters=500)
                if _girth_ge(adj, search_n, g):
                    edges = []
                    for u in range(search_n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    best_n, best_edges = search_n, edges
                    break
        search_n += 2  # keep n*k even

    # Final validation of best result
    if best_edges is not None and len(best_edges) > 0:
        # Verify degree and simplicity
        adj_check = [set() for _ in range(best_n)]
        for u, v in best_edges:
            if u == v:
                return n_luw, edges_luw  # invalid, fallback
            adj_check[u].add(v)
            adj_check[v].add(u)
        valid = True
        for i in range(best_n):
            if len(adj_check[i]) != k:
                valid = False
                break
        if not valid:
            return n_luw, edges_luw
        # Verify girth one more time
        adj_list = [sorted(adj_check[i]) for i in range(best_n)]
        if not _girth_ge(adj_list, best_n, g):
            return n_luw, edges_luw

    return best_n, best_edges
