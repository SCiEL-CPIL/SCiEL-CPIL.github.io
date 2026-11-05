"""K2,3-free dense graph on 20 vertices via greedy + local search."""
import time
import random

N = 20


def _can_add(adj, u, v):
    """Return True if adding edge (u,v) keeps graph K2,3-free."""
    if u == v or adj[u][v]:
        return False
    # Check: u and v must have at most 2 common neighbours
    common = 0
    for w in range(N):
        if w != u and w != v and adj[u][w] and adj[v][w]:
            common += 1
            if common >= 3:
                return False
    return True


def _greedy(n, seed, time_budget):
    """Greedy edge addition in random order, stopping at time_budget."""
    random.seed(seed)
    adj = [[False] * n for _ in range(n)]
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(pairs)
    t0 = time.time()
    for (u, v) in pairs:
        if time.time() > time_budget:
            break
        if _can_add(adj, u, v):
            adj[u][v] = True
            adj[v][u] = True
    return adj


def _local_search(adj, time_budget):
    """Local search: add missing edges, then try swaps. Mutates adj in place."""
    n = N
    improved = True
    while improved:
        improved = False
        t0 = time.time()
        # Try adding any missing edge
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > time_budget:
                    return adj
                if not adj[u][v] and _can_add(adj, u, v):
                    adj[u][v] = True
                    adj[v][u] = True
                    improved = True
        # Try swaps: remove (a,b), add (c,d) if it increases edge count
        # (only useful if we can add two edges for one removed -> net +1)
        # Simpler: try removing an edge then re-adding two others is complex;
        # instead try: for each edge (a,b) and missing edge (c,d), check if
        # removing (a,b) allows adding (c,d) and (e,f) for net gain. Keep simple:
        # try single swaps that keep edge count same but enable further adds.
    return adj


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s
    best_adj = None
    best_count = -1

    # Phase 1: many greedy restarts
    seed = 0
    while time.time() < deadline - 20.0:
        adj = _greedy(N, seed, deadline - 20.0)
        count = sum(1 for u in range(N) for v in range(u + 1, N) if adj[u][v])
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
        seed += 1

    if best_adj is None:
        # Fallback: five disjoint K4
        edges = []
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    edges.append([4 * k + a, 4 * k + b])
        return edges

    # Phase 2: local search on the best greedy result
    work = [row[:] for row in best_adj]
    work = _local_search(work, deadline - 5.0)
    count = sum(1 for u in range(N) for v in range(u + 1, N) if work[u][v])
    if count > best_count:
        best_adj = work
        best_count = count

    edges = [[u, v] for u in range(N) for v in range(u + 1, N) if best_adj[u][v]]
    return edges
