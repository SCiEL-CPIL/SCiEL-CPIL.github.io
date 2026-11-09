import time
import random
import itertools

# EVOLVE-BLOCK-START

def _has_c4_for_edge(n, adj, u, v):
    """Return True if adding edge (u,v) would create a C4.
    A C4 is created iff u and v already have >=2 common neighbours."""
    cu = adj[u]
    cv = adj[v]
    common = 0
    # iterate over smaller set
    if len(cu) < len(cv):
        for x in cu:
            if x in cv:
                common += 1
                if common >= 2:
                    return True
    else:
        for x in cv:
            if x in cu:
                common += 1
                if common >= 2:
                    return True
    return False


def _independent_number_greedy(n, adj, k, rng, trials=200):
    """Try to find an independent set of size k.
    Returns True if found (i.e. alpha >= k)."""
    for _ in range(trials):
        # random greedy: repeatedly pick a random vertex, remove its neighbours
        remaining = set(range(n))
        chosen = []
        while remaining:
            v = rng.choice(sorted(remaining))
            chosen.append(v)
            remaining.discard(v)
            # remove neighbours of v from remaining
            for w in adj[v]:
                remaining.discard(w)
            if len(chosen) >= k:
                return True
        # if we could fill k, we already returned; otherwise this trial failed
        if len(chosen) >= k:
            return True
    return False


def _alpha_is_less_than_k(n, adj, k, rng, trials=100):
    """Return True if we believe alpha < k (i.e. no independent k-set found)."""
    return not _independent_number_greedy(n, adj, k, rng, trials)


def _build_random(n, target_edges, rng, deadline):
    """Randomly add edges while keeping C4-free, aiming for target_edges."""
    adj = [set() for _ in range(n)]
    edges = []
    # candidate edge list
    candidates = list(itertools.combinations(range(n), 2))
    rng.shuffle(candidates)
    added = 0
    for (u, v) in candidates:
        if time.time() > deadline:
            break
        if added >= target_edges:
            break
        if v in adj[u]:
            continue
        if _has_c4_for_edge(n, adj, u, v):
            continue
        adj[u].add(v)
        adj[v].add(u)
        edges.append([u, v])
        added += 1
    return adj, edges


def _try_improve(n, adj, edges, k, rng, deadline):
    """Try to add more edges (keeping C4-free) to reduce independence number.
    Returns new (adj, edges) if improved (alpha<k and more edges), else None."""
    best_adj, best_edges = adj, edges
    for _ in range(2000):
        if time.time() > deadline:
            break
        # pick a random non-edge
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v or v in adj[u]:
            continue
        if _has_c4_for_edge(n, adj, u, v):
            continue
        # tentatively add
        adj[u].add(v)
        adj[v].add(u)
        edges.append([u, v])
        # check alpha
        if _alpha_is_less_than_k(n, adj, k, rng, trials=60):
            best_adj, best_edges = [s.copy() for s in adj], list(edges)
        else:
            # revert
            adj[u].discard(v)
            adj[v].discard(u)
            edges.pop()
    return best_adj, best_edges


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s
    rng = random.Random(12345)

    # We want alpha < 11, i.e. alpha <= 10. So k = 11 means "no independent 11-set".
    k = 11

    # Try increasing n from a high target downward until we find a valid graph.
    best = None
    # Start from a decent target; the best known is 39.
    for n in [39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
        # For a C4-free graph, max edges ~ n*sqrt(n)/2. We'll aim for a dense target.
        # Use a heuristic target.
        target_edges = int(n * (n ** 0.5) / 2)
        attempts = 0
        found = False
        while time.time() < deadline and attempts < 40 and not found:
            attempts += 1
            adj, edges = _build_random(n, target_edges, rng, deadline)
            if len(edges) < target_edges * 0.4:
                continue
            # Check alpha < 11
            if _alpha_is_less_than_k(n, adj, k, rng, trials=150):
                # try to improve
                adj2, edges2 = _try_improve(n, adj, edges, k, rng, deadline)
                if _alpha_is_less_than_k(n, adj2, k, rng, trials=150):
                    best = {'n': n, 'edges': edges2}
                    found = True
                    break
        if found:
            break

    if best is None:
        # Fallback: the seed construction (10 disjoint triangles), n=30, alpha=10.
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        best = {'n': 30, 'edges': edges}

    return best

# EVOLVE-BLOCK-END
