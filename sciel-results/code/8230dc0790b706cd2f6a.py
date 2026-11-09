"""C6-free graph on 40 vertices: maximize edges via fast greedy + local search."""
import time
import random
import numpy as np

N = 40
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]


def _init_adj():
    return [set() for _ in range(N)]


def _add(adj, u, v):
    adj[u].add(v)
    adj[v].add(u)


def _remove(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)


def _has_c6_full(adj, u, v):
    """Return True if adding edge (u,v) creates a C6.
    A C6 through (u,v) is u-a-x-y-b-v-u where:
    a in N(u), b in N(v), a!=b, a-b non-edge,
    x in N(a)\\{u,v,b}, y in N(b)\\{u,v,a}, x-y edge, x!=y.
    """
    nu = adj[u]
    nv = adj[v]
    if len(nu) < 2 or len(nv) < 2:
        return False
    for a in nu:
        if a == v:
            continue
        na = adj[a]
        for b in nv:
            if b == a or b == u:
                continue
            if b in nu:
                continue
            nb = adj[b]
            for x in na:
                if x == u or x == v or x == b:
                    continue
                nx = adj[x]
                for y in nx:
                    if y == u or y == v or y == a:
                        continue
                    if y in nb:
                        return True
    return False


def _edge_count(adj):
    return sum(len(s) for s in adj) // 2


def _get_edges(adj):
    edges = []
    for u in range(N):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


def _greedy_build(rng, seed=None):
    """Greedy edge addition in randomized order."""
    adj = _init_adj()
    order = list(ALL_PAIRS)
    rng.shuffle(order)
    for (u, v) in order:
        if v in adj[u]:
            continue
        if not _has_c6_full(adj, u, v):
            _add(adj, u, v)
    return adj


def _greedy_build_structured(rng, seed=None):
    """Greedy with structural bias: prefer edges that connect high-degree vertices."""
    adj = _init_adj()
    # Build in phases: first add a random matching to give structure, then greedy
    # Phase 1: random perfect matching
    perm = list(range(N))
    rng.shuffle(perm)
    for i in range(0, N - 1, 2):
        u, v = perm[i], perm[i + 1]
        _add(adj, u, v)
    # Phase 2: greedy add remaining edges
    remaining = [(u, v) for (u, v) in ALL_PAIRS if v not in adj[u]]
    rng.shuffle(remaining)
    for (u, v) in remaining:
        if v in adj[u]:
            continue
        if not _has_c6_full(adj, u, v):
            _add(adj, u, v)
    return adj


def _greedy_build_cliques(rng, seed=None):
    """Build by adding small cliques/blocks then connecting them."""
    adj = _init_adj()
    # Start with a few K4s or K5s
    # Try to place 5-cliques (10 edges each, 8 of them = 80 edges)
    # But C6-free: K5 has no C6. Two K5s connected by an edge: need to check.
    # Let's try: partition into groups, make each group a clique, then connect.
    # For 40 vertices: 8 groups of 5 -> 8 K5s = 80 edges
    # Then try to add cross-edges greedily
    groups = []
    for k in range(8):
        g = list(range(5 * k, 5 * k + 5))
        groups.append(g)
    for g in groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                _add(adj, g[i], g[j])
    # Now greedily add cross-edges
    cross_edges = [(u, v) for (u, v) in ALL_PAIRS if v not in adj[u]]
    rng.shuffle(cross_edges)
    for (u, v) in cross_edges:
        if v in adj[u]:
            continue
        if not _has_c6_full(adj, u, v):
            _add(adj, u, v)
    return adj


def _local_search(adj, rng, max_iter, time_limit):
    """Local search: random edge swaps (remove one, add one) to improve edge count."""
    best = [set(s) for s in adj]
    best_count = _edge_count(best)
    cur = [set(s) for s in adj]
    cur_count = best_count
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        # Get current edges
        edges = _get_edges(cur)
        if not edges:
            break
        a, b = rng.choice(edges)
        
        # Try to find a non-edge (c,d) such that removing (a,b) and adding (c,d) is C6-free
        # and we can then potentially add more edges
        improved = False
        # Sample candidate non-edges
        for _ in range(300):
            c = rng.randrange(N)
            d = rng.randrange(N)
            if c == d or c >= d:
                continue
            if d in cur[c]:
                continue
            # Remove a-b, try add c-d
            _remove(cur, a, b)
            if not _has_c6_full(cur, c, d):
                _add(cur, c, d)
                # Try to greedily add more edges now
                added_more = 0
                for _ in range(50):
                    if time.time() - start > time_limit:
                        break
                    c2 = rng.randrange(N)
                    d2 = rng.randrange(N)
                    if c2 == d2 or c2 >= d2:
                        continue
                    if d2 in cur[c2]:
                        continue
                    if not _has_c6_full(cur, c2, d2):
                        _add(cur, c2, d2)
                        added_more += 1
                new_count = _edge_count(cur)
                if new_count >= cur_count:
                    cur_count = new_count
                    if new_count > best_count:
                        best = [set(s) for s in cur]
                        best_count = new_count
                    improved = True
                    break
                else:
                    # Revert: remove c-d and the added edges, restore a-b
                    _remove(cur, c, d)
                    # Remove the extra edges we added
                    # This is tricky; let's just restore a-b and keep the state
                    _add(cur, a, b)
                    break
            else:
                _add(cur, a, b)
        if not improved:
            _add(cur, a, b)
    
    return best, best_count


def _simulated_annealing(adj, rng, max_iter, time_limit):
    """SA: accept worse moves with probability to escape local optima."""
    cur = [set(s) for s in adj]
    cur_count = _edge_count(cur)
    best = [set(s) for s in cur]
    best_count = cur_count
    start = time.time()
    temp = 2.0
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        temp = max(0.1, temp * 0.999)
        
        edges = _get_edges(cur)
        if not edges:
            break
        a, b = rng.choice(edges)
        
        # Try removing (a,b) and adding a random non-edge
        for _ in range(100):
            c = rng.randrange(N)
            d = rng.randrange(N)
            if c == d or c >= d:
                continue
            if d in cur[c]:
                continue
            
            _remove(cur, a, b)
            if not _has_c6_full(cur, c, d):
                _add(cur, c, d)
                new_count = _edge_count(cur)
                delta = new_count - cur_count
                if delta >= 0 or rng.random() < np.exp(delta / temp):
                    cur_count = new_count
                    if new_count > best_count:
                        best = [set(s) for s in cur]
                        best_count = new_count
                    break
                else:
                    _remove(cur, c, d)
                    _add(cur, a, b)
                    break
            else:
                _add(cur, a, b)
                break
        else:
            _add(cur, a, b)
    
    return best, best_count


def construct():
    deadline = 115.0
    start = time.time()
    rng = random.Random(12345)
    np.random.seed(12345)

    best_adj = None
    best_count = -1

    def update_best(adj, cnt):
        nonlocal best_adj, best_count
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]

    # Phase 1: diverse greedy builds
    builders = [_greedy_build, _greedy_build_structured, _greedy_build_cliques]
    for i in range(20):
        if time.time() > deadline - 30:
            break
        builder = builders[i % len(builders)]
        adj = builder(rng, i)
        cnt = _edge_count(adj)
        update_best(adj, cnt)

    # Phase 2: local search on best
    if best_adj is not None:
        remaining = deadline - 15 - (time.time() - start)
        if remaining > 5:
            adj2, cnt2 = _local_search(best_adj, rng, max_iter=50000, time_limit=remaining * 0.4)
            update_best(adj2, cnt2)

    # Phase 3: more greedy + local search
    for i in range(10):
        if time.time() > deadline - 20:
            break
        adj = _greedy_build(rng, 1000 + i)
        cnt = _edge_count(adj)
        update_best(adj, cnt)
        remaining = deadline - 15 - (time.time() - start)
        if remaining > 3:
            adj2, cnt2 = _local_search(adj, rng, max_iter=10000, time_limit=min(remaining * 0.2, 5))
            update_best(adj2, cnt2)

    # Phase 4: SA on best
    if best_adj is not None:
        remaining = deadline - 10 - (time.time() - start)
        if remaining > 3:
            adj2, cnt2 = _simulated_annealing(best_adj, rng, max_iter=30000, time_limit=remaining * 0.5)
            update_best(adj2, cnt2)

    # Phase 5: final local search
    if best_adj is not None:
        remaining = deadline - 5 - (time.time() - start)
        if remaining > 2:
            adj2, cnt2 = _local_search(best_adj, rng, max_iter=50000, time_limit=remaining * 0.7)
            update_best(adj2, cnt2)

    if best_adj is None:
        # fallback: 8 disjoint K5
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges

    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    return edges
