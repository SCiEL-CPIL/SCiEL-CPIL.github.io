"""C6-free graph on 40 vertices: maximize edges via greedy build + local search."""
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
    
    A C6 containing the new edge (u,v) has the form:
    u - a - x - y - b - v - u
    where a in N(u), b in N(v), a-b is NOT an edge,
    and x,y are distinct vertices not in {u,v,a,b} with edges a-x, x-y, y-b.
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
                continue  # a-b edge -> would create smaller cycle, not valid C6
            nb = adj[b]
            # Need x in N(a)\{u,v,b} and y in N(b)\{u,v,a} with x-y edge, x!=y
            # Iterate over the smaller set
            if len(na) < len(nb):
                for x in na:
                    if x == u or x == v or x == b:
                        continue
                    nx = adj[x]
                    for y in nx:
                        if y == u or y == v or y == a:
                            continue
                        if y in nb:
                            return True
            else:
                for y in nb:
                    if y == u or y == v or y == a:
                        continue
                    ny = adj[y]
                    for x in ny:
                        if x == u or x == v or x == a:
                            continue
                        if x in na:
                            return True
    return False


def _edge_count(adj):
    return sum(len(s) for s in adj) // 2


def _greedy_build(rng, seed):
    """Build a C6-free graph greedily with randomized edge ordering."""
    adj = _init_adj()
    order = list(ALL_PAIRS)
    rng.shuffle(order)
    for (u, v) in order:
        if v in adj[u]:
            continue
        if not _has_c6_full(adj, u, v):
            _add(adj, u, v)
    return adj


def _greedy_add_phase(adj, rng, max_tries, time_limit, start_time):
    """Try to add random non-edges that don't create C6."""
    for _ in range(max_tries):
        if time.time() - start_time > time_limit:
            break
        u = rng.randrange(N)
        v = rng.randrange(N)
        if u == v or v in adj[u]:
            continue
        if not _has_c6_full(adj, u, v):
            _add(adj, u, v)


def _local_search(adj, rng, max_iter, time_limit):
    """Local search: try edge swaps (remove one, add another) and pure additions."""
    cur = [set(s) for s in adj]
    cur_count = _edge_count(cur)
    best = [set(s) for s in cur]
    best_count = cur_count
    start = time.time()
    
    # Precompute edge list for faster sampling
    def get_edges(a):
        edges = []
        for u in range(N):
            for v in a[u]:
                if u < v:
                    edges.append((u, v))
        return edges
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Try a random edge swap
        edges = get_edges(cur)
        if not edges:
            break
        a, b = rng.choice(edges)
        
        # Try to add a random non-edge after removing (a,b)
        found = False
        for _ in range(300):
            c = rng.randrange(N)
            d = rng.randrange(N)
            if c == d or c >= d:
                continue
            if d in cur[c]:
                continue
            if (c, d) == (a, b) or (c, d) == (b, a):
                continue
            
            _remove(cur, a, b)
            ok = not _has_c6_full(cur, c, d)
            if ok:
                _add(cur, c, d)
                new_count = _edge_count(cur)
                if new_count >= cur_count:
                    cur_count = new_count
                    if new_count > best_count:
                        best = [set(s) for s in cur]
                        best_count = new_count
                    found = True
                    break
                else:
                    _remove(cur, c, d)
            else:
                _remove(cur, c, d)
            _add(cur, a, b)
        
        if not found:
            _add(cur, a, b)
            # Also try pure addition (no removal)
            for _ in range(50):
                c = rng.randrange(N)
                d = rng.randrange(N)
                if c == d or c >= d:
                    continue
                if d in cur[c]:
                    continue
                if not _has_c6_full(cur, c, d):
                    _add(cur, c, d)
                    new_count = _edge_count(cur)
                    if new_count > best_count:
                        best = [set(s) for s in cur]
                        best_count = new_count
                    break
    return best, best_count


def _multi_start_greedy_local(rng, time_budget):
    """Multiple greedy starts followed by local search."""
    best_adj = None
    best_count = -1
    start = time.time()
    
    n_greedy = 15
    for i in range(n_greedy):
        if time.time() - start > time_budget * 0.5:
            break
        adj = _greedy_build(rng, i)
        # Quick greedy add phase
        _greedy_add_phase(adj, rng, 5000, 3.0, time.time())
        cnt = _edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
    
    # Local search from best
    if best_adj is not None:
        remaining = time_budget - (time.time() - start) - 5
        if remaining > 3:
            adj2, cnt2 = _local_search(best_adj, rng, max_iter=30000, time_limit=remaining * 0.6)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2
    
    return best_adj, best_count


def _intensive_local_search(adj, rng, time_budget):
    """Run intensive local search with multiple restarts."""
    best_adj = [set(s) for s in adj]
    best_count = _edge_count(adj)
    start = time.time()
    
    for restart in range(5):
        remaining = time_budget - (time.time() - start) - 3
        if remaining < 2:
            break
        cur = [set(s) for s in best_adj]
        cur_count = _edge_count(cur)
        
        # Perturb: randomly remove a few edges
        n_remove = rng.randint(3, 8)
        edges = []
        for u in range(N):
            for v in cur[u]:
                if u < v:
                    edges.append((u, v))
        for _ in range(n_remove):
            if not edges:
                break
            a, b = rng.choice(edges)
            _remove(cur, a, b)
            edges.remove((a, b))
        
        # Rebuild greedily
        _greedy_add_phase(cur, rng, 10000, 5.0, time.time())
        cnt = _edge_count(cur)
        
        # Local search
        remaining2 = time_budget - (time.time() - start) - 3
        if remaining2 > 2:
            adj2, cnt2 = _local_search(cur, rng, max_iter=20000, time_limit=remaining2 * 0.5)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2
    
    return best_adj, best_count


def _validate_c6_free(edges):
    """Verify the graph has no C6."""
    adj = _init_adj()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check all edges for C6
    for u in range(N):
        for v in adj[u]:
            if u < v:
                if _has_c6_full(adj, u, v):
                    return False
    return True


def _fallback():
    """8 disjoint copies of K5 (56 edges)."""
    edges = []
    for k in range(8):
        edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    return edges


def construct():
    deadline = 145.0  # Safety margin from 150s
    start = time.time()
    rng = random.Random(12345)
    np.random.seed(12345)
    
    best_adj = None
    best_count = -1
    
    # Phase 1: Multi-start greedy + local search
    time_p1 = 50.0
    adj1, cnt1 = _multi_start_greedy_local(rng, time_p1)
    if adj1 is not None and cnt1 > best_count:
        best_count = cnt1
        best_adj = adj1
    
    # Phase 2: Intensive local search from best
    if best_adj is not None:
        elapsed = time.time() - start
        remaining = deadline - elapsed - 10
        if remaining > 5:
            adj2, cnt2 = _intensive_local_search(best_adj, rng, remaining * 0.7)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2
    
    # Phase 3: Final local search polish
    if best_adj is not None:
        elapsed = time.time() - start
        remaining = deadline - elapsed - 5
        if remaining > 3:
            adj3, cnt3 = _local_search(best_adj, rng, max_iter=50000, time_limit=remaining * 0.8)
            if cnt3 > best_count:
                best_count = cnt3
                best_adj = adj3
    
    # Fallback if nothing found
    if best_adj is None or best_count < 56:
        return _fallback()
    
    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    # Quick validation (skip full check if time is tight)
    if time.time() - start < deadline - 2:
        if not _validate_c6_free(edges):
            return _fallback()
    
    return edges
