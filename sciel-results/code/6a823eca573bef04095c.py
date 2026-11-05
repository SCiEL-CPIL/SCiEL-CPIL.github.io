"""C4-free graph search for R(C4,K11) > n, aiming for n=39."""
import time
import random
import sys

# EVOLVE-BLOCK-START
_DEADLINE = 118.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() >= _DEADLINE

def _build_polarity_graph(q=5):
    """Polarity graph of PG(2,q): (q^2+q+1) vertices, C4-free."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv = pow(x, q-2, q)
                    cx, cy, cz = 1, (y*inv) % q, (z*inv) % q
                elif y != 0:
                    inv = pow(y, q-2, q)
                    cx, cy, cz = 0, 1, (z*inv) % q
                else:
                    cx, cy, cz = 0, 0, 1
                key = (cx, cy, cz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    npts = len(points)
    adj = [set() for _ in range(npts)]
    for i in range(npts):
        for j in range(i+1, npts):
            a, b = points[i], points[j]
            s = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]) % q
            if s == 0:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _check_c4(adj):
    """Return True if no two vertices share >=2 common neighbours."""
    n = len(adj)
    for i in range(n):
        for j in range(i+1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def _greedy_alpha_lb(adj):
    """Greedy lower bound on independence number."""
    n = len(adj)
    remaining = set(range(n))
    alpha = 0
    while remaining:
        best_v = None
        best_deg = 10**9
        for v in remaining:
            d = len(adj[v] & remaining)
            if d < best_deg:
                best_deg = d
                best_v = v
        alpha += 1
        remaining.discard(best_v)
        remaining -= adj[best_v]
    return alpha

def _has_independent_set(adj, limit=11):
    """Exact check: does an independent set of size >= limit exist?"""
    n = len(adj)
    if n < limit:
        return False
    # Bitmask representation
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m

    sys.setrecursionlimit(10000)

    # Precompute for speed: use a recursive branch-and-bound
    # Order vertices by degree (higher degree first for exclusion branching)
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    # Remap
    new_idx = {v: i for i, v in enumerate(order)}
    new_masks = [0] * n
    for v in range(n):
        m = 0
        for u in adj[v]:
            m |= (1 << new_idx[u])
        new_masks[new_idx[v]] = m

    # Use Bron-Kerbosch style on complement for max independent set
    # Actually, simpler: branch on vertices in order, include/exclude
    best = [0]

    def branch(rem_mask, current_size):
        if current_size >= limit:
            return True
        if rem_mask == 0:
            return False
        # Upper bound: current_size + popcount(rem_mask)
        if current_size + rem_mask.bit_count() < limit:
            return False
        # Pick a vertex from rem_mask
        # Choose vertex with max degree in remaining for better pruning
        rem = rem_mask
        best_v = -1
        best_deg = -1
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            deg = (rem_mask & new_masks[v]).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        # Branch 1: include best_v
        new_rem = rem_mask & ~(new_masks[best_v] | (1 << best_v))
        if branch(new_rem, current_size + 1):
            return True
        # Branch 2: exclude best_v
        new_rem2 = rem_mask & ~(1 << best_v)
        if branch(new_rem2, current_size):
            return True
        return False

    full = (1 << n) - 1
    return branch(full, 0)

def _try_add_vertex(adj, target_n):
    """Try to add one vertex while maintaining C4-free and alpha < 11."""
    n = len(adj)
    if n >= target_n:
        return None, False

    # Try random connection patterns
    for attempt in range(100):
        if _deadline_hit():
            return None, False
        new_neigh = set()
        for v in range(n):
            if random.random() < 0.25:
                new_neigh.add(v)
        # Check C4 condition: for any two neighbors a,b of new vertex,
        # they must not share a common neighbor in the existing graph
        valid = True
        neigh_list = list(new_neigh)
        for i in range(len(neigh_list)):
            for j in range(i+1, len(neigh_list)):
                a, b = neigh_list[i], neigh_list[j]
                if len(adj[a] & adj[b]) >= 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        # Check alpha < 11 in the new graph
        new_adj = adj + [set()]
        new_adj[-1] = new_neigh
        for v in new_neigh:
            new_adj[v].add(n)
        # Quick greedy check
        if _greedy_alpha_lb(new_adj) >= 11:
            # Do exact check
            if _has_independent_set(new_adj, 11):
                continue
        return new_adj, True
    return None, False

def _local_search_add_edges(adj, target_n, max_steps=500):
    """Try to add edges to reduce alpha while maintaining C4-free."""
    n = len(adj)
    adj = [set(s) for s in adj]
    for step in range(max_steps):
        if _deadline_hit():
            break
        # Try adding a random edge
        u, v = random.sample(range(n), 2)
        if v in adj[u]:
            continue
        # Check C4: u and v must not share a common neighbor
        if len(adj[u] & adj[v]) >= 1:
            continue
        # Try the edge
        adj[u].add(v)
        adj[v].add(u)
        # Check alpha
        if _greedy_alpha_lb(adj) >= 11:
            if _has_independent_set(adj, 11):
                adj[u].discard(v)
                adj[v].discard(u)
                continue
        # Keep the edge
    return adj

def construct():
    random.seed(42)
    best_n = 30
    best_edges = None

    # Start with 10 disjoint triangles (n=30, alpha=10)
    adj = [set() for _ in range(30)]
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        adj[a].add(b); adj[b].add(a)
        adj[a].add(c); adj[c].add(a)
        adj[b].add(c); adj[c].add(b)

    # Try to grow to 39
    for target in range(31, 40):
        if _deadline_hit():
            break
        success = False
        for attempt in range(200):
            if _deadline_hit():
                break
            new_adj, ok = _try_add_vertex(adj, target)
            if ok:
                adj = new_adj
                success = True
                break
        if not success:
            break

    n = len(adj)
    if n > best_n:
        best_n = n
        best_edges = [[min(u,v), max(u,v)] for u in range(n) for v in adj[u] if u < v]

    # Also try the polarity graph of PG(2,5) as a starting point
    if not _deadline_hit():
        pg_adj = _build_polarity_graph(5)
        if _check_c4(pg_adj):
            pn = len(pg_adj)
            if _has_independent_set(pg_adj, 11):
                pass  # alpha >= 11, not valid directly
            else:
                if pn > best_n:
                    best_n = pn
                    best_edges = [[min(u,v), max(u,v)] for u in range(pn) for v in pg_adj[u] if u < v]

    # Try local search on the best graph found
    if best_edges is not None and not _deadline_hit():
        # Rebuild adj from best_edges
        adj2 = [set() for _ in range(best_n)]
        for u, v in best_edges:
            adj2[u].add(v)
            adj2[v].add(u)
        # Try to add more vertices
        for target in range(best_n + 1, 40):
            if _deadline_hit():
                break
            for attempt in range(100):
                if _deadline_hit():
                    break
                new_adj, ok = _try_add_vertex(adj2, target)
                if ok:
                    adj2 = new_adj
                    best_n = target
                    best_edges = [[min(u,v), max(u,v)] for u in range(best_n) for v in adj2[u] if u < v]
                    break
            else:
                break

    if best_edges is None:
        # Fallback: 10 disjoint triangles
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a,b], [a,c], [b,c]]
        return {'n': 30, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
