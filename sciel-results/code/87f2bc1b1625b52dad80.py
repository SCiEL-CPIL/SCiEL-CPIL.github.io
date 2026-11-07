"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

N = 15
DEADLINE = time.time() + 80.0  # safety margin before 90s kill

def _valid(g, deg):
    """Full validation: no simple 3-edge path s-u-v-t (s!=t) with deg(s)==deg(t)."""
    for u in range(N):
        nu = g[u]
        for v in nu:
            nv = g[v]
            for s in nu:
                if s == v:
                    continue
                ds = deg[s]
                for t in nv:
                    if t == u or t == s:
                        continue
                    if deg[t] == ds:
                        return False
    return True

def _try_add(g, deg, u, v):
    """Check if adding edge (u,v) preserves validity. Returns True if valid."""
    # After adding, deg[u] and deg[v] increase by 1
    # Affected paths: those through edge (u,v), or edges incident to u or v
    # Simplest: check all edges that could be affected
    # An edge (a,b) is affected if a or b is u or v
    for a in range(N):
        if a not in g[u] and a not in g[v] and a != u and a != v:
            continue
        for b in g[a]:
            if b <= a:
                continue
            # edge (a,b) exists; check it
            na = g[a]
            nb = g[b]
            # simulate degrees
            da = deg[a] + (1 if a == u or a == v else 0)
            db = deg[b] + (1 if b == u or b == v else 0)
            for s in na:
                if s == b:
                    continue
                ds = deg[s] + (1 if s == u or s == v else 0)
                for t in nb:
                    if t == a or t == s:
                        continue
                    dt = deg[t] + (1 if t == u or t == v else 0)
                    if ds == dt:
                        return False
    # Also need to check the new edge (u,v) itself
    nu = g[u]
    nv = g[v]
    du = deg[u] + 1
    dv = deg[v] + 1
    for s in nu:
        if s == v:
            continue
        ds = deg[s] + (1 if s == u or s == v else 0)
        for t in nv:
            if t == u or t == s:
                continue
            dt = deg[t] + (1 if t == u or t == v else 0)
            if ds == dt:
                return False
    return True

def _greedy(seed):
    rng = random.Random(seed)
    g = [set() for _ in range(N)]
    deg = [0] * N
    # Start with a random matching or star to break symmetry
    # Try random edge order
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(edges)
    for u, v in edges:
        if u in g[v]:
            continue
        if _try_add(g, deg, u, v):
            g[u].add(v)
            g[v].add(u)
            deg[u] += 1
            deg[v] += 1
    return len(deg) // 2, g, deg

def construct():
    best_edges = None
    best_count = 0
    t0 = time.time()
    seed = 42
    # Multiple restarts
    for restart in range(500):
        if time.time() > DEADLINE:
            break
        count, g, deg = _greedy(seed)
        if count > best_count:
            best_count = count
            best_edges = g
            best_deg = deg
        seed += 1
        # If we've found a very dense graph, we can stop early
        if best_count >= 40:
            break
    # Final validation
    if best_edges is None:
        # Fallback: star
        g = [set() for _ in range(N)]
        for v in range(1, N):
            g[0].add(v)
            g[v].add(0)
        deg = [N - 1] + [1] * (N - 1)
        return [[0, v] for v in range(1, N)]
    # Validate
    if not _valid(best_edges, best_deg):
        # Fallback to star
        return [[0, v] for v in range(1, N)]
    # Build edge list
    result = []
    for u in range(N):
        for v in best_edges[u]:
            if u < v:
                result.append([int(u), int(v)])
    return result
# EVOLVE-BLOCK-END
