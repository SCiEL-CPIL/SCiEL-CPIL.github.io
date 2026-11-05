"""Optimize a 16-vertex, 65-edge graph to minimize the maximum triangle degree sum.

Design family: local search over the 65-edge neighborhood of K_{8,8}+1 using
edge swaps (keeps edge count fixed at 65). We use a smooth soft-max
(log-sum-exp) surrogate during simulated annealing to escape local optima,
then an exact min-max polish pass. A hard deadline guard returns the best
valid result found with a safety margin before the 150 s wall-clock limit.
"""
import time
import random
import math
from itertools import combinations

N = 16
E = 65
DEADLINE = 132.0  # safety margin before the 150 s evaluator kill

# Precompute the full vertex-pair list (120 possible edges).
PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N)]
PAIR_INDEX = {p: i for i, p in enumerate(PAIRS)}


def _base_adj():
    """K_{8,8} plus one intra edge -> 65 edges."""
    adj = [[0] * N for _ in range(N)]
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
    adj[0][1] = 1
    adj[1][0] = 1
    return adj


def _degrees(adj):
    return [sum(adj[i]) for i in range(N)]


def _triangles(adj):
    """Return list of (u,v,w) with u<v<w that form a triangle."""
    tris = []
    for u, v, w in combinations(range(N), 3):
        if adj[u][v] and adj[v][w] and adj[w][u]:
            tris.append((u, v, w))
    return tris


def _max_tri_sum(adj, tris):
    deg = _degrees(adj)
    best = -1
    for (u, v, w) in tris:
        s = deg[u] + deg[v] + deg[w]
        if s > best:
            best = s
    return best


def _soft_max(tri_sums, t):
    """Smooth surrogate for max(tri_sums) via log-sum-exp, scaled to ~max."""
    if not tri_sums:
        return 0.0
    m = max(tri_sums)
    if t <= 1e-9:
        return float(m)
    s = sum(math.exp((x - m) / t) for x in tri_sums)
    return m + t * math.log(s)


def _swap(adj, a, b):
    """Remove edge a, add edge b (both given as (u,v)). Caller checks validity."""
    u, v = a
    adj[u][v] = 0
    adj[v][u] = 0
    x, y = b
    adj[x][y] = 1
    adj[y][x] = 1


def _restore(adj, a, b):
    """Undo a swap: re-add a, remove b."""
    x, y = b
    adj[x][y] = 0
    adj[y][x] = 0
    u, v = a
    adj[u][v] = 1
    adj[v][u] = 1


def _candidate_swaps(adj):
    """Yield (a,b) swaps: remove an edge a, add a non-edge b.
    b must not be a, and must not already be present."""
    present = []
    nonpresent = []
    for (u, v) in PAIRS:
        if adj[u][v]:
            present.append((u, v))
        else:
            nonpresent.append((u, v))
    return present, nonpresent


def _optimize(adj, rng, start_time, t0, t1, anneal_steps, polish_steps):
    """Run annealing (soft-max) then polish (exact). Returns (best_score, best_adj)."""
    present, nonpresent = _candidate_swaps(adj)
    # Build a neighbor-pair index for fast non-edge lookup
    present_set = set(present)

    deg = _degrees(adj)
    tris = _triangles(adj)
    # Precompute triangle vertex lists for fast degree-sum updates
    tri_deg = [deg[u] + deg[v] + deg[w] for (u, v, w) in tris]
    soft = _soft_max(tri_deg, t0)
    exact = max(tri_deg) if tri_deg else 0

    best_exact = exact
    best_adj = [row[:] for row in adj]

    # ---- Annealing phase (soft-max surrogate) ----
    for step in range(anneal_steps):
        if time.time() > DEADLINE:
            break
        # temperature schedule
        frac = step / max(1, anneal_steps - 1)
        cur_t = t0 + (t1 - t0) * frac

        # pick a random swap
        a = present[rng.randrange(len(present))]
        b = nonpresent[rng.randrange(len(nonpresent))]
        if b in present_set:
            continue

        # apply swap
        _swap(adj, a, b)
        present_set.discard(a)
        present_set.add(b)

        # recompute degrees and triangle sums (fast enough for n=16)
        new_deg = _degrees(adj)
        new_tris = _triangles(adj)
        new_tri_deg = [new_deg[u] + new_deg[v] + new_deg[w] for (u, v, w) in new_tris]
        new_soft = _soft_max(new_tri_deg, cur_t)
        new_exact = max(new_tri_deg) if new_tri_deg else 0

        delta = new_soft - soft
        if delta <= 0 or rng.random() < math.exp(-delta / max(1e-9, cur_t)):
            # accept
            soft = new_soft
            exact = new_exact
            deg = new_deg
            tris = new_tris
            tri_deg = new_tri_deg
            if new_exact < best_exact:
                best_exact = new_exact
                best_adj = [row[:] for row in adj]
        else:
            # reject: restore
            _restore(adj, a, b)
            present_set.discard(b)
            present_set.add(a)

    # ---- Polish phase (exact min-max) ----
    present, nonpresent = _candidate_swaps(adj)
    present_set = set(present)
    deg = _degrees(adj)
    tris = _triangles(adj)
    tri_deg = [deg[u] + deg[v] + deg[w] for (u, v, w) in tris]
    exact = max(tri_deg) if tri_deg else 0
    if exact < best_exact:
        best_exact = exact
        best_adj = [row[:] for row in adj]

    for step in range(polish_steps):
        if time.time() > DEADLINE:
            break
        # greedy: try all swaps, take the one minimizing exact max
        best_swap = None
        best_val = exact
        for a in present:
            for b in nonpresent:
                if b in present_set:
                    continue
                _swap(adj, a, b)
                d2 = _degrees(adj)
                t2 = _triangles(adj)
                v2 = max((d2[u] + d2[v] + d2[w]) for (u, v, w) in t2) if t2 else 0
                _restore(adj, a, b)
                if v2 < best_val:
                    best_val = v2
                    best_swap = (a, b)
        if best_swap is None:
            break
        a, b = best_swap
        _swap(adj, a, b)
        present_set.discard(a)
        present_set.add(b)
        deg = _degrees(adj)
        tris = _triangles(adj)
        tri_deg = [deg[u] + deg[v] + deg[w] for (u, v, w) in tris]
        exact = max(tri_deg) if tri_deg else 0
        if exact < best_exact:
            best_exact = exact
            best_adj = [row[:] for row in adj]

    return best_exact, best_adj


def construct():
    start = time.time()
    rng = random.Random(12345)

    best_score = None
    best_adj = None

    adj = _base_adj()

    # First quick pass to get a baseline
    s, a = _optimize(adj, rng, start, t0=5.0, t1=0.5,
                     anneal_steps=4000, polish_steps=30)
    best_score = s
    best_adj = a

    # Multiple restarts while we have time
    restart = 0
    while time.time() < DEADLINE - 8.0:
        restart += 1
        # fresh start from base
        adj = _base_adj()
        # randomize the intra edge choice for diversity
        # (keep structure: 8-8 split, add a random intra edge)
        # Rebuild: remove base intra edge, add a random one
        adj[0][1] = 0
        adj[1][0] = 0
        # pick random intra-part edge
        if rng.random() < 0.5:
            u, v = rng.randrange(8), rng.randrange(8)
            if u == v:
                u = (u + 1) % 8
        else:
            u, v = rng.randrange(8, 16), rng.randrange(8, 16)
            if u == v:
                u = (u + 1) % 8
        adj[u][v] = 1
        adj[v][u] = 1

        s, a = _optimize(adj, rng, start, t0=5.0, t1=0.3,
                         anneal_steps=6000, polish_steps=40)
        if best_score is None or s < best_score:
            best_score = s
            best_adj = a
        if best_score <= 23:
            break

    # Final exact verification / polish on the best
    if best_adj is not None:
        s, a = _optimize(best_adj, rng, start, t0=1.0, t1=0.1,
                         anneal_steps=0, polish_steps=20)
        if s < best_score:
            best_score = s
            best_adj = a

    # Build edge list from best_adj
    edges = []
    for (u, v) in PAIRS:
        if best_adj[u][v]:
            edges.append([u, v])

    # Safety: must have exactly 65 edges
    assert len(edges) == E, f"edge count {len(edges)} != {E}"

    return {'n': N, 'edges': edges}


if __name__ == '__main__':
    res = construct()
    adj = [[0] * N for _ in range(N)]
    for (u, v) in res['edges']:
        adj[u][v] = 1
        adj[v][u] = 1
    deg = _degrees(adj)
    tris = _triangles(adj)
    mx = max((deg[u] + deg[v] + deg[w]) for (u, v, w) in tris) if tris else 0
    print("edges:", len(res['edges']), "max triangle degree sum:", mx)
