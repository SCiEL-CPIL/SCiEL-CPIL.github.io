import time
import random
from itertools import combinations

_START = time.time()
DEADLINE = 115.0  # 150s limit, ~35s safety margin

def _time_left():
    return DEADLINE - (time.time() - _START)

# Precompute all 4-cycles once (independent of graph)
def _precompute_cycles(n):
    cycles = []
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        # 3 ways to make a 4-cycle on 4 labeled vertices
        cycles.append(((a, b), (b, c), (c, d), (d, a)))
        cycles.append(((a, b), (b, d), (d, c), (c, a)))
        cycles.append(((a, c), (c, b), (b, d), (d, a)))
    return cycles

_CYCLES = _precompute_cycles(20)

def _validate(edges):
    """Return True iff every 4-cycle that is fully present is rainbow."""
    edge_set = {}
    for u, v, c in edges:
        edge_set[(u, v)] = c
        edge_set[(v, u)] = c
    for cyc in _CYCLES:
        cols = []
        ok = True
        for (u, v) in cyc:
            c = edge_set.get((u, v))
            if c is None:
                ok = False
                break
            cols.append(c)
        if ok and len(set(cols)) != 4:
            return False
    return True

def _greedy_build(seed, initial_edges, order_type='random'):
    """Build a valid coloured graph, optionally starting from initial_edges."""
    n = 20
    num_colors = 20
    rng = random.Random(seed)
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    adj = [[-1] * n for _ in range(n)]
    neighbors = [set() for _ in range(n)]
    edges = []

    if initial_edges:
        for u, v, c in initial_edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
        existing = set((min(u, v), max(u, v)) for u, v, _ in edges)
        remaining = [e for e in all_edges if e not in existing]
    else:
        remaining = all_edges[:]

    if order_type == 'random':
        rng.shuffle(remaining)
    elif order_type == 'reverse':
        remaining.reverse()

    for (u, v) in remaining:
        forbidden = set()
        impossible = False
        
        # Calculate forbidden colors and check for impossibility
        # Also gather color frequencies for heuristic choice
        color_freq = {}
        
        # Count current color usage in neighbors of u and v
        for a in neighbors[u]:
            if a != v:
                c = adj[u][a]
                color_freq[c] = color_freq.get(c, 0) + 1
        for b in neighbors[v]:
            if b != u:
                c = adj[v][b]
                color_freq[c] = color_freq.get(c, 0) + 1

        for a in neighbors[u]:
            if a == v:
                continue
            ca = adj[u][a]
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                if adj[a][b] != -1:
                    cb = adj[a][b]
                    cv = adj[b][v]
                    # Colors in the cycle u-a-b-v-u are ca, cb, cv, and new_color
                    # new_color must be distinct from ca, cb, cv
                    if ca == cb or cb == cv or ca == cv:
                        # The existing edges already form a non-rainbow cycle part?
                        # No, the cycle is u-a-b-v-u. The edges are (u,a), (a,b), (b,v), (v,u).
                        # If ca, cb, cv are not distinct, then no matter what color we pick for (v,u),
                        # the set {ca, cb, cv, new_color} will have duplicates among ca, cb, cv.
                        # Thus, this specific 4-cycle cannot be made rainbow.
                        # So we cannot add (u,v).
                        impossible = True
                        break
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
            if impossible:
                break
        
        if not impossible:
            # Find valid colors
            valid_colors = [c for c in range(num_colors) if c not in forbidden]
            if valid_colors:
                # Heuristic: choose the color with minimum frequency in neighbors
                best_color = min(valid_colors, key=lambda c: color_freq.get(c, 0))
                adj[u][v] = best_color
                adj[v][u] = best_color
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, best_color])
    return edges

def _bipartite_seed(rng):
    """Build a K_{10,10}-style bipartite core with a linear colouring,
    removing edges to fix any non-rainbow 4-cycles."""
    n = 20
    A = list(range(10))
    B = list(range(10, 20))
    a = rng.randint(0, 19)
    b = rng.randint(1, 19)
    c0 = rng.randint(0, 19)

    def col(u, v):
        return (a * u + b * v + c0) % 20

    ec = {}
    for u in A:
        for v in B:
            ec[(u, v)] = col(u, v)
            ec[(v, u)] = col(u, v)

    bad = []
    for i in range(10):
        for j in range(i + 1, 10):
            u1, u2 = A[i], A[j]
            for k in range(10):
                for l in range(k + 1, 10):
                    v1, v2 = B[k], B[l]
                    cols = {ec[(u1, v1)], ec[(u2, v1)], ec[(u2, v2)], ec[(u1, v2)]}
                    if len(cols) < 4:
                        bad.append([(u1, v1), (u2, v1), (u2, v2), (u1, v2)])

    remove = set()
    rem = bad[:]
    while rem and _time_left() > 15.0:
        best_e, best_hits = None, 0
        cands = set()
        for cyc in rem:
            for e in cyc:
                if e not in remove:
                    cands.add(e)
        if not cands:
            break
        for e in cands:
            hits = sum(1 for cyc in rem if e in cyc)
            if hits > best_hits:
                best_hits = hits
                best_e = e
        if best_e is None:
            break
        remove.add(best_e)
        rem = [cyc for cyc in rem if best_e not in cyc]

    out = []
    for u in A:
        for v in B:
            if (u, v) not in remove:
                out.append([u, v, ec[(u, v)]])
    return out

def construct():
    rng = random.Random()
    best_edges = []
    best_count = 0

    # Phase 1: random greedy restarts (no seed)
    for ot in ('random', 'natural', 'reverse'):
        for i in range(60):
            if _time_left() < 20.0:
                break
            edges = _greedy_build(i * 7 + 1, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    # Phase 2: bipartite-seeded greedy
    for i in range(20):
        if _time_left() < 20.0:
            break
        seed_edges = _bipartite_seed(rng)
        edges = _greedy_build(1000 + i, seed_edges, 'random')
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    # Phase 3: local search (remove/re-add)
    current = best_edges[:]
    for it in range(4000):
        if _time_left() < 8.0:
            break
        k = random.randint(1, 5)
        if len(current) <= k:
            continue
        to_remove = random.sample(current, k)
        rset = set((min(u, v), max(u, v)) for u, v, _ in to_remove)
        base = [e for e in current if (min(e[0], e[1]), max(e[0], e[1])) not in rset]
        new_edges = _greedy_build(it * 131 + random.randint(0, 100), base, 'random')
        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best_edges = new_edges
            current = new_edges[:]

    # Final safety: guarantee validity of what we return
    if not _validate(best_edges):
        # Fallback: rebuild a guaranteed-valid solution
        for ot in ('random', 'natural', 'reverse'):
            for i in range(20):
                if _time_left() < 2.0:
                    break
                e = _greedy_build(5000 + i, None, ot)
                if _validate(e) and len(e) > best_count:
                    best_count = len(e)
                    best_edges = e
        # If still invalid (shouldn't happen), return a trivial valid graph
        if not _validate(best_edges):
            best_edges = [[0, v, 0] for v in range(1, 20)]

    return best_edges
