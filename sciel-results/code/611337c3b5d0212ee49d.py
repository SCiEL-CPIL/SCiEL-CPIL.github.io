import time
import random
from itertools import combinations

_START = time.time()
DEADLINE = 130.0  # 150s limit, ~20s safety margin

N = 20
N_COLORS = 20
ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N)]


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
    if not edges:
        return True
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


def _build_state(edges):
    adj = [[-1] * N for _ in range(N)]
    neighbors = [set() for _ in range(N)]
    for u, v, c in edges:
        adj[u][v] = c
        adj[v][u] = c
        neighbors[u].add(v)
        neighbors[v].add(u)
    return adj, neighbors


def _fast_check_after_change(adj, neighbors, u, v):
    """Check only the 4-cycles that involve edge (u,v).

    A 4-cycle containing (u,v) has the form u - x - y - v - u, i.e.
    x in N(u), y in N(v), x != v, y != u, x != y, and (x,y) is an edge.
    The four edges are (u,x),(x,y),(y,v),(v,u).
    """
    cv = adj[v][u]
    for x in list(neighbors[u]):
        if x == v:
            continue
        cx = adj[u][x]
        for y in list(neighbors[v]):
            if y == u or y == x:
                continue
            if adj[x][y] == -1:
                continue
            cols = (cx, adj[x][y], adj[y][v], cv)
            if len(set(cols)) != 4:
                return False
    return True


def _forbidden_and_possible(adj, neighbors, u, v):
    """Return (possible, forbidden) for adding edge (u,v)."""
    forbidden = set()
    possible = True
    for a in list(neighbors[u]):
        if a == v:
            continue
        ca = adj[u][a]
        for b in list(neighbors[v]):
            if b == u or b == a:
                continue
            if adj[a][b] != -1:
                cb = adj[a][b]
                cv = adj[b][v]
                if ca == cb or cb == cv or ca == cv:
                    possible = False
                    break
                forbidden.add(ca)
                forbidden.add(cb)
                forbidden.add(cv)
        if not possible:
            break
    return possible, forbidden


def _greedy_build(seed, initial_edges, order_type='random'):
    """Build a valid coloured graph, optionally starting from initial_edges."""
    rng = random.Random(seed)
    adj = [[-1] * N for _ in range(N)]
    neighbors = [set() for _ in range(N)]
    edges = []

    if initial_edges:
        for u, v, c in initial_edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
        existing = set((min(u, v), max(u, v)) for u, v, _ in edges)
        remaining = [e for e in ALL_PAIRS if e not in existing]
    else:
        remaining = ALL_PAIRS[:]

    if order_type == 'random':
        rng.shuffle(remaining)
    elif order_type == 'reverse':
        remaining.reverse()

    for (u, v) in remaining:
        if v in neighbors[u]:
            continue
        possible, forbidden = _forbidden_and_possible(adj, neighbors, u, v)
        if not possible:
            continue
        # Choose color that minimizes future conflicts: prefer colors
        # that appear least among neighbors of u and v
        color_freq = {}
        for a in neighbors[u]:
            if a != v:
                ca = adj[u][a]
                color_freq[ca] = color_freq.get(ca, 0) + 1
        for b in neighbors[v]:
            if b != u:
                cb = adj[v][b]
                color_freq[cb] = color_freq.get(cb, 0) + 1
        available = [c for c in range(N_COLORS) if c not in forbidden]
        if available:
            rng.shuffle(available)
            available.sort(key=lambda c: color_freq.get(c, 0))
            c = available[0]
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
    return edges


def _bipartite_seed(rng):
    """Build a K_{10,10}-style bipartite core with a linear colouring,
    removing edges to fix any non-rainbow 4-cycles."""
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
        best_e, best_score = None, -1
        cands = set()
        for cyc in rem:
            for e in cyc:
                if e not in remove:
                    cands.add(e)
        if not cands:
            break
        for e in cands:
            hits = sum(1 for cyc in rem if e in cyc)
            score = hits + rng.random() * 0.5
            if score > best_score:
                best_score = score
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


def _try_add_missing(adj, neighbors, current, rng, max_candidates=30):
    """Try to add missing edges with valid colors. Returns new edge list or None."""
    existing = set((min(u, v), max(u, v)) for u, v, _ in current)
    missing = [e for e in ALL_PAIRS if e not in existing]
    if not missing:
        return None
    rng.shuffle(missing)
    new_edges = current[:]
    changed = False
    for (u, v) in missing[:max_candidates]:
        if v in neighbors[u]:
            continue
        possible, forbidden = _forbidden_and_possible(adj, neighbors, u, v)
        if not possible:
            continue
        available = [c for c in range(N_COLORS) if c not in forbidden]
        if available:
            c = rng.choice(available)
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            new_edges.append([u, v, c])
            changed = True
    return new_edges if changed else None


def construct():
    rng = random.Random()
    best_edges = []
    best_count = 0

    # Phase 1: random greedy restarts (no seed)
    for ot in ('random', 'natural', 'reverse'):
        for i in range(60):
            if _time_left() < 25.0:
                break
            edges = _greedy_build(i * 7 + 1, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]

    # Phase 2: bipartite-seeded greedy
    for i in range(25):
        if _time_left() < 25.0:
            break
        seed_edges = _bipartite_seed(rng)
        edges = _greedy_build(1000 + i, seed_edges, 'random')
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]

    # Phase 3: local search with re-color + remove + add moves
    current = best_edges[:]
    it = 0
    while _time_left() > 8.0:
        it += 1
        adj, neighbors = _build_state(current)
        move_type = random.random()

        if move_type < 0.55 and len(current) > 0:
            # ---- Re-color move (cheap via incremental check) ----
            u, v, old_c = random.choice(current)
            tried = random.sample([c for c in range(N_COLORS) if c != old_c],
                                  k=min(8, N_COLORS - 1))
            improved = False
            for nc in tried:
                adj[u][v] = nc
                adj[v][u] = nc
                if not _fast_check_after_change(adj, neighbors, u, v):
                    adj[u][v] = old_c
                    adj[v][u] = old_c
                    continue
                # valid re-color; build candidate edge list
                cand = [e[:] for e in current]
                for e in cand:
                    if (min(e[0], e[1]), max(e[0], e[1])) == (min(u, v), max(u, v)):
                        e[2] = nc
                        break
                # re-greedy to fill gaps unlocked by the re-color
                seed = it * 131 + random.randint(0, 1000)
                new_edges = _greedy_build(seed, cand, 'random')
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges[:]
                    current = new_edges[:]
                    improved = True
                    break
                # revert
                adj[u][v] = old_c
                adj[v][u] = old_c
            if not improved:
                # try to add missing edges on the current (recolored) state
                added = _try_add_missing(adj, neighbors, current, rng)
                if added and len(added) > best_count:
                    best_count = len(added)
                    best_edges = added[:]
                    current = added[:]

        elif move_type < 0.85:
            # ---- Remove + re-greedy move ----
            k = random.randint(1, 4)
            if len(current) > k:
                edges_to_remove = random.sample(current, k)
                rset = set((min(u, v), max(u, v)) for u, v, _ in edges_to_remove)
                base = [e for e in current
                        if (min(e[0], e[1]), max(e[0], e[1])) not in rset]
                seed = it * 131 + random.randint(0, 1000)
                new_edges = _greedy_build(seed, base, 'random')
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges[:]
                    current = new_edges[:]
        else:
            # ---- Pure add move ----
            added = _try_add_missing(adj, neighbors, current, rng, max_candidates=40)
            if added and len(added) > best_count:
                best_count = len(added)
                best_edges = added[:]
                current = added[:]

    # Final safety: guarantee validity of what we return
    if not _validate(best_edges):
        for ot in ('random', 'natural', 'reverse'):
            for i in range(20):
                if _time_left() < 2.0:
                    break
                e = _greedy_build(5000 + i, None, ot)
                if _validate(e) and len(e) > best_count:
                    best_count = len(e)
                    best_edges = e[:]
        if not _validate(best_edges):
            best_edges = [[0, v, 0] for v in range(1, N)]

    # Final polish: try to add any remaining missing edges (deterministic pass)
    adj, neighbors = _build_state(best_edges)
    existing = set((min(u, v), max(u, v)) for u, v, _ in best_edges)
    missing = [e for e in ALL_PAIRS if e not in existing]
    if missing and _time_left() > 2.0:
        for (u, v) in missing:
            if _time_left() < 2.0:
                break
            if v in neighbors[u]:
                continue
            possible, forbidden = _forbidden_and_possible(adj, neighbors, u, v)
            if possible:
                available = [c for c in range(N_COLORS) if c not in forbidden]
                if available:
                    c = available[0]
                    best_edges.append([u, v, c])
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)

    return best_edges
