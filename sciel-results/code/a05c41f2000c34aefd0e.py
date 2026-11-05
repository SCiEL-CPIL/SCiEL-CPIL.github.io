import time
import random
from itertools import combinations

_START = time.time()
DEADLINE = 115.0  # 150s limit, ~35s safety margin


def _time_left():
    return DEADLINE - (time.time() - _START)


# ---------------------------------------------------------------------------
# Precompute all 4-cycles once (independent of the graph).
# For every 4-set of vertices there are exactly 3 distinct 4-cycle structures.
# ---------------------------------------------------------------------------
def _precompute_cycles(n):
    cycles = []
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        cycles.append(((a, b), (b, c), (c, d), (d, a)))
        cycles.append(((a, b), (b, d), (d, c), (c, a)))
        cycles.append(((a, c), (c, b), (b, d), (d, a)))
    return cycles


_CYCLES = _precompute_cycles(20)
_N = 20
_NUM_COLORS = 20
_ALL_EDGES = [(u, v) for u in range(_N) for v in range(u + 1, _N)]


def _validate(edges):
    """True iff every fully-present 4-cycle is rainbow (4 distinct colours)."""
    edge_set = {}
    for u, v, c in edges:
        edge_set[(u, v)] = c
        edge_set[(v, u)] = c
    for cyc in _CYCLES:
        c0 = edge_set.get(cyc[0][0], cyc[0][1])
        c1 = edge_set.get(cyc[1][0], cyc[1][1])
        c2 = edge_set.get(cyc[2][0], cyc[2][1])
        c3 = edge_set.get(cyc[3][0], cyc[3][1])
        if c0 is None or c1 is None or c2 is None or c3 is None:
            continue
        if c0 == c1 or c0 == c2 or c0 == c3 or c1 == c2 or c1 == c3 or c2 == c3:
            return False
    return True


def _build_adj(edges):
    adj = [[-1] * _N for _ in range(_N)]
    neighbors = [set() for _ in range(_N)]
    for u, v, c in edges:
        adj[u][v] = c
        adj[v][u] = c
        neighbors[u].add(v)
        neighbors[v].add(u)
    return adj, neighbors


def _forbidden_for(adj, neighbors, u, v):
    """Return (possible, forbidden_set) for adding edge (u,v)."""
    forbidden = set()
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
                if ca == cb or cb == cv or ca == cv:
                    return False, forbidden
                forbidden.add(ca)
                forbidden.add(cb)
                forbidden.add(cv)
    return True, forbidden


def _greedy_build(seed, initial_edges=None, order_type='random'):
    """Build a valid coloured graph, optionally extending initial_edges."""
    rng = random.Random(seed)
    adj = [[-1] * _N for _ in range(_N)]
    neighbors = [set() for _ in range(_N)]
    edges = []

    if initial_edges:
        for u, v, c in initial_edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
        existing = set((min(u, v), max(u, v)) for u, v, _ in edges)
        remaining = [e for e in _ALL_EDGES if e not in existing]
    else:
        remaining = _ALL_EDGES[:]

    if order_type == 'random':
        rng.shuffle(remaining)
    elif order_type == 'reverse':
        remaining.reverse()

    for (u, v) in remaining:
        possible, forbidden = _forbidden_for(adj, neighbors, u, v)
        if not possible:
            continue
        # Prefer colours least used among the neighbours of u and v.
        color_freq = {}
        for a in neighbors[u]:
            if a != v:
                ca = adj[u][a]
                color_freq[ca] = color_freq.get(ca, 0) + 1
        for b in neighbors[v]:
            if b != u:
                cb = adj[v][b]
                color_freq[cb] = color_freq.get(cb, 0) + 1
        available = [c for c in range(_NUM_COLORS) if c not in forbidden]
        if not available:
            continue
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
    """K_{10,10}-style bipartite core with a linear colouring; remove edges to
    break every non-rainbow 4-cycle."""
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
    while rem and _time_left() > 25.0:
        best_e, best_score = None, -1.0
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


def _simulated_annealing(seed, initial_edges):
    """SA over add / remove / recolor moves, driven by the fast validator."""
    rng = random.Random(seed)
    state = [e[:] for e in initial_edges]
    adj, neighbors = _build_adj(state)
    best_state = [e[:] for e in state]
    best_len = len(state)
    edge_set = set((min(u, v), max(u, v)) for u, v, _ in state)

    temp = 1.0
    cooling = 0.9995

    while _time_left() > 12.0:
        move_type = rng.random()
        accepted = False

        if move_type < 0.4:
            # Add a missing edge with a valid colour.
            missing = [e for e in _ALL_EDGES if e not in edge_set]
            if missing:
                u, v = rng.choice(missing)
                possible, forbidden = _forbidden_for(adj, neighbors, u, v)
                if possible:
                    avail = [c for c in range(_NUM_COLORS) if c not in forbidden]
                    if avail:
                        c = rng.choice(avail)
                        old = len(state)
                        state.append([u, v, c])
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edge_set.add((u, v))
                        if _validate(state):
                            accepted = True
                            if len(state) > best_len:
                                best_len = len(state)
                                best_state = [e[:] for e in state]
                        else:
                            state.pop()
                            adj[u][v] = -1
                            adj[v][u] = -1
                            neighbors[u].discard(v)
                            neighbors[v].discard(u)
                            edge_set.discard((u, v))
                            if rng.random() < temp:
                                accepted = True
        elif move_type < 0.65:
            # Remove an edge (always valid) to escape a local optimum.
            if state:
                idx = rng.randrange(len(state))
                u, v, c = state[idx]
                state.pop(idx)
                adj[u][v] = -1
                adj[v][u] = -1
                neighbors[u].discard(v)
                neighbors[v].discard(u)
                edge_set.discard((u, v))
                accepted = True
        else:
            # Recolour an existing edge.
            if state:
                idx = rng.randrange(len(state))
                u, v, old_c = state[idx]
                adj[u][v] = -1
                adj[v][u] = -1
                neighbors[u].discard(v)
                neighbors[v].discard(u)
                possible, forbidden = _forbidden_for(adj, neighbors, u, v)
                if possible:
                    avail = [c for c in range(_NUM_COLORS) if c not in forbidden and c != old_c]
                    if avail:
                        new_c = rng.choice(avail)
                        state[idx][2] = new_c
                        adj[u][v] = new_c
                        adj[v][u] = new_c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        if _validate(state):
                            accepted = True
                        else:
                            state[idx][2] = old_c
                            adj[u][v] = old_c
                            adj[v][u] = old_c
                            if rng.random() < temp:
                                accepted = True
                            else:
                                state[idx][2] = old_c
                                adj[u][v] = old_c
                                adj[v][u] = old_c
                                neighbors[u].add(v)
                                neighbors[v].add(u)
                else:
                    adj[u][v] = old_c
                    adj[v][u] = old_c
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        if not accepted:
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        else:
            temp = min(temp * 1.02, 1.0)

    return best_state


def _final_polish(edges):
    """Greedily add any remaining missing edges that fit."""
    if _time_left() < 3.0:
        return edges
    adj, neighbors = _build_adj(edges)
    existing = set((min(u, v), max(u, v)) for u, v, _ in edges)
    missing = [e for e in _ALL_EDGES if e not in existing]
    rng = random.Random(99991)
    rng.shuffle(missing)
    for (u, v) in missing:
        if _time_left() < 3.0:
            break
        possible, forbidden = _forbidden_for(adj, neighbors, u, v)
        if not possible:
            continue
        avail = [c for c in range(_NUM_COLORS) if c not in forbidden]
        if not avail:
            continue
        c = avail[0]
        edges.append([u, v, c])
        adj[u][v] = c
        adj[v][u] = c
        neighbors[u].add(v)
        neighbors[v].add(u)
    return edges


def construct():
    rng = random.Random()
    best_edges = []
    best_count = 0

    def consider(edges):
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    # Phase 1: random greedy restarts.
    for ot in ('random', 'natural', 'reverse'):
        for i in range(40):
            if _time_left() < 40.0:
                break
            edges = _greedy_build(i * 7 + 1, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    # Phase 2: bipartite-seeded greedy.
    for i in range(15):
        if _time_left() < 40.0:
            break
        seed_edges = _bipartite_seed(rng)
        edges = _greedy_build(1000 + i, seed_edges, 'random')
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    # Phase 3: simulated annealing on the best construction.
    if _time_left() > 25.0:
        sa = _simulated_annealing(424242, best_edges)
        if len(sa) > best_count:
            best_count = len(sa)
            best_edges = sa

    # Phase 4: remove-rebuild restarts around the current best.
    it = 0
    while _time_left() > 15.0:
        it += 1
        if len(best_edges) <= 3:
            break
        k = random.randint(1, 3)
        to_remove = random.sample(best_edges, k)
        rset = set((min(u, v), max(u, v)) for u, v, _ in to_remove)
        base = [e for e in best_edges if (min(e[0], e[1]), max(e[0], e[1])) not in rset]
        new_edges = _greedy_build(it * 131 + random.randint(0, 100), base, 'random')
        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best_edges = new_edges

    # Final safety: guarantee validity.
    if not _validate(best_edges):
        for ot in ('random', 'natural', 'reverse'):
            for i in range(20):
                if _time_left() < 2.0:
                    break
                e = _greedy_build(5000 + i, None, ot)
                if _validate(e) and len(e) > best_count:
                    best_count = len(e)
                    best_edges = e
        if not _validate(best_edges):
            best_edges = [[0, v, 0] for v in range(1, _N)]

    # Final polish.
    best_edges = _final_polish(best_edges)
    return best_edges
