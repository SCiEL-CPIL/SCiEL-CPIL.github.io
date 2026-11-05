import time
import random
from itertools import combinations

_START = time.time()
DEADLINE = 118.0  # 150s limit, ~32s safety margin

def _time_left():
    return DEADLINE - (time.time() - _START)

def _can_add_edge(adj, neighbors, u, v, num_colors):
    """Check if edge (u,v) can be added with some color, return the color or -1."""
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
                    return -1
                forbidden.add(ca)
                forbidden.add(cb)
                forbidden.add(cv)
    if len(forbidden) >= num_colors:
        return -1
    return min(c for c in range(num_colors) if c not in forbidden)

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
    elif order_type == 'degree_heuristic':
        # Try edges between low-degree vertices first (they have fewer constraints)
        deg = [len(neighbors[i]) for i in range(n)]
        def edge_key(e):
            return deg[e[0]] + deg[e[1]]
        remaining.sort(key=edge_key)

    for (u, v) in remaining:
        color = _can_add_edge(adj, neighbors, u, v, num_colors)
        if color >= 0:
            adj[u][v] = color
            adj[v][u] = color
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, color])
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
    for ot in ('random', 'natural', 'reverse', 'degree_heuristic'):
        for i in range(40):
            if _time_left() < 25.0:
                break
            edges = _greedy_build(i * 7 + 1, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    # Phase 2: bipartite-seeded greedy
    for i in range(15):
        if _time_left() < 25.0:
            break
        seed_edges = _bipartite_seed(rng)
        for ot in ('random', 'degree_heuristic'):
            edges = _greedy_build(1000 + i * 2 + (0 if ot == 'random' else 1), seed_edges, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    # Phase 3: local search with smarter moves
    current = best_edges[:]
    current_adj = [[-1] * 20 for _ in range(20)]
    current_neighbors = [set() for _ in range(20)]
    for u, v, c in current:
        current_adj[u][v] = c
        current_adj[v][u] = c
        current_neighbors[u].add(v)
        current_neighbors[v].add(u)

    def rebuild_adj(edges):
        adj = [[-1] * 20 for _ in range(20)]
        neighbors = [set() for _ in range(20)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        return adj, neighbors

    for it in range(3000):
        if _time_left() < 10.0:
            break
        move_type = random.random()
        
        if move_type < 0.3:
            # Single edge removal and re-add with different color
            if not current:
                continue
            u, v, c = random.choice(current)
            # Remove edge
            current_adj[u][v] = -1
            current_adj[v][u] = -1
            current_neighbors[u].discard(v)
            current_neighbors[v].discard(u)
            base = [e for e in current if not (e[0] == u and e[1] == v) and not (e[0] == v and e[1] == u)]
            # Try to re-add with different color
            new_color = _can_add_edge(current_adj, current_neighbors, u, v, 20)
            if new_color >= 0 and new_color != c:
                current_adj[u][v] = new_color
                current_adj[v][u] = new_color
                current_neighbors[u].add(v)
                current_neighbors[v].add(u)
                base.append([u, v, new_color])
                if len(base) > best_count:
                    best_count = len(base)
                    best_edges = base[:]
                    current = base[:]
            else:
                # Restore original
                current_adj[u][v] = c
                current_adj[v][u] = c
                current_neighbors[u].add(v)
                current_neighbors[v].add(u)
                current = base + [[u, v, c]]
        
        elif move_type < 0.6:
            # Remove 1-2 edges, try to add missing edges
            k = random.randint(1, 2)
            if len(current) <= k:
                continue
            to_remove = random.sample(current, k)
            rset = set((min(u, v), max(u, v)) for u, v, _ in to_remove)
            for u, v, c in to_remove:
                current_adj[u][v] = -1
                current_adj[v][u] = -1
                current_neighbors[u].discard(v)
                current_neighbors[v].discard(u)
            base = [e for e in current if (min(e[0], e[1]), max(e[0], e[1])) not in rset]
            # Try to add all missing edges
            all_edges = [(u, v) for u in range(20) for v in range(u + 1, 20)]
            existing = set((min(u, v), max(u, v)) for u, v, _ in base)
            missing = [e for e in all_edges if e not in existing]
            random.shuffle(missing)
            added = 0
            for u, v in missing:
                color = _can_add_edge(current_adj, current_neighbors, u, v, 20)
                if color >= 0:
                    current_adj[u][v] = color
                    current_adj[v][u] = color
                    current_neighbors[u].add(v)
                    current_neighbors[v].add(u)
                    base.append([u, v, color])
                    added += 1
            if len(base) > best_count:
                best_count = len(base)
                best_edges = base[:]
                current = base[:]
            else:
                # Restore removed edges
                for u, v, c in to_remove:
                    current_adj[u][v] = c
                    current_adj[v][u] = c
                    current_neighbors[u].add(v)
                    current_neighbors[v].add(u)
                current = base + to_remove
        
        else:
            # Try to add a single missing edge
            all_edges = [(u, v) for u in range(20) for v in range(u + 1, 20)]
            existing = set((min(u, v), max(u, v)) for u, v, _ in current)
            missing = [e for e in all_edges if e not in existing]
            if not missing:
                continue
            random.shuffle(missing)
            added = False
            for u, v in missing[:20]:  # Try up to 20 candidates
                color = _can_add_edge(current_adj, current_neighbors, u, v, 20)
                if color >= 0:
                    current_adj[u][v] = color
                    current_adj[v][u] = color
                    current_neighbors[u].add(v)
                    current_neighbors[v].add(u)
                    current.append([u, v, color])
                    added = True
                    break
            if added and len(current) > best_count:
                best_count = len(current)
                best_edges = current[:]

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
