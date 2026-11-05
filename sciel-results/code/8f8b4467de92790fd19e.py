"""Construct dense 4-cycle-rainbow edge-coloured graph on 20 vertices.

Combines:
  * fast greedy builder with forbidden-colour computation
  * bipartite seeding to avoid C4-free local optima
  * local search with re-color, remove+re-greedy, and color-class swap moves
  * fast incremental validity check
  * NumPy-based fast validation
"""
import time
import random
from itertools import combinations
import numpy as np

N = 20
N_COLORS = 20
DEADLINE = 132.0

ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N)]


def _validate(edges):
    """Check that every 4-cycle has four distinct colours."""
    if not edges:
        return True
    adj = np.full((N, N), -1, dtype=np.int8)
    for u, v, c in edges:
        adj[u, v] = c
        adj[v, u] = c

    for a in range(N):
        for b in range(a + 1, N):
            if adj[a, b] == -1:
                continue
            for c in range(b + 1, N):
                if adj[a, c] == -1 or adj[b, c] == -1:
                    continue
                for d in range(c + 1, N):
                    # Check cycle a-b-c-d-a
                    if adj[b, c] != -1 and adj[c, d] != -1 and adj[d, a] != -1:
                        cols = (adj[a, b], adj[b, c], adj[c, d], adj[d, a])
                        if len(set(cols)) != 4:
                            return False
                    # Check cycle a-b-d-c-a
                    if adj[a, d] != -1 and adj[b, d] != -1 and adj[c, d] != -1:
                        cols = (adj[a, b], adj[b, d], adj[d, c], adj[c, a])
                        if len(set(cols)) != 4:
                            return False
                    # Check cycle a-c-b-d-a
                    if adj[a, c] != -1 and adj[c, b] != -1 and adj[b, d] != -1 and adj[d, a] != -1:
                        cols = (adj[a, c], adj[c, b], adj[b, d], adj[d, a])
                        if len(set(cols)) != 4:
                            return False
    return True


def _fast_check_after_change(adj, neighbors, changed_edge):
    """Check only the 4-cycles that involve changed_edge=(u,v)."""
    u, v = changed_edge
    for x in list(neighbors[u]):
        if x == v:
            continue
        cx = adj[u][x]
        for y in list(neighbors[v]):
            if y == u or y == x:
                continue
            if adj[x][y] == -1:
                continue
            cols = (cx, adj[x][y], adj[y][v], adj[v][u])
            if len(set(cols)) != 4:
                return False
    return True


def _check_color_class_swap(adj, neighbors, color_a, color_b):
    """Check if swapping colors color_a and color_b preserves rainbow-4-cycle property."""
    edges_a = []
    edges_b = []
    for u in range(N):
        for v in range(u + 1, N):
            c = adj[u][v]
            if c == color_a:
                edges_a.append((u, v))
            elif c == color_b:
                edges_b.append((u, v))

    if not edges_a or not edges_b:
        return True

    affected = set()
    for u, v in edges_a:
        affected.add((u, v))
        affected.add((v, u))
    for u, v in edges_b:
        affected.add((u, v))
        affected.add((v, u))

    verts = list(range(N))
    for a, b, c, d in combinations(verts, 4):
        cycles = [
            [(a, b), (b, c), (c, d), (d, a)],
            [(a, b), (b, d), (d, c), (c, a)],
            [(a, c), (c, b), (b, d), (d, a)],
        ]
        for cyc in cycles:
            valid_cycle = True
            for (u, v) in cyc:
                if adj[u][v] == -1:
                    valid_cycle = False
                    break
            if not valid_cycle:
                continue
            has_affected = False
            for (u, v) in cyc:
                if (u, v) in affected or (v, u) in affected:
                    has_affected = True
                    break
            if not has_affected:
                continue
            cols = []
            for (u, v) in cyc:
                c = adj[u][v]
                if c == color_a:
                    c = color_b
                elif c == color_b:
                    c = color_a
                cols.append(c)
            if len(set(cols)) != 4:
                return False
    return True


def construct():
    start = time.time()

    def elapsed():
        return time.time() - start

    def deadline_ok():
        return elapsed() < DEADLINE

    def greedy_build(seed, initial_edges=None, order_type='random'):
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
            existing_set = set((min(u, v), max(u, v)) for u, v, _ in edges)
            remaining = [e for e in ALL_PAIRS if e not in existing_set]
        else:
            remaining = ALL_PAIRS[:]

        if order_type == 'random':
            rng.shuffle(remaining)
        elif order_type == 'reverse':
            remaining.reverse()

        for (u, v) in remaining:
            if v in neighbors[u]:
                continue

            impossible = False
            forbidden = set()

            for a in list(neighbors[u]):
                if a == v:
                    continue
                ca = adj[u][a]
                for b in list(neighbors[v]):
                    if b == u or b == a:
                        continue
                    cb = adj[a][b]
                    if cb == -1:
                        continue
                    cv = adj[b][v]
                    if ca == cb or cb == cv or ca == cv:
                        impossible = True
                        break
                    forbidden.add(ca)
                    forbidden.add(cb)
                    forbidden.add(cv)
                if impossible:
                    break

            if not impossible:
                if forbidden:
                    candidates = [c for c in range(N_COLORS) if c not in forbidden]
                    if candidates:
                        c = rng.choice(candidates)
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                else:
                    c = rng.randrange(N_COLORS)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])

        return edges

    def bipartite_seed(seed, density=0.3):
        """Create a bipartite graph seed to avoid C4-free local optima."""
        rng = random.Random(seed)
        group_a = list(range(10))
        group_b = list(range(10, 20))
        edges = []
        for a in group_a:
            for b in group_b:
                if rng.random() < density:
                    c = rng.randrange(N_COLORS)
                    edges.append([a, b, c])
        return edges

    best_edges = None
    best_count = 0

    # ---- Phase 1: multiple greedy restarts ----
    # First try bipartite seeds
    for i in range(30):
        if not deadline_ok():
            break
        seed_edges = bipartite_seed(i * 31 + 7, density=0.25 + 0.01 * (i % 5))
        edges = greedy_build(i * 100 + 1, seed_edges, 'random')
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    # Then try standard greedy restarts
    order_types = ['random', 'natural', 'reverse']
    for ot in order_types:
        for i in range(40):
            if not deadline_ok():
                break
            edges = greedy_build(i * 7 + 13, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    if best_edges is None:
        best_edges = [[0, v, v % N_COLORS] for v in range(1, N)]
        best_count = len(best_edges)

    # ---- Phase 2: local search ----
    current_best = best_edges[:]
    refine_iter = 0
    max_refine = 8000

    while deadline_ok() and refine_iter < max_refine:
        refine_iter += 1

        adj = [[-1] * N for _ in range(N)]
        neighbors = [set() for _ in range(N)]
        for u, v, c in current_best:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)

        move = random.random()
        candidate = None

        if move < 0.40 and len(current_best) > 0:
            # ---- Re-color move ----
            u, v, c0 = random.choice(current_best)
            tried = random.sample([c for c in range(N_COLORS) if c != c0],
                                  k=min(8, N_COLORS - 1))
            for nc in tried:
                adj[u][v] = nc
                adj[v][u] = nc
                if not _fast_check_after_change(adj, neighbors, (u, v)):
                    adj[u][v] = c0
                    adj[v][u] = c0
                    continue
                cand = [e[:] for e in current_best]
                for e in cand:
                    if (min(e[0], e[1]), max(e[0], e[1])) == (min(u, v), max(u, v)):
                        e[2] = nc
                        break
                seed = refine_iter * 1000 + random.randint(0, 1000)
                new_edges = greedy_build(seed, cand, 'random')
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges
                    current_best = new_edges[:]
                    candidate = new_edges
                    break
        elif move < 0.55:
            # ---- Color-class swap move ----
            c1, c2 = random.sample(range(N_COLORS), 2)
            if _check_color_class_swap(adj, neighbors, c1, c2):
                cand = []
                for e in current_best:
                    if e[2] == c1:
                        cand.append([e[0], e[1], c2])
                    elif e[2] == c2:
                        cand.append([e[0], e[1], c1])
                    else:
                        cand.append([e[0], e[1], e[2]])
                seed = refine_iter * 1000 + random.randint(0, 1000)
                new_edges = greedy_build(seed, cand, 'random')
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges
                    current_best = new_edges[:]
                    candidate = new_edges
        elif move < 0.70:
            # ---- Bipartite seed + re-greedy ----
            seed_edges = bipartite_seed(refine_iter * 17 + 3, density=0.2 + 0.05 * random.random())
            # Keep some existing edges to maintain density
            if len(current_best) > 10:
                keep = random.sample(current_best, min(15, len(current_best)))
                keep_set = set((min(u, v), max(u, v)) for u, v, _ in keep)
                combined = keep + [e for e in seed_edges if (min(e[0], e[1]), max(e[0], e[1])) not in keep_set]
            else:
                combined = seed_edges
            seed = refine_iter * 1000 + random.randint(0, 1000)
            new_edges = greedy_build(seed, combined, 'random')
            if len(new_edges) > best_count:
                best_count = len(new_edges)
                best_edges = new_edges
                current_best = new_edges[:]
                candidate = new_edges
        else:
            # ---- Remove + re-greedy move ----
            k = random.randint(1, 6)
            if len(current_best) > k:
                edges_to_remove = random.sample(current_best, k)
                remove_set = set((min(u, v), max(u, v)) for u, v, _ in edges_to_remove)
                base_edges = [e for e in current_best
                              if (min(e[0], e[1]), max(e[0], e[1])) not in remove_set]
                seed = refine_iter * 1000 + random.randint(0, 1000)
                new_edges = greedy_build(seed, base_edges, 'random')
                if len(new_edges) > best_count:
                    best_count = len(new_edges)
                    best_edges = new_edges
                    current_best = new_edges[:]
                    candidate = new_edges

        # Periodic validation
        if refine_iter % 500 == 0 and best_edges is not None:
            if not _validate(best_edges):
                # Reset to a safe state
                current_best = [[0, v, v % N_COLORS] for v in range(1, N)]
                best_count = len(current_best)
                best_edges = current_best[:]

    # Final validation
    if best_edges is not None and not _validate(best_edges):
        return [[0, v, v % N_COLORS] for v in range(1, N)]

    return best_edges if best_edges is not None else [[0, v, v % N_COLORS] for v in range(1, N)]


if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
    print(f"Valid: {_validate(result)}")
