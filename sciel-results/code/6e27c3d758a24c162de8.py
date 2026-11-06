import time
import random
import math
import sys

N = 20
C = 20
DEADLINE = 145.0


def _validate(edges):
    """Return True iff every 4-cycle is rainbow."""
    if not edges:
        return True
    adj = {}
    neigh = [set() for _ in range(N)]
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
        neigh[u].add(v)
        neigh[v].add(u)

    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = neigh[u] & neigh[v]
            common.discard(u)
            common.discard(v)
            common_list = list(common)
            for i in range(len(common_list)):
                x = common_list[i]
                for j in range(i + 1, len(common_list)):
                    y = common_list[j]
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
    return True


def _check_add(adj, neigh, u, v, c):
    """Check if adding edge (u,v) with color c maintains rainbow 4-cycles."""
    for x in neigh[v]:
        if x == u:
            continue
        for y in neigh[u]:
            if y == v or y == x:
                continue
            if y not in neigh[x]:
                continue
            if c == adj[(v, x)] or c == adj[(x, y)] or c == adj[(y, u)]:
                return False
            if adj[(v, x)] == adj[(x, y)] or adj[(v, x)] == adj[(y, u)]:
                return False
            if adj[(x, y)] == adj[(y, u)]:
                return False
    return True


def _greedy_with_shuffled_colors(order, rng, adj=None, neigh=None, color_at=None, edges=None):
    """Greedy with random color ordering among available colors."""
    if adj is None:
        adj = {}
    if neigh is None:
        neigh = [set() for _ in range(N)]
    if color_at is None:
        color_at = [[None]*N for _ in range(N)]
    if edges is None:
        edges = []

    for (u, v) in order:
        if (u, v) in adj or (v, u) in adj:
            continue
        used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
        used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
        available = [c for c in range(C) if c not in used_u and c not in used_v]
        if not available:
            continue
        rng.shuffle(available)
        for c in available:
            if _check_add(adj, neigh, u, v, c):
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                color_at[u][v] = c
                color_at[v][u] = c
                edges.append([u, v, c])
                break
    return edges, adj, neigh, color_at


def _try_add_edges(edges, adj, neigh, color_at, rng, time_budget):
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    existing = set()
    for u, v, c in edges:
        existing.add((min(u,v), max(u,v)))
    candidates = [(u,v) for (u,v) in all_pairs if (u,v) not in existing]
    rng.shuffle(candidates)
    for (u, v) in candidates:
        if time.time() > time_budget:
            break
        used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
        used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
        available = [c for c in range(C) if c not in used_u and c not in used_v]
        if not available:
            continue
        rng.shuffle(available)
        for c in available:
            if _check_add(adj, neigh, u, v, c):
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                color_at[u][v] = c
                color_at[v][u] = c
                edges.append([u, v, c])
                break
    return edges


def _backtracking_build(rng, time_budget, max_edges_target=None):
    """Backtracking with most-constrained-first edge selection."""
    adj = {}
    neigh = [set() for _ in range(N)]
    color_at = [[None]*N for _ in range(N)]
    edges = []
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]

    # Order edges by "difficulty": pairs with many common neighbors first
    # (more constraints to satisfy)
    def pair_difficulty(u, v):
        common = neigh[u] & neigh[v]
        common.discard(u)
        common.discard(v)
        return len(common)

    # Use a priority approach: repeatedly pick the most-constrained addable edge
    # but with randomness to escape local optima
    attempts = 0
    max_attempts = 5000
    stuck_count = 0

    while len(edges) < 190 and time.time() < time_budget and attempts < max_attempts:
        attempts += 1
        existing = set()
        for u, v, c in edges:
            existing.add((min(u,v), max(u,v)))
        candidates = [(u,v) for (u,v) in all_pairs if (u,v) not in existing]
        if not candidates:
            break

        # Score candidates: prefer those with fewer common neighbors (easier)
        # but also consider degree balance
        scored = []
        for (u, v) in candidates:
            common = neigh[u] & neigh[v]
            common.discard(u)
            common.discard(v)
            deg_sum = len(neigh[u]) + len(neigh[v])
            # We want edges that are easy to add (few common neighbors)
            score = len(common) * 10 + deg_sum * 0.1
            scored.append((score, u, v))
        scored.sort(key=lambda x: x[0])

        # Pick from top-K with randomness
        K = min(8, len(scored))
        if rng.random() < 0.3:
            # Sometimes pick a random one to diversify
            u, v = rng.choice(candidates)
        else:
            u, v = scored[rng.randrange(K)][1], scored[rng.randrange(K)][2]

        used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
        used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
        available = [c for c in range(C) if c not in used_u and c not in used_v]
        if not available:
            stuck_count += 1
            continue
        rng.shuffle(available)
        added = False
        for c in available:
            if _check_add(adj, neigh, u, v, c):
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                color_at[u][v] = c
                color_at[v][u] = c
                edges.append([u, v, c])
                added = True
                stuck_count = 0
                break

        if not added:
            stuck_count += 1
            if stuck_count > 200:
                # Restart with a fresh random seed
                break

    return edges


def _backtracking_with_removal(rng, time_budget):
    """Backtracking that can remove edges to unblock others."""
    adj = {}
    neigh = [set() for _ in range(N)]
    color_at = [[None]*N for _ in range(N)]
    edges = []
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]

    for _ in range(200):
        if time.time() > time_budget:
            break
        existing = set()
        for u, v, c in edges:
            existing.add((min(u,v), max(u,v)))
        candidates = [(u,v) for (u,v) in all_pairs if (u,v) not in existing]
        if not candidates:
            break

        rng.shuffle(candidates)
        added_any = False
        for (u, v) in candidates:
            if time.time() > time_budget:
                break
            used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
            used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
            available = [c for c in range(C) if c not in used_u and c not in used_v]
            if not available:
                continue
            rng.shuffle(available)
            for c in available:
                if _check_add(adj, neigh, u, v, c):
                    adj[(u, v)] = c
                    adj[(v, u)] = c
                    neigh[u].add(v)
                    neigh[v].add(u)
                    color_at[u][v] = c
                    color_at[v][u] = c
                    edges.append([u, v, c])
                    added_any = True
                    break

        if not added_any and edges:
            # Remove a random edge and try again
            idx = rng.randrange(len(edges))
            a, b, c = edges[idx]
            edges.pop(idx)
            del adj[(a, b)]
            del adj[(b, a)]
            neigh[a].discard(b)
            neigh[b].discard(a)
            color_at[a][b] = None
            color_at[b][a] = None

    return edges


def _simulated_annealing(edges, adj, neigh, color_at, rng, time_budget):
    """Simulated annealing on the colored edge set."""
    current_edges = edges[:]
    current_adj = {k: v for k, v in adj.items()}
    current_neigh = [set(s) for s in neigh]
    current_color_at = [[c for c in row] for row in color_at]
    best_edges = current_edges[:]
    best_len = len(current_edges)

    start_temp = 2.0
    end_temp = 0.05
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    n_steps = 0
    max_steps = 50000

    while time.time() < time_budget and n_steps < max_steps:
        n_steps += 1
        temp = start_temp * (end_temp / start_temp) ** (n_steps / max_steps)

        existing = set()
        for u, v, c in current_edges:
            existing.add((min(u,v), max(u,v)))
        candidates = [(u,v) for (u,v) in all_pairs if (u,v) not in existing]

        if candidates and rng.random() < 0.6:
            # Try adding an edge
            u, v = rng.choice(candidates)
            used_u = set(current_color_at[u][w] for w in current_neigh[u] if current_color_at[u][w] is not None)
            used_v = set(current_color_at[v][w] for w in current_neigh[v] if current_color_at[v][w] is not None)
            available = [c for c in range(C) if c not in used_u and c not in used_v]
            if available:
                rng.shuffle(available)
                for c in available:
                    if _check_add(current_adj, current_neigh, u, v, c):
                        current_adj[(u, v)] = c
                        current_adj[(v, u)] = c
                        current_neigh[u].add(v)
                        current_neigh[v].add(u)
                        current_color_at[u][v] = c
                        current_color_at[v][u] = c
                        current_edges.append([u, v, c])
                        if len(current_edges) > best_len:
                            best_len = len(current_edges)
                            best_edges = current_edges[:]
                        break
        else:
            # Try removing an edge
            if current_edges:
                idx = rng.randrange(len(current_edges))
                a, b, c = current_edges[idx]
                current_edges.pop(idx)
                del current_adj[(a, b)]
                del current_adj[(b, a)]
                current_neigh[a].discard(b)
                current_neigh[b].discard(a)
                current_color_at[a][b] = None
                current_color_at[b][a] = None
                # Accept with probability based on temperature (always accept removals that allow future additions)
                # For now, just accept
                # Try to add edges back
                existing2 = set()
                for u, v, cc in current_edges:
                    existing2.add((min(u,v), max(u,v)))
                candidates2 = [(u2,v2) for (u2,v2) in all_pairs if (u2,v2) not in existing2]
                rng.shuffle(candidates2)
                for (u2, v2) in candidates2[:20]:
                    if time.time() > time_budget:
                        break
                    used_u = set(current_color_at[u2][w] for w in current_neigh[u2] if current_color_at[u2][w] is not None)
                    used_v = set(current_color_at[v2][w] for w in current_neigh[v2] if current_color_at[v2][w] is not None)
                    available2 = [c for c in range(C) if c not in used_u and c not in used_v]
                    if not available2:
                        continue
                    rng.shuffle(available2)
                    for c in available2:
                        if _check_add(current_adj, current_neigh, u2, v2, c):
                            current_adj[(u2, v2)] = c
                            current_adj[(v2, u2)] = c
                            current_neigh[u2].add(v2)
                            current_neigh[v2].add(u2)
                            current_color_at[u2][v2] = c
                            current_color_at[v2][u2] = c
                            current_edges.append([u2, v2, c])
                            break

    return best_edges


def _color_class_decomposition(rng, time_budget):
    """Build by assigning edges to color classes, ensuring each color class is a matching
    and 4-cycles don't repeat colors."""
    adj = {}
    neigh = [set() for _ in range(N)]
    color_at = [[None]*N for _ in range(N)]
    edges = []
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]

    # Process colors one at a time; for each color, build a matching
    color_order = list(range(C))
    rng.shuffle(color_order)

    for c in color_order:
        if time.time() > time_budget:
            break
        # Find all pairs that can take color c (no other edge at u or v uses c, and rainbow constraint)
        used_u = set(w for w in range(N) if color_at[0][w] == c)  # placeholder
        # Actually: for each vertex, check if any incident edge uses color c
        vertex_used_c = [False]*N
        for u in range(N):
            for w in neigh[u]:
                if color_at[u][w] == c:
                    vertex_used_c[u] = True

        # Candidate edges for color c: both endpoints don't use c
        candidates = [(u,v) for (u,v) in all_pairs if not vertex_used_c[u] and not vertex_used_c[v] and (u,v) not in adj and (v,u) not in adj]
        rng.shuffle(candidates)

        # Greedily add edges of color c (must form a matching)
        matched = [False]*N
        for (u, v) in candidates:
            if matched[u] or matched[v]:
                continue
            if _check_add(adj, neigh, u, v, c):
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                color_at[u][v] = c
                color_at[v][u] = c
                edges.append([u, v, c])
                matched[u] = True
                matched[v] = True

    return edges


def _turan_blowup(rng, time_budget):
    """Construct a Turan-type blow-up: partition vertices into parts, 
    connect parts in a C4-free pattern, color by part-pair."""
    # Use a 5-partition (Turan graph T(20,5) has parts of size 4)
    # T(20,5) is C4-free? No, Turan graphs can have C4s.
    # Instead, use a known C4-free dense structure.
    # Actually, let's just use random greedy which is simpler and works.
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    order = all_pairs[:]
    rng.shuffle(order)
    edges, adj, neigh, color_at = _greedy_with_shuffled_colors(order, rng)
    return edges


def construct():
    start = time.time()
    best_edges = []
    best_len = 0

    # Strategy 1: Backtracking with most-constrained-first
    if time.time() < DEADLINE - 40:
        for seed in [1, 2, 3]:
            if time.time() > DEADLINE - 40:
                break
            rng = random.Random(seed * 1000 + 7)
            edges = _backtracking_build(rng, DEADLINE - 40)
            if len(edges) > best_len:
                best_len = len(edges)
                best_edges = edges[:]

    # Strategy 2: Backtracking with removal
    if time.time() < DEADLINE - 30:
        for seed in [10, 11, 12]:
            if time.time() > DEADLINE - 30:
                break
            rng = random.Random(seed * 1000 + 13)
            edges = _backtracking_with_removal(rng, DEADLINE - 30)
            if len(edges) > best_len:
                best_len = len(edges)
                best_edges = edges[:]

    # Strategy 3: Color class decomposition
    if time.time() < DEADLINE - 20:
        for seed in [20, 21, 22]:
            if time.time() > DEADLINE - 20:
                break
            rng = random.Random(seed * 1000 + 17)
            edges = _color_class_decomposition(rng, DEADLINE - 20)
            if len(edges) > best_len:
                best_len = len(edges)
                best_edges = edges[:]

    # Strategy 4: Standard greedy
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    for seed in [30, 31, 32, 33]:
        if time.time() > DEADLINE - 10:
            break
        rng = random.Random(seed * 1000 + 19)
        order = all_pairs[:]
        rng.shuffle(order)
        edges, adj, neigh, color_at = _greedy_with_shuffled_colors(order, rng)
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges[:]

    # Local search / simulated annealing on best
    if best_edges and time.time() < DEADLINE - 5:
        adj = {}
        neigh = [set() for _ in range(N)]
        color_at = [[None]*N for _ in range(N)]
        for u, v, c in best_edges:
            adj[(u, v)] = c
            adj[(v, u)] = c
            neigh[u].add(v)
            neigh[v].add(u)
            color_at[u][v] = c
            color_at[v][u] = c

        rng = random.Random(12345)
        improved = _simulated_annealing(best_edges[:], adj, neigh, color_at, rng, DEADLINE - 2)
        if len(improved) > best_len:
            best_len = len(improved)
            best_edges = improved

    # Final attempt: try to add edges
    if best_edges and time.time() < DEADLINE - 1:
        adj = {}
        neigh = [set() for _ in range(N)]
        color_at = [[None]*N for _ in range(N)]
        for u, v, c in best_edges:
            adj[(u, v)] = c
            adj[(v, u)] = c
            neigh[u].add(v)
            neigh[v].add(u)
            color_at[u][v] = c
            color_at[v][u] = c
        rng = random.Random(99999)
        final = _try_add_edges(best_edges[:], adj, neigh, color_at, rng, DEADLINE - 0.5)
        if len(final) > best_len:
            best_edges = final

    if not best_edges:
        return [[0, v, v] for v in range(1, N)]

    return best_edges


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
