import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135
    n = 20
    num_colors = 20

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)

    def check_validity_fast(edges):
        adj = np.full((n, n), -1, dtype=np.int8)
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
        for u in range(n):
            for v in range(n):
                if u == v or adj[u][v] == -1:
                    continue
                for w in range(n):
                    if w == u or w == v or adj[v][w] == -1:
                        continue
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if adj[w][x] == -1 or adj[x][u] == -1:
                            continue
                        c1 = adj[u][v]
                        c2 = adj[v][w]
                        c3 = adj[w][x]
                        c4 = adj[x][u]
                        if len({c1, c2, c3, c4}) != 4:
                            return False
        return True

    def greedy_build(seed, init_edges=None, time_limit=None):
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []

        if init_edges:
            for u, v, c in init_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            existing = set((min(u,v), max(u,v)) for u, v, c in edges)
            remaining = [e for e in all_edges if e not in existing]
        else:
            remaining = all_edges[:]

        rng.shuffle(remaining)

        for (u, v) in remaining:
            if time_limit and time.time() > time_limit:
                break

            forbidden = set()
            feasible = True

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
                            feasible = False
                            break
                        forbidden.add(ca)
                        forbidden.add(cb)
                        forbidden.add(cv)
                if not feasible:
                    break

            if feasible:
                if forbidden:
                    candidates = [c for c in range(num_colors) if c not in forbidden]
                    if candidates:
                        c = rng.choice(candidates)
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                else:
                    c = rng.randrange(num_colors)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])

        return edges

    def bipartite_seed(seed):
        rng = random.Random(seed)
        group_a = list(range(10))
        group_b = list(range(10, 20))

        edges = []
        for a in group_a:
            for b in group_b:
                if rng.random() < 0.3:
                    c = rng.randrange(num_colors)
                    edges.append([a, b, c])

        return edges

    def simulated_annealing(seed, time_limit):
        """SA over edge presence with matching-based color assignment."""
        rng = random.Random(seed)

        # State: for each edge, 0=absent, 1=present
        # Colors assigned greedily as matchings
        state = np.zeros(m, dtype=np.int8)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []

        def assign_colors():
            """Assign colors to present edges such that each color class is a matching."""
            nonlocal adj, neighbors, edges
            adj = [[-1] * n for _ in range(n)]
            neighbors = [set() for _ in range(n)]
            edges = []

            # Group present edges
            present = []
            for i, (u, v) in enumerate(all_edges):
                if state[i]:
                    present.append((u, v))

            # Shuffle for randomization
            rng.shuffle(present)

            # Greedy matching-based coloring
            color_used = [set() for _ in range(num_colors)]  # vertices used per color

            for u, v in present:
                placed = False
                # Try to place in an existing color matching
                for c in range(num_colors):
                    if u not in color_used[c] and v not in color_used[c]:
                        # Check 4-cycle constraint
                        valid = True
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
                                    if ca == c or c == cv or cv == cb or ca == cb:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if valid:
                            adj[u][v] = c
                            adj[v][u] = c
                            neighbors[u].add(v)
                            neighbors[v].add(u)
                            color_used[c].add(u)
                            color_used[c].add(v)
                            edges.append([u, v, c])
                            placed = True
                            break
                if not placed:
                    # Edge cannot be colored; mark absent
                    state[all_edges.index((u, v))] = 0

            return edges

        def count_violations():
            """Count 4-cycle color violations."""
            adj_local = [[-1] * n for _ in range(n)]
            for u, v, c in edges:
                adj_local[u][v] = c
                adj_local[v][u] = c

            violations = 0
            for u in range(n):
                for v in range(n):
                    if u == v or adj_local[u][v] == -1:
                        continue
                    for w in range(n):
                        if w == u or w == v or adj_local[v][w] == -1:
                            continue
                        for x in range(n):
                            if x == u or x == v or x == w:
                                continue
                            if adj_local[w][x] == -1 or adj_local[x][u] == -1:
                                continue
                            c1 = adj_local[u][v]
                            c2 = adj_local[v][w]
                            c3 = adj_local[w][x]
                            c4 = adj_local[x][u]
                            if len({c1, c2, c3, c4}) != 4:
                                violations += 1
            return violations

        # Initialize with random sparse graph
        for i in range(m):
            if rng.random() < 0.3:
                state[i] = 1

        edges = assign_colors()
        best_edges = edges[:]
        best_count = len(edges)
        current_violations = count_violations()

        T = 10.0
        T_min = 0.01
        cooling = 0.995

        while time.time() < time_limit and T > T_min:
            # Pick a random edge to toggle
            idx = rng.randint(0, m - 1)
            old_state = state[idx]
            state[idx] = 1 - old_state

            new_edges = assign_colors()
            new_violations = count_violations()
            new_count = len(new_edges)

            # Energy: -count + violations
            old_energy = -best_count + current_violations
            new_energy = -new_count + new_violations
            delta_e = new_energy - old_energy

            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                best_edges = new_edges[:]
                best_count = new_count
                current_violations = new_violations
            else:
                state[idx] = old_state

            T *= cooling

        # Final greedy fill
        if time.time() < time_limit:
            final = greedy_build(seed + 100000, best_edges, time_limit=time_limit)
            if len(final) > best_count:
                best_edges = final
                best_count = len(final)

        return best_edges

    best_count = 0
    best_edges = []

    # Phase 1: Greedy random restarts
    phase1_deadline = start_time + 30
    for i in range(300):
        if time.time() > phase1_deadline:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]

    # Phase 2: Bipartite-seeded greedy
    phase2_deadline = start_time + 50
    for i in range(100):
        if time.time() > phase2_deadline:
            break
        seed_edges = bipartite_seed(i)
        edges = greedy_build(10000 + i, seed_edges)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]

    # Phase 3: Simulated annealing
    phase3_deadline = start_time + 110
    sa_count = 0
    while time.time() < phase3_deadline and sa_count < 15:
        sa_time = min(phase3_deadline, time.time() + 3)
        sa_result = simulated_annealing(20000 + sa_count * 7, sa_time)
        if len(sa_result) > best_count:
            best_count = len(sa_result)
            best_edges = sa_result[:]
        sa_count += 1

    # Phase 4: Local search — remove k edges, refill
    current_best = best_edges[:]
    phase4_deadline = start_time + 130
    while time.time() < phase4_deadline:
        k = random.randint(1, 8)
        if len(current_best) <= k:
            break

        to_remove = random.sample(current_best, k)
        rem_set = set((min(u,v), max(u,v)) for u, v, c in to_remove)
        base = [e for e in current_best if (min(e[0], e[1]), max(e[0], e[1])) not in rem_set]

        seed = random.randint(0, 10**9)
        new_edges = greedy_build(seed, base, time_limit=phase4_deadline)

        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best_edges = new_edges[:]
            current_best = new_edges[:]

    # Final verification
    if best_edges and not check_validity_fast(best_edges):
        best_edges = []
        for i in range(100):
            candidate = greedy_build(50000 + i)
            if check_validity_fast(candidate):
                best_edges = candidate
                break
        if not best_edges:
            best_edges = [[0, v, v % 20] for v in range(1, 20)]

    if not best_edges:
        best_edges = [[0, v, v % 20] for v in range(1, 20)]

    return best_edges
