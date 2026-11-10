"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def enumerate_cycles(edges_set, n, time_budget, max_cycles=2000, max_depth=12, num_starts=30):
        """
        Enumerate a diverse set of cycles using randomized DFS from multiple starts.
        Returns list of cycles, each cycle is a frozenset of edge_keys plus the vertex list.
        """
        adj = defaultdict(set)
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)

        cycles = []  # list of (vertex_list, frozenset_of_edge_keys)
        seen_edge_sets = set()

        start_time = time.time()
        rng = random.Random(42)

        # Get vertices with degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            return cycles

        # Try multiple random orderings of starts
        for trial in range(3):
            if time.time() - start_time > time_budget:
                break
            if len(cycles) >= max_cycles:
                break
            shuffled_starts = candidates[:]
            rng.shuffle(shuffled_starts)

            for start in shuffled_starts[:num_starts]:
                if time.time() - start_time > time_budget:
                    break
                if len(cycles) >= max_cycles:
                    break

                # Iterative DFS with randomized neighbor order
                stack = [(start, [start], {start}, 0)]
                found_in_this_start = 0
                max_found_per_start = 3  # limit cycles per start for diversity

                while stack and found_in_this_start < max_found_per_start:
                    if time.time() - start_time > time_budget:
                        break
                    if len(cycles) >= max_cycles:
                        break

                    current, path, visited, depth = stack.pop()
                    if depth >= max_depth:
                        continue

                    neighbors = list(adj[current])
                    rng.shuffle(neighbors)

                    for nb in neighbors:
                        if nb == start and len(path) >= 3:
                            # Found a cycle
                            cycle_verts = path[:]
                            cycle_edges = frozenset(
                                edge_key(cycle_verts[i], cycle_verts[(i + 1) % len(cycle_verts)])
                                for i in range(len(cycle_verts))
                            )
                            if cycle_edges not in seen_edge_sets:
                                seen_edge_sets.add(cycle_edges)
                                cycles.append((cycle_verts, cycle_edges))
                                found_in_this_start += 1
                                break
                        elif nb not in visited:
                            stack.append((nb, path + [nb], visited | {nb}, depth + 1))

        return cycles

    def solve_with_ilp(edges, time_budget):
        """
        ILP-based decomposition:
        1. Enumerate candidate cycles.
        2. Formulate set-partitioning: max sum x_i s.t. each edge used <= 1.
        3. Solve with scipy milp.
        4. Remaining edges become single-edge pieces.
        """
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np

        remaining = set()
        for e in edges:
            ek = edge_key(e[0], e[1])
            remaining.add(ek)

        if not remaining:
            return []

        edge_list = sorted(remaining)
        m = len(edge_list)
        edge_index = {e: i for i, e in enumerate(edge_list)}

        # Enumerate cycles
        cycles = enumerate_cycles(remaining, max(edge_list, key=lambda x: x[1])[1] + 1,
                                   time_budget * 0.4, max_cycles=1500, max_depth=12, num_starts=25)

        if not cycles:
            # No cycles found, return all singles
            return [list(e) for e in edge_list]

        # Build incidence matrix: A[i_cycle, j_edge] = 1 if cycle i uses edge j
        k = len(cycles)
        A = np.zeros((k, m), dtype=np.float64)
        for i, (verts, ekeys) in enumerate(cycles):
            for ek in ekeys:
                if ek in edge_index:
                    A[i, edge_index[ek]] = 1.0

        # Objective: maximize sum(x_i)  =>  minimize -sum(x_i)
        c = -np.ones(k)

        # Constraints: A @ x <= 1 (each edge used at most once)
        # x_i in {0, 1}
        lb = np.zeros(k)
        ub = np.ones(k)
        constraints = [LinearConstraint(A, -np.inf, 1.0)]

        start_time = time.time()
        try:
            res = milp(c=c, constraints=constraints, integrality=np.ones(k),
                       bounds=Bounds(lb, ub), options={'time_limit': max(1, time_budget * 0.5)})
        except Exception:
            res = None

        # Extract solution
        used_edges = set()
        pieces = []

        if res is not None and res.status == 0 and res.x is not None:
            for i in range(k):
                if res.x[i] > 0.5:
                    verts, ekeys = cycles[i]
                    # Verify all edges are still available
                    valid = True
                    for ek in ekeys:
                        if ek in used_edges:
                            valid = False
                            break
                    if valid:
                        for ek in ekeys:
                            used_edges.add(ek)
                        pieces.append(verts[:])
        else:
            # Fallback: greedy from enumerated cycles
            # Sort cycles by length descending, pick greedily
            sorted_cycles = sorted(cycles, key=lambda x: len(x[0]), reverse=True)
            for verts, ekeys in sorted_cycles:
                if time.time() - start_time > time_budget * 0.8:
                    break
                valid = True
                for ek in ekeys:
                    if ek in used_edges:
                        valid = False
                        break
                if valid:
                    for ek in ekeys:
                        used_edges.add(ek)
                    pieces.append(verts[:])

        # Remaining edges as singles
        for ek in edge_list:
            if ek not in used_edges:
                pieces.append([ek[0], ek[1]])

        return pieces

    def solve_with_greedy_fallback(edges, time_budget):
        """Simple greedy fallback if ILP is too slow or fails."""
        remaining = set()
        for e in edges:
            remaining.add(edge_key(e[0], e[1]))

        if not remaining:
            return []

        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []
        used_edges = set()

        def commit_cycle(cycle_vertices):
            for i in range(len(cycle_vertices)):
                u = cycle_vertices[i]
                v = cycle_vertices[(i + 1) % len(cycle_vertices)]
                ek = edge_key(u, v)
                used_edges.add(ek)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle_vertices[:])

        def commit_single_edge(u, v):
            ek = edge_key(u, v)
            used_edges.add(ek)
            remaining.discard(ek)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])

        # Greedy: find cycles via randomized DFS, prefer longer ones
        progress = True
        while progress and remaining:
            if time.time() > time.time() + time_budget:  # won't happen but guard
                break
            if not remaining:
                break

            progress = False
            best_cycle = None
            best_len = 0

            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                break
            random.shuffle(candidates)

            for start in candidates[:15]:
                if not remaining:
                    break
                # Quick DFS for a cycle
                path = [start]
                visited = {start}

                def dfs(current, depth, max_depth=10):
                    if depth >= max_depth:
                        return None
                    for nb in list(adj[current]):
                        if nb == start and len(path) >= 3:
                            return path[:]
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            result = dfs(nb, depth + 1, max_depth)
                            if result is not None:
                                return result
                            path.pop()
                            visited.discard(nb)
                    return None

                cycle = dfs(start, 0)
                if cycle is not None and len(cycle) > best_len:
                    best_cycle = cycle
                    best_len = len(cycle)
                    if best_len >= 8:
                        break

            if best_cycle is not None:
                commit_cycle(best_cycle)
                progress = True
            else:
                break

        for u, v in list(remaining):
            if (u, v) not in used_edges:
                commit_single_edge(u, v)

        return pieces

    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]

    for spec in specs:
        if time.time() > deadline - 15:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        time_for_this = min(40, deadline - time.time() - 10)
        if time_for_this < 3:
            time_for_this = 3

        # Try ILP approach
        best_pieces = None
        best_count = float('inf')

        # First attempt: ILP
        try:
            pieces = solve_with_ilp(edges, time_for_this)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
        except Exception:
            pass

        # If time remains, try a few greedy restarts for comparison
        if time.time() < deadline - 10 and best_count > 0:
            for seed in range(5):
                if time.time() > deadline - 10:
                    break
                random.seed(seed)
                try:
                    pieces = solve_with_greedy_fallback(edges, 5)
                    count = len(pieces)
                    if count < best_count:
                        best_count = count
                        best_pieces = pieces
                except Exception:
                    pass

        if best_pieces is None:
            best_pieces = [list(e) for e in edges]

        results.append(best_pieces)

    return results
# EVOLVE-BLOCK-END
