"""Decompositions of random graphs into few cycles and edges (Erdős-Gallai style)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

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

    def two_core(vertices, adj):
        """Compute the 2-core of the subgraph induced by vertices."""
        if not vertices:
            return {}
        
        deg = {v: len([nb for nb in adj.get(v, []) if nb in vertices]) for v in vertices}
        queue = deque([v for v in vertices if deg[v] < 2])
        removed = set()
        
        while queue:
            v = queue.popleft()
            if v in removed:
                continue
            removed.add(v)
            for nb in adj.get(v, []):
                if nb in vertices and nb not in removed:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        queue.append(nb)
        
        core_vertices = vertices - removed
        core_adj = defaultdict(set)
        for v in core_vertices:
            for nb in adj.get(v, []):
                if nb in core_vertices:
                    core_adj[v].add(nb)
        return dict(core_adj)

    def find_cycle_dfs(adj, start, time_limit, max_depth=50):
        """Find a cycle containing start using randomized DFS."""
        best_cycle = None
        best_len = 0

        def dfs(path, visited, current, depth):
            nonlocal best_cycle, best_len
            if time.time() > time_limit:
                return
            if depth >= max_depth:
                return

            neighbors = list(adj[current])
            random.shuffle(neighbors)

            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                elif nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(path, visited, nb, depth + 1)
                    path.pop()
                    visited.discard(nb)

        dfs([start], {start}, start, 0)
        return best_cycle

    def find_cycles_greedy(adj, time_limit, seed):
        """Find edge-disjoint cycles greedily, preferring longer cycles."""
        random.seed(seed)
        remaining_edges = set()
        for u in adj:
            for v in adj[u]:
                ek = edge_key(u, v)
                remaining_edges.add(ek)

        cycles = []
        remaining_adj = defaultdict(set)
        for u, v in remaining_edges:
            remaining_adj[u].add(v)
            remaining_adj[v].add(u)

        iterations = 0
        max_iterations = len(remaining_edges) + 10

        while remaining_edges and iterations < max_iterations:
            if time.time() > time_limit:
                break
            iterations += 1

            # Get 2-core to restrict search
            vertices_with_deg2 = [v for v in remaining_adj if len(remaining_adj[v]) >= 2]
            if not vertices_with_deg2:
                break

            # Try to find a cycle from a random starting vertex in the core
            random.shuffle(vertices_with_deg2)
            found_cycle = None

            # Try multiple starts
            for start in vertices_with_deg2[:5]:
                core = two_core(set(remaining_adj.keys()), remaining_adj)
                if start not in core or len(core[start]) < 2:
                    continue

                cycle = find_cycle_dfs(core, start, time_limit, max_depth=40)
                if cycle and len(cycle) >= 3:
                    found_cycle = cycle
                    break

            if found_cycle:
                cycles.append(found_cycle)
                # Remove cycle edges
                for i in range(len(found_cycle)):
                    u = found_cycle[i]
                    v = found_cycle[(i + 1) % len(found_cycle)]
                    ek = edge_key(u, v)
                    if ek in remaining_edges:
                        remaining_edges.discard(ek)
                    if v in remaining_adj.get(u, set()):
                        remaining_adj[u].discard(v)
                    if u in remaining_adj.get(v, set()):
                        remaining_adj[v].discard(u)
            else:
                break

        return cycles, remaining_edges, remaining_adj

    def decompose_graph(edges, time_limit, seed):
        """Decompose edges into cycles and single edges."""
        random.seed(seed)
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

        # Phase 1: Find cycles greedily
        cycles, remaining_edges, remaining_adj = find_cycles_greedy(
            adj, time_limit, seed
        )

        for cycle in cycles:
            pieces.append(cycle)

        # Update remaining
        remaining = set()
        for u, v in remaining_edges:
            remaining.add(edge_key(u, v))

        # Rebuild adj from remaining
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)

        # Phase 2: Try to find more cycles with different randomization
        if remaining and time.time() < time_limit - 5:
            extra_cycles, remaining2, adj2 = find_cycles_greedy(
                adj, time_limit, seed + 1000
            )
            for cycle in extra_cycles:
                pieces.append(cycle)
            remaining = set(remaining2)
            adj = adj2

        # Phase 3: Remaining edges as singles
        for u, v in remaining:
            pieces.append([u, v])

        return pieces

    def verify_decomposition(edges, pieces):
        """Verify that pieces cover all edges exactly once."""
        edge_set = set()
        for e in edges:
            edge_set.add(edge_key(e[0], e[1]))

        used_edges = set()
        for piece in pieces:
            if len(piece) == 2:
                ek = edge_key(piece[0], piece[1])
                if ek in used_edges:
                    return False
                used_edges.add(ek)
            elif len(piece) >= 3:
                # Check all vertices distinct
                if len(set(piece)) != len(piece):
                    return False
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    ek = edge_key(u, v)
                    if ek in used_edges:
                        return False
                    used_edges.add(ek)
            else:
                return False

        return used_edges == edge_set

    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]

    for spec in specs:
        if time.time() > deadline - 15:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)

        best_pieces = None
        best_count = float('inf')

        time_per_instance = min(40, deadline - time.time() - 15)
        if time_per_instance < 10:
            time_per_instance = 10

        # Run multiple trials with different random seeds
        num_trials = 15
        trial_time = time_per_instance / num_trials

        for t in range(num_trials):
            if time.time() > deadline - 10:
                break

            seed = 5000 + t * 7
            trial_deadline = time.time() + trial_time

            pieces = decompose_graph(edges, trial_deadline, seed)

            # Verify
            if not verify_decomposition(edges, pieces):
                continue

            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces

        if best_pieces is None:
            # Fallback: all edges as singles
            best_pieces = [list(e) for e in edges]

        results.append(best_pieces)

    return results
# EVOLVE-BLOCK-END
