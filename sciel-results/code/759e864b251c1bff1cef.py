import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def decompose_graph(edges, time_limit):
        if not edges:
            return []

        # Build adjacency structure
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(edges)
        pieces = []

        def update_adj():
            adj.clear()
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)

        def remove_cycle_edges(cycle):
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i + 1]
                if (u, v) in remaining_edges:
                    remaining_edges.remove((u, v))
                elif (v, u) in remaining_edges:
                    remaining_edges.remove((v, u))
            u, v = cycle[-1], cycle[0]
            if (u, v) in remaining_edges:
                remaining_edges.remove((u, v))
            elif (v, u) in remaining_edges:
                remaining_edges.remove((v, u))
            update_adj()

        def find_longest_cycle_dfs(start, remaining_edges, adj, time_limit):
            """DFS to find the longest cycle starting and ending at 'start'."""
            best_cycle = None
            best_len = 0

            # State: (current, path)
            # We want to avoid revisiting vertices in the same path
            visited = set()
            visited.add(start)
            path = [start]

            def dfs(current, path):
                nonlocal best_cycle, best_len
                if time.time() > time_limit:
                    return

                # Try to close the cycle
                if len(path) >= 3:
                    if (current, start) in remaining_edges or (start, current) in remaining_edges:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                        # Don't return immediately, keep looking for longer ones if possible
                        # But for speed, we might want to stop early if we found a very long one

                # Extend path
                for neighbor in sorted(adj[current]):
                    if time.time() > time_limit:
                        return
                    if neighbor == start:
                        continue  # Handled above
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()
                    visited.remove(neighbor)

            dfs(start, path)
            return best_cycle

        def find_best_cycle(time_limit):
            if not remaining_edges:
                return None

            # Heuristic: Start from high-degree vertices
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)

            best_cycle = None
            best_len = 0

            # Try to find the longest cycle
            # We limit the number of starts to avoid explosion
            max_starts = min(20, len(candidates))
            for start in candidates[:max_starts]:
                if time.time() > time_limit:
                    break

                cycle = find_longest_cycle_dfs(start, remaining_edges, adj, time_limit)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                    # If we found a Hamiltonian cycle, stop
                    if best_len == len(adj):
                        break

            if best_cycle:
                return best_cycle

            # Fallback: Find any cycle (e.g., triangle)
            # This is for robustness if DFS fails or times out without finding a cycle
            for u in list(adj.keys()):
                if len(adj[u]) < 2:
                    continue
                for v in list(adj[u]):
                    if v <= u:
                        continue
                    for w in list(adj[v]):
                        if w == u or w == v:
                            continue
                        if (u, w) in remaining_edges or (w, u) in remaining_edges:
                            # Triangle u-v-w-u
                            return [u, v, w]

            return None

        # Main loop: extract cycles
        while remaining_edges:
            if time.time() > time_limit:
                # Time limit reached. Take all remaining edges as pieces.
                for e in list(remaining_edges):
                    pieces.append(list(e))
                break

            cycle = find_best_cycle(time_limit)

            if cycle:
                pieces.append(cycle)
                remove_cycle_edges(cycle)
            else:
                # No cycle found. The remaining graph is a forest (or edges).
                # We must take each edge as a separate piece.
                # Optimization: If the remaining graph is disconnected, we can process components,
                # but since no cycles exist, every edge is a piece.
                for e in list(remaining_edges):
                    pieces.append(list(e))
                break

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time based on graph size
        # N=20: ~200 edges, easy
        # N=25: ~300 edges, medium
        # N=30: ~400 edges, hard
        # We give more time to larger graphs.
        if i == 0:
            time_alloc = 30.0
        elif i == 1:
            time_alloc = 40.0
        else:
            time_alloc = 60.0

        current_limit = time.time() + time_alloc
        global_limit = deadline
        if current_limit > global_limit:
            current_limit = global_limit

        # Ensure we have at least some time
        if current_limit < time.time() + 1.0:
            current_limit = time.time() + 1.0

        best_pieces = decompose_graph(edges, current_limit)
        results.append(best_pieces)

        # If we are running out of time globally, fallback for the rest
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                # Trivial decomposition: each edge is a piece
                results.append([list(e) for e in graphs[j]])
            break

    return results

if __name__ == "__main__":
    res = construct()
    # Quick sanity check
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        # Re-generate to check
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)

        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                # Check distinct vertices
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                # Check edges
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k + 1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break

        if valid and used_edges != edge_set:
            valid = False

        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
