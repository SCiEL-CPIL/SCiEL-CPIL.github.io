import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def build_adj(edge_set):
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def try_longest_path_cycle(adj, remaining_edges, time_limit, max_depth=25):
        """Find a long cycle by trying to extend a path from each edge endpoint."""
        best_cycle = None
        best_len = 0

        vertices = sorted(adj.keys())
        if not vertices:
            return None

        # Try from each vertex, do a randomized DFS to find long paths
        rng_local = random.Random(42)
        shuffled_vertices = vertices[:]
        rng_local.shuffle(shuffled_vertices)

        for start in shuffled_vertices:
            if time.time() > time_limit:
                break
            if not adj[start]:
                continue

            # DFS to find longest path starting from 'start'
            best_path = [start]

            def dfs(node, path, visited):
                nonlocal best_path, best_len
                if time.time() > time_limit:
                    return
                if len(path) > len(best_path):
                    best_path = path[:]
                    best_len = len(path)

                if len(path) >= max_depth:
                    return

                neighbors = list(adj[node])
                rng_local.shuffle(neighbors)

                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, path, visited)
                        path.pop()
                        visited.discard(nb)
                        if time.time() > time_limit:
                            return

            dfs(start, [start], {start})

        # Check if best_path endpoints form a cycle
        if best_path and len(best_path) >= 3:
            u, v = best_path[0], best_path[-1]
            e = tuple(sorted((u, v)))
            if e in remaining_edges:
                return list(best_path)

        return None

    def greedy_decompose(edges, time_limit, strategy_seed=0):
        """Greedy decomposition: repeatedly find cycles and extract them."""
        remaining_edges = set(edges)
        pieces = []

        while remaining_edges:
            if time.time() > time_limit:
                break

            adj = build_adj(remaining_edges)

            # Strategy 1: Try to find a long cycle via longest path
            cycle = try_longest_path_cycle(adj, remaining_edges, time_limit, max_depth=30)

            # Strategy 2: If that fails, find any cycle via BFS
            if not cycle:
                cycle = find_any_cycle(adj, remaining_edges, time_limit)

            if cycle:
                pieces.append(cycle)
                # Remove cycle edges
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    e = tuple(sorted((u, v)))
                    remaining_edges.discard(e)
                u, v = cycle[-1], cycle[0]
                e = tuple(sorted((u, v)))
                remaining_edges.discard(e)
            else:
                # No more cycles, add all remaining as single edges
                for e in remaining_edges:
                    pieces.append(list(e))
                remaining_edges.clear()
                break

        return pieces

    def find_any_cycle(adj, remaining_edges, time_limit):
        """Find any cycle using BFS from each edge."""
        for u, v in list(remaining_edges):
            if time.time() > time_limit:
                break
            # Remove edge temporarily
            remaining_edges.discard((u, v))
            adj[u].discard(v)
            adj[v].discard(u)

            # BFS from u to v
            queue = deque([(u, [u])])
            visited = {u}
            found = None
            while queue:
                if time.time() > time_limit:
                    break
                node, path = queue.popleft()
                if node == v:
                    found = path
                    break
                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, path + [nb]))

            # Restore
            remaining_edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)

            if found and len(found) >= 3:
                return found

        return None

    def improve_decomposition(edges, pieces, time_limit):
        """Try to merge single edges into cycles or extend cycles."""
        # Build edge set from pieces
        edge_to_piece = {}
        for i, piece in enumerate(pieces):
            if len(piece) == 2:
                e = tuple(sorted(piece))
                edge_to_piece[e] = ('edge', i)
            else:
                for k in range(len(piece) - 1):
                    e = tuple(sorted((piece[k], piece[k+1])))
                    edge_to_piece[e] = ('cycle', i)
                e = tuple(sorted((piece[-1], piece[0])))
                edge_to_piece[e] = ('cycle', i)

        remaining = set(edges)
        for e in edge_to_piece:
            remaining.discard(e)

        # Try to find cycles among single edges + cycle edges
        # This is complex; instead, try local improvements
        # For each single edge, check if it can be part of a new cycle with other single edges
        single_edges = [e for e in edges if edge_to_piece.get(e, (None,))[0] == 'edge']

        # Try to find cycles in the subgraph of single edges
        if single_edges:
            single_adj = defaultdict(set)
            for u, v in single_edges:
                single_adj[u].add(v)
                single_adj[v].add(u)

            # Find cycles in single-edge subgraph
            used_in_new_cycle = set()
            for u, v in list(single_edges):
                if time.time() > time_limit:
                    break
                if u in used_in_new_cycle or v in used_in_new_cycle:
                    continue
                # Check if there's a path from u to v in single_adj avoiding direct edge
                single_adj[u].discard(v)
                single_adj[v].discard(u)

                queue = deque([(u, [u])])
                visited = {u}
                found = None
                while queue:
                    node, path = queue.popleft()
                    if node == v:
                        found = path
                        break
                    for nb in single_adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            queue.append((nb, path + [nb]))

                single_adj[u].add(v)
                single_adj[v].add(u)

                if found and len(found) >= 3:
                    # Found a cycle, replace single edges with this cycle
                    cycle_edges = set()
                    for k in range(len(found) - 1):
                        e = tuple(sorted((found[k], found[k+1])))
                        cycle_edges.add(e)
                    e = tuple(sorted((found[-1], found[0])))
                    cycle_edges.add(e)

                    # Remove the single-edge pieces
                    new_pieces = []
                    for piece in pieces:
                        if len(piece) == 2:
                            e = tuple(sorted(piece))
                            if e in cycle_edges:
                                continue
                        new_pieces.append(piece)
                    new_pieces.append(found)
                    return new_pieces

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        remaining_time = deadline - time.time()
        graphs_left = len(graphs) - i
        time_per_graph = max(2.0, remaining_time / graphs_left * 0.85)
        time_limit = time.time() + time_per_graph

        # Try multiple strategies and pick best
        best_pieces = None
        best_count = float('inf')

        # Strategy A: Standard greedy
        t_limit_a = time.time() + time_per_graph * 0.4
        pieces_a = greedy_decompose(edges, t_limit_a)
        if len(pieces_a) < best_count:
            best_count = len(pieces_a)
            best_pieces = pieces_a

        # Strategy B: With improvement
        if time.time() < deadline - 2.0:
            t_limit_b = time.time() + (time_per_graph * 0.3)
            pieces_b = greedy_decompose(edges, t_limit_b)
            pieces_b = improve_decomposition(edges, pieces_b, time.time() + time_per_graph * 0.2)
            if len(pieces_b) < best_count:
                best_count = len(pieces_b)
                best_pieces = pieces_b

        # Strategy C: Different randomization
        if time.time() < deadline - 2.0:
            t_limit_c = time.time() + (time_per_graph * 0.3)
            pieces_c = greedy_decompose(edges, t_limit_c)
            if len(pieces_c) < best_count:
                best_count = len(pieces_c)
                best_pieces = pieces_c

        if best_pieces is None:
            best_pieces = [list(e) for e in edges]

        results.append(best_pieces)

        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break

    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
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
                if len(set(piece)) != len(piece):
                    valid = False
                    break
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
    print(f"Total pieces: {sum(len(r) for r in res)}")
