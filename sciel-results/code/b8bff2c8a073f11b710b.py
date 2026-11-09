import time
from collections import defaultdict, deque
from random import Random

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def greedy_path_decomposition(edges, max_time=None):
        """
        Decompose edges into edge-disjoint simple paths covering all edges.
        Uses a greedy approach: repeatedly find a long simple path in the
        remaining edge set, using a DFS-based heuristic.
        """
        if not edges:
            return []

        # Build adjacency from remaining edges
        def build_adj(edge_set):
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def find_long_path(adj, edge_set, start=None, max_depth=50):
            """
            Find a reasonably long simple path using DFS with backtracking.
            Returns (path_vertices, path_edges_set).
            """
            if not edge_set:
                return None

            # Try to find a path starting from a vertex with high degree
            deg = {v: len(neigh) for v, neigh in adj.items()}
            # Sort vertices by degree descending
            sorted_verts = sorted(deg.keys(), key=lambda x: -deg[x])

            best_path = None
            best_len = 0

            for start_v in sorted_verts[:min(20, len(sorted_verts))]:
                if start_v not in adj or not adj[start_v]:
                    continue

                # DFS to find a long simple path
                visited = {start_v}
                used_edges = set()
                path = [start_v]

                def dfs(current, depth):
                    nonlocal best_path, best_len
                    if depth >= max_depth:
                        if len(path) - 1 > best_len:
                            best_len = len(path) - 1
                            best_path = list(path)
                        return

                    # Get neighbors not yet visited, prefer those with higher degree
                    neighbors = list(adj[current])
                    # Sort by degree descending for heuristic
                    neighbors.sort(key=lambda x: -len(adj.get(x, [])))

                    for nb in neighbors:
                        if nb in visited:
                            continue
                        edge = tuple(sorted((current, nb)))
                        if edge not in edge_set or edge in used_edges:
                            continue

                        # Check if this edge is still available
                        if edge not in edge_set:
                            continue

                        visited.add(nb)
                        used_edges.add(edge)
                        path.append(nb)

                        dfs(nb, depth + 1)

                        path.pop()
                        visited.remove(nb)
                        used_edges.remove(edge)

                        # Early termination if we've found a good path
                        if len(path) - 1 > best_len:
                            best_len = len(path) - 1
                            best_path = list(path)

                dfs(start_v, 0)

                if best_path and len(best_path) - 1 >= best_len:
                    break

            if best_path is None:
                # Fallback: just pick any single edge
                e = next(iter(edge_set))
                return [list(e), {e}]

            # Reconstruct edge set from path
            path_edges = set()
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i + 1])))
                path_edges.add(e)

            return [best_path, path_edges]

        # Alternative: use a simpler greedy that builds paths by extending
        def greedy_extend(edges_remaining):
            """
            Greedy path decomposition:
            1. Pick a starting edge.
            2. Extend the path at both ends as far as possible.
            3. Remove those edges, repeat.
            """
            paths = []
            remaining = set(edges_remaining)

            while remaining:
                if time.time() > deadline:
                    break

                # Start with a random edge (or the first one)
                start_edge = next(iter(remaining))
                remaining.discard(start_edge)

                # Build path starting from this edge
                # path is a list of vertices
                # We'll build it by extending from both ends

                # Determine orientation
                u, v = start_edge
                path = [u, v]
                used_edges = {start_edge}

                # Extend from the left (u side)
                while True:
                    if time.time() > deadline:
                        break
                    current = path[0]
                    extended = False
                    # Get neighbors of current via remaining edges
                    neighbors = []
                    for e in remaining:
                        if current in e:
                            other = e[0] if e[1] == current else e[1]
                            if other not in path:
                                neighbors.append((other, e))
                    if not neighbors:
                        break
                    # Pick the neighbor that leads to the longest extension
                    # Simple heuristic: pick one with fewest options (leaf-like)
                    neighbors.sort(key=lambda x: len([e for e in remaining if x[0] in e]))
                    nb, edge = neighbors[0]
                    path.insert(0, nb)
                    remaining.discard(edge)
                    used_edges.add(edge)
                    extended = True
                    if not extended:
                        break

                # Extend from the right (v side)
                while True:
                    if time.time() > deadline:
                        break
                    current = path[-1]
                    extended = False
                    neighbors = []
                    for e in remaining:
                        if current in e:
                            other = e[0] if e[1] == current else e[1]
                            if other not in path:
                                neighbors.append((other, e))
                    if not neighbors:
                        break
                    neighbors.sort(key=lambda x: len([e for e in remaining if x[0] in e]))
                    nb, edge = neighbors[0]
                    path.append(nb)
                    remaining.discard(edge)
                    used_edges.add(edge)
                    extended = True
                    if not extended:
                        break

                paths.append(path)

            return paths

        # Try the greedy extend approach
        paths = greedy_extend(set(edges))

        # Verify all edges are covered
        covered = set()
        for path in paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i + 1])))
                covered.add(e)

        if covered != set(edges):
            # Add missing edges as single-edge paths
            for e in set(edges) - covered:
                paths.append([e[0], e[1]])

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Emergency: return single-edge paths for remaining
            edges = instance(*spec)
            results.append([[e[0], e[1]] for e in edges])
            continue

        edges = instance(*spec)
        paths = greedy_path_decomposition(edges)
        results.append(paths)

    return results
