import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

        remaining_edges = set(edges)
        pieces = []

        def rebuild_adj():
            adj = defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            return adj

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

        def find_cycle_greedy(adj):
            """Find a reasonably long cycle using greedy + local search."""
            if not remaining_edges:
                return None

            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)

            best_cycle = None
            best_len = 0

            for start in candidates[:12]:
                # Try multiple starts with greedy path extension
                for attempt in range(3):
                    visited = {start}
                    path = [start]
                    current = start

                    # Greedy: always go to highest-degree unvisited neighbor
                    while True:
                        neighbors = [n for n in adj[current] if n not in visited]
                        if not neighbors:
                            break
                        # Prefer higher degree
                        neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                        nxt = neighbors[0]
                        visited.add(nxt)
                        path.append(nxt)
                        current = nxt

                    # Check if cycle can be closed
                    if len(path) >= 3 and path[-1] in adj[path[0]]:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)

            # If greedy didn't find anything, try DFS for longer cycles
            if best_cycle is None or best_len < 5:
                search_deadline = time.time() + 0.5
                for start in candidates[:6]:
                    if time.time() > search_deadline:
                        break
                    visited = {start}
                    path = [start]

                    def dfs(current, path):
                        nonlocal best_cycle, best_len
                        if time.time() > search_deadline:
                            return
                        if len(path) >= 3 and current in adj[start]:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = list(path)
                        for neighbor in sorted(adj[current], key=lambda x: len(adj[x]), reverse=True):
                            if neighbor == start:
                                continue
                            if neighbor not in visited:
                                visited.add(neighbor)
                                path.append(neighbor)
                                dfs(neighbor, path)
                                path.pop()
                                visited.remove(neighbor)
                                if time.time() > search_deadline:
                                    return
                    dfs(start, path)

            return best_cycle

        def find_triangles(adj):
            """Find all triangles in the remaining graph."""
            triangles = []
            for u in adj:
                for v in adj[u]:
                    if v <= u:
                        continue
                    for w in adj[v]:
                        if w <= v:
                            continue
                        if w in adj[u]:
                            triangles.append([u, v, w])
            return triangles

        # Main decomposition loop
        max_iterations = 500
        iteration = 0
        while remaining_edges and iteration < max_iterations:
            if time.time() > time_limit:
                break
            iteration += 1

            adj = rebuild_adj()

            # Try to find a long cycle first
            cycle = find_cycle_greedy(adj)

            if cycle:
                pieces.append(cycle)
                remove_cycle_edges(cycle)
            else:
                # Try to find triangles
                triangles = find_triangles(adj)
                if triangles:
                    # Pick the triangle that removes the most "useful" edges
                    best_tri = triangles[0]
                    for tri in triangles:
                        # Prefer triangles where all 3 vertices have high degree
                        score = sum(len(adj[v]) for v in tri)
                        if len(adj[best_tri[0]]) + len(adj[best_tri[1]]) + len(adj[best_tri[2]]) < score:
                            best_tri = tri
                    pieces.append(best_tri)
                    remove_cycle_edges(best_tri)
                else:
                    # No cycles left; add remaining edges individually
                    for e in list(remaining_edges):
                        pieces.append(list(e))
                        remaining_edges.remove(e)
                    break

        # If we still have edges (time ran out), add them individually
        for e in list(remaining_edges):
            pieces.append(list(e))
            remaining_edges.remove(e)

        return pieces

    results = []
    time_limits = [40.0, 45.0, 50.0]

    for i, edges in enumerate(graphs):
        global_remaining = deadline - time.time()
        if global_remaining < 2.0:
            # Not enough time; fallback to trivial decomposition
            for j in range(i, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break

        current_limit = min(time_limits[i], global_remaining - 2.0)
        if current_limit < 1.0:
            current_limit = 1.0

        time_limit = time.time() + current_limit
        pieces = decompose_graph(edges, time_limit)
        results.append(pieces)

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
    total = sum(len(r) for r in res)
    print(f"Total pieces: {total}")
