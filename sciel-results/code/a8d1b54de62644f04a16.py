"""Rewritten program using matching-based path decomposition for cycle packing."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n):
        """Decompose graph into minimum pieces using matching-based path decomposition."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []

        def get_adj(remaining):
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def find_cycles_by_matching(remaining, rng, max_iter=50):
            """Find cycles by decomposing into paths via matching, then joining endpoints."""
            cycles = []
            single_edges = []
            current_remaining = set(remaining)
            adj = get_adj(current_remaining)

            for iteration in range(max_iter):
                if time.time() > deadline - 10:
                    break
                if not current_remaining:
                    break

                # Find a cycle using randomized DFS with multiple attempts
                cycle = _find_best_cycle(adj, current_remaining, rng)
                
                if cycle and len(cycle) >= 3:
                    # Remove cycle edges
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        current_remaining.discard(e)
                    cycles.append(cycle)
                else:
                    # No cycle found; take a single edge
                    if current_remaining:
                        e = tuple(sorted(rng.choice(list(current_remaining))))
                        current_remaining.discard(e)
                        single_edges.append(list(e))
                        # Update adjacency
                        u, v = e
                        if v in adj[u]:
                            adj[u].discard(v)
                            if not adj[u]:
                                del adj[u]
                        if u in adj[v]:
                            adj[v].discard(u)
                            if not adj[v]:
                                del adj[v]

            return cycles, single_edges

        def _find_best_cycle(adj, remaining, rng, max_attempts=20):
            """Find the best (longest) cycle using multiple strategies."""
            best_cycle = None

            # Compute current degrees
            degs = {}
            for u in adj:
                degs[u] = len([nb for nb in adj[u] if (min(u, nb), max(u, nb)) in remaining])

            if not degs:
                return None

            # Strategy 1: Try from high-degree vertices with randomized DFS
            high_deg_verts = sorted([u for u in degs if degs[u] >= 2],
                                     key=lambda u: (-degs[u], rng.random()))

            for start in high_deg_verts[:min(8, len(high_deg_verts))]:
                if time.time() > deadline - 10:
                    break

                # Multiple attempts from this start
                for attempt in range(5):
                    cycle = _dfs_cycle(start, adj, remaining, rng, max_depth=100)
                    if cycle and (best_cycle is None or len(cycle) > len(best_cycle)):
                        best_cycle = cycle
                        if len(best_cycle) >= 8:  # Good enough
                            break

                if best_cycle and len(best_cycle) >= 8:
                    break

            # Strategy 2: Try to extend short cycles
            if best_cycle and len(best_cycle) < 5:
                # Try a different approach: build a path and close it
                for start in high_deg_verts[:5]:
                    if time.time() > deadline - 10:
                        break
                    cycle = _greedy_cycle(start, adj, remaining, rng)
                    if cycle and (best_cycle is None or len(cycle) > len(best_cycle)):
                        best_cycle = cycle

            return best_cycle

        def _dfs_cycle(start, adj, remaining, rng, max_depth=100):
            """Find a cycle through start using randomized DFS."""
            best = None

            for _ in range(3):  # Multiple random restarts
                path = [start]
                visited = {start}
                current = start
                found = False

                for _ in range(max_depth):
                    neighbors = [nb for nb in adj[current]
                                if (min(current, nb), max(current, nb)) in remaining]
                    if not neighbors:
                        break

                    unvisited = [nb for nb in neighbors if nb not in visited]

                    if unvisited:
                        # Prefer higher-degree unvisited neighbors
                        unvisited_sorted = sorted(unvisited,
                                                  key=lambda nb: (-len(adj[nb]), rng.random()))
                        next_node = unvisited_sorted[0]
                    elif start in neighbors and len(path) >= 3:
                        next_node = start
                        found = True
                    else:
                        break

                    if found:
                        if len(path) >= 3:
                            if best is None or len(path) > len(best):
                                best = path[:]
                        break

                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

                if found and len(path) >= 3:
                    if best is None or len(path) > len(best):
                        best = path[:]

            return best

        def _greedy_cycle(start, adj, remaining, rng):
            """Build a cycle greedily, preferring to close early if possible."""
            path = [start]
            visited = {start}
            current = start
            found = False

            for _ in range(150):
                neighbors = [nb for nb in adj[current]
                            if (min(current, nb), max(current, nb)) in remaining]
                if not neighbors:
                    break

                unvisited = [nb for nb in neighbors if nb not in visited]

                if unvisited:
                    # Prefer neighbors that can lead to closing the cycle
                    # i.e., neighbors of start or with high degree
                    def score(nb):
                        s = 0
                        if start in adj.get(nb, set()):
                            s += 10
                        s += len(adj.get(nb, set()))
                        return -s + rng.random() * 0.1
                    unvisited_sorted = sorted(unvisited, key=score)
                    next_node = unvisited_sorted[0]
                elif start in neighbors and len(path) >= 3:
                    next_node = start
                    found = True
                else:
                    break

                if found:
                    if len(path) >= 3:
                        return path[:]
                    break

                path.append(next_node)
                visited.add(next_node)
                current = next_node

            return None

        def local_search(pieces, remaining_edges, rng, max_swaps=200):
            """Try to improve the decomposition by swapping pieces."""
            if not pieces:
                return pieces

            adj = defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)

            improved = True
            swaps = 0
            while improved and swaps < max_swaps:
                improved = False
                swaps += 1
                if time.time() > deadline - 5:
                    break

                # Try to merge two single edges into a cycle if possible
                singles = [i for i, p in enumerate(pieces) if len(p) == 2]
                for i in singles:
                    if time.time() > deadline - 5:
                        break
                    e1 = tuple(sorted(pieces[i]))
                    u1, v1 = e1
                    # Find another single edge that shares a vertex
                    for j in singles:
                        if j == i:
                            continue
                        e2 = tuple(sorted(pieces[j]))
                        u2, v2 = e2

                        # Check if we can form a cycle: u1-v1-x-y-u1 or similar
                        # This requires edges (v1, x), (x, y), (y, u1) in remaining
                        # For simplicity, try to find a path between u1 and v1 of length >= 2
                        # through remaining edges

                        # Try: is there a vertex w such that (v1,w) and (w,u1) are in remaining?
                        for w in adj.get(v1, set()):
                            if w == u1:
                                continue
                            e_v1w = tuple(sorted((v1, w)))
                            e_wu1 = tuple(sorted((w, u1)))
                            if e_v1w in remaining_edges and e_wu1 in remaining_edges:
                                # Found a triangle: u1-v1-w-u1
                                new_cycle = [u1, v1, w]
                                # Remove the two single edges and add the cycle
                                pieces[i] = new_cycle
                                pieces.pop(j)
                                remaining_edges.discard(e1)
                                remaining_edges.discard(e2)
                                remaining_edges.discard(e_v1w)
                                remaining_edges.discard(e_wu1)
                                # Update adj
                                for x, y in [(u1, v1), (v1, u1), (u1, w), (w, u1), (v1, w), (w, v1)]:
                                    if y in adj.get(x, set()):
                                        adj[x].discard(y)
                                improved = True
                                break
                        if improved:
                            break

                # Try to extend a cycle by absorbing a single edge
                for i, p in enumerate(pieces):
                    if time.time() > deadline - 5:
                        break
                    if len(p) < 3:
                        continue
                    # Try to insert a single edge into the cycle
                    # For each single edge (a,b), check if we can replace an edge in the cycle
                    # This is complex; skip for now

            return pieces

        # Main decomposition
        best_pieces = None
        best_count = len(edge_set) + 1

        max_restarts = 100
        rng = random.Random(42)

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            cycles, singles = find_cycles_by_matching(edge_set, rng)
            pieces = cycles + singles
            count = len(pieces)

            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]

            if best_count == 1:
                break

        # Local search improvement
        if best_pieces:
            # Reconstruct remaining edges for local search
            used_edges = set()
            for p in best_pieces:
                if len(p) == 2:
                    used_edges.add(tuple(sorted(p)))
                else:
                    for i in range(len(p)):
                        u, v = p[i], p[(i + 1) % len(p)]
                        used_edges.add(tuple(sorted((u, v))))
            remaining = edge_set - used_edges
            best_pieces = local_search(best_pieces, remaining, rng)

        return best_pieces if best_pieces else [list(e) for e in edge_set]

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose_graph(edges, n)
        results.append(pieces)
        if time.time() > deadline - 3:
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break

    return results
# EVOLVE-BLOCK-END
