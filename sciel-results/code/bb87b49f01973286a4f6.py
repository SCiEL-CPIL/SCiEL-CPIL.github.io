"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    # Deadline guard: kill at 120s wall-clock (verifier limit), safety margin ~15s
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY = 15.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose edges into edge-disjoint cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        # Build adjacency structure
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Remaining edges as a set for O(1) lookup
        remaining = set(tuple(sorted(e)) for e in edges)

        def find_cycle(start, used_edges):
            """Find a simple cycle starting from 'start' using DFS in the subgraph of remaining edges.
            Returns list of vertices in the cycle (without repeating start at end), or None."""
            # Build local adjacency from remaining edges
            local_adj = defaultdict(set)
            for (u, v) in remaining:
                if (u, v) not in used_edges:
                    local_adj[u].add(v)
                    local_adj[v].add(u)

            # DFS to find a cycle
            parent = {}
            visited = {start: None}
            stack = [start]

            while stack:
                node = stack.pop()
                for neighbor in sorted(local_adj.get(node, [])):
                    if neighbor == start and len(visited) >= 3:
                        # Found a cycle back to start
                        # Reconstruct path
                        cycle = [node]
                        cur = node
                        while cur != start:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle.reverse()
                        return cycle
                    if neighbor not in visited:
                        visited[neighbor] = node
                        parent[neighbor] = node
                        stack.append(neighbor)
            return None

        def find_longest_cycle(start, used_edges):
            """Try to find a longer cycle by exploring more paths."""
            local_adj = defaultdict(set)
            for (u, v) in remaining:
                if (u, v) not in used_edges:
                    local_adj[u].add(v)
                    local_adj[v].add(u)

            best_cycle = None
            best_len = 0

            # Iterative DFS with path tracking
            stack = [(start, [start], {start})]
            while stack:
                node, path, seen = stack.pop()
                if len(path) > best_len:
                    # Check if we can close the cycle
                    if start in local_adj.get(node, set()) and len(path) >= 3:
                        if (min(start, node), max(start, node)) not in used_edges:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = list(path)
                for neighbor in sorted(local_adj.get(node, [])):
                    if neighbor == start:
                        if len(path) >= 3:
                            if (min(start, node), max(start, node)) not in used_edges:
                                if len(path) > best_len:
                                    best_len = len(path)
                                    best_cycle = list(path)
                        continue
                    if neighbor not in seen:
                        new_path = path + [neighbor]
                        new_seen = seen | {neighbor}
                        # Prune: don't explore if path is already longer than best possible
                        stack.append((neighbor, new_path, new_seen))

            return best_cycle

        pieces = []
        used_edges = set()

        # Greedy: repeatedly find a cycle, add it, remove its edges
        while remaining:
            if time.time() - start_time > DEADLINE - SAFETY:
                break

            # Try to find a cycle from various starting points
            best_cycle = None
            best_len = 0

            # Get candidate starting vertices (those with degree >= 2 in remaining graph)
            candidates = []
            for (u, v) in remaining:
                if (u, v) not in used_edges:
                    candidates.append(u)
                    candidates.append(v)

            # Try a subset of candidates for speed
            random.shuffle(candidates)
            candidates = candidates[:min(len(candidates), 30)]

            for start in candidates:
                if time.time() - start_time > DEADLINE - SAFETY:
                    break

                # First try simple DFS cycle
                cycle = find_cycle(start, used_edges)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle

                # If we have a decent cycle, also try to extend
                if best_len >= 5:
                    # Try longest cycle search from this start
                    lc = find_longest_cycle(start, used_edges)
                    if lc and len(lc) > best_len:
                        best_len = len(lc)
                        best_cycle = lc

            if best_cycle:
                # Add the cycle as a piece
                pieces.append(best_cycle)
                # Mark its edges as used
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = (min(u, v), max(u, v))
                    used_edges.add(edge)
                    if edge in remaining:
                        remaining.discard(edge)
            else:
                # No cycle found; take one edge as a single-edge piece
                # Prefer edges whose removal might help future cycles
                # Just take an arbitrary remaining edge
                edge = next(iter(remaining))
                u, v = edge
                pieces.append([u, v])
                used_edges.add(edge)
                remaining.discard(edge)

        # Verify all edges are covered
        covered = set()
        for piece in pieces:
            if len(piece) == 2:
                covered.add((min(piece[0], piece[1]), max(piece[0], piece[1])))
            else:
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    covered.add((min(u, v), max(u, v)))

        # Check coverage
        all_edges = set(tuple(sorted(e)) for e in edges)
        if covered != all_edges:
            # Fallback: return all edges as single pieces
            return [[u, v] for u, v in edges]

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - start_time > DEADLINE - SAFETY:
            # Not enough time; use trivial decomposition for remaining
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
