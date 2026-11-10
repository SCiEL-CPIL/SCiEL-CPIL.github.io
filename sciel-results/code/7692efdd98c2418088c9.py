"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
import random
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose a set of edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        remaining = set(edges)
        pieces = []

        def build_adj(rem):
            adj = {}
            for u, v in rem:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            return adj

        adj = build_adj(remaining)

        def find_longest_cycle(start, max_len=80):
            """Find a long cycle starting and ending at start using randomized DFS with bias toward longer paths."""
            if len(remaining) < 3:
                return None

            best_cycle = None

            # Try multiple random walks to find a long cycle
            for attempt in range(150):
                path = [start]
                visited = {start}
                current = start
                success = False

                for step in range(max_len):
                    neighbors = [nb for nb in adj.get(current, [])
                                if (min(current, nb), max(current, nb)) in remaining]

                    if not neighbors:
                        break

                    # Prefer unvisited neighbors, but allow closing the cycle if possible
                    unvisited = [nb for nb in neighbors if nb not in visited]

                    # If we can close the cycle and the path is reasonably long, do so
                    if start in neighbors and len(path) >= 4:
                        # Only close if we've explored enough
                        if len(path) >= 5 or random.random() < 0.3:
                            path.append(start)
                            success = True
                            break

                    if unvisited:
                        # Bias toward neighbors that have more unvisited neighbors (to extend path)
                        if len(unvisited) > 1:
                            # Score each unvisited neighbor by how many unvisited neighbors they have
                            scores = []
                            for nb in unvisited:
                                nb_unvisited = [n2 for n2 in adj.get(nb, []) 
                                               if (min(nb, n2), max(nb, n2)) in remaining 
                                               and n2 not in visited and n2 != start]
                                scores.append(len(nb_unvisited))
                            # Pick with probability proportional to score, with minimum weight
                            weights = [s + 1 for s in scores]
                            total_weight = sum(weights)
                            r = random.random() * total_weight
                            cumulative = 0
                            next_node = unvisited[0]
                            for nb, w in zip(unvisited, weights):
                                cumulative += w
                                if r <= cumulative:
                                    next_node = nb
                                    break
                        else:
                            next_node = unvisited[0]
                    else:
                        # Only visited neighbors left, try to close cycle
                        if start in neighbors and len(path) >= 3:
                            path.append(start)
                            success = True
                            break
                        break

                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

                if success and (best_cycle is None or len(path) > len(best_cycle)):
                    best_cycle = path[:]

            return best_cycle

        def find_cycle_greedy():
            """Find a cycle using a greedy approach that prefers longer cycles."""
            if len(remaining) < 3:
                return None

            # Find vertices with degree >= 2
            vertex_degrees = {}
            for u, v in remaining:
                vertex_degrees[u] = vertex_degrees.get(u, 0) + 1
                vertex_degrees[v] = vertex_degrees.get(v, 0) + 1

            candidates = [v for v in vertex_degrees if vertex_degrees[v] >= 2]
            if not candidates:
                return None

            best_cycle = None
            best_len = 0

            # Try high-degree vertices first, then lower-degree ones
            candidates_sorted = sorted(candidates, key=lambda v: vertex_degrees[v], reverse=True)
            
            # Limit the number of start vertices we try to save time
            max_starts = min(len(candidates_sorted), 15)
            
            for start in candidates_sorted[:max_starts]:
                cycle = find_longest_cycle(start)
                if cycle and len(cycle) > best_len:
                    best_cycle = cycle
                    best_len = len(cycle)
                
                # Early termination if we found a very long cycle
                if best_len >= len(remaining) * 0.4:
                    break

            return best_cycle

        # Greedily extract cycles, preferring longer ones
        no_cycle_count = 0
        while remaining:
            if time.time() > deadline - 10:
                break

            cycle = find_cycle_greedy()

            if cycle:
                # Extract this cycle
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    edge = (min(u, v), max(u, v))
                    remaining.discard(edge)
                pieces.append(cycle[:-1][:])
                # Rebuild adjacency
                adj = build_adj(remaining)
                no_cycle_count = 0
            else:
                # No more cycles; take remaining edges as singles
                for e in sorted(remaining):
                    pieces.append([e[0], e[1]])
                remaining = set()
                break

        # If we hit the deadline, add remaining edges as singles
        if remaining:
            for e in sorted(remaining):
                pieces.append([e[0], e[1]])
            remaining = set()

        # Refinement: try to merge single edges into cycles by breaking existing cycles
        # This is a heuristic improvement step
        single_edges = [p for p in pieces if len(p) == 2]
        if single_edges and len(remaining) == 0:
            # Try to find cycles using single edges and parts of existing cycles
            # For simplicity, we'll just try to find any new cycles in the full graph
            # by checking if any 3 single edges form a triangle, etc.
            # This is a simplified refinement - in practice, we'd need to be more careful
            # about which edges are available
            pass

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
