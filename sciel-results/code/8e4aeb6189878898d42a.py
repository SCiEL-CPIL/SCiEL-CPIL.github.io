"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose edges into cycles and single edges to minimize piece count."""
        if not edges:
            return []

        # Build adjacency from remaining edges
        def build_adj(rem_edges):
            adj = {}
            for u, v in rem_edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            return adj

        remaining = set((u, v) if u < v else (v, u) for u, v in edges)
        cycles = []

        def find_shortest_cycle(start, adj):
            """Find shortest cycle through start using BFS."""
            # BFS from each neighbor of start, looking for path back to start
            best = None
            for nb in sorted(adj.get(start, [])):
                visited = {start: None, nb: start}
                queue = [(nb, [start, nb])]
                while queue:
                    node, path = queue.pop(0)
                    for nxt in sorted(adj.get(node, [])):
                        if nxt == start and len(path) >= 2:
                            if best is None or len(path) < len(best):
                                best = path
                            break
                        if nxt not in visited:
                            visited[nxt] = node
                            queue.append((nxt, path + [nxt]))
            return best

        # Greedy cycle extraction: repeatedly find and remove longest cycles
        max_iter = 500
        for _ in range(max_iter):
            if time.time() - t0 > DEADLINE - 5:
                break
            if not remaining:
                break

            adj = build_adj(remaining)

            # Find best (longest) cycle
            best_cycle = None
            for u in sorted(adj.keys()):
                if not adj.get(u):
                    continue
                # Check u has edges in remaining
                if not any((min(u, v), max(u, v)) in remaining for v in adj[u]):
                    continue
                cycle = find_shortest_cycle(u, adj)
                if cycle and len(cycle) >= 3:
                    # Verify all edges exist in remaining
                    valid = True
                    for i in range(len(cycle) - 1):
                        a, b = cycle[i], cycle[i + 1]
                        e = (min(a, b), max(a, b))
                        if e not in remaining:
                            valid = False
                            break
                    if valid:
                        if best_cycle is None or len(cycle) > len(best_cycle):
                            best_cycle = cycle

            if best_cycle and len(best_cycle) >= 3:
                cycles.append(best_cycle)
                for i in range(len(best_cycle) - 1):
                    a, b = best_cycle[i], best_cycle[i + 1]
                    remaining.discard((min(a, b), max(a, b)))
            else:
                break

        # Remaining edges become single-edge pieces
        pieces = [list(c) for c in cycles]
        for e in sorted(remaining):
            pieces.append(list(e))
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            results.append(decompose(edges))

    return results
# EVOLVE-BLOCK-END
