"""Rewritten program: greedy maximal-clique partition on the remaining graph."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        edges += cross_edges
        return edges

    def solve_split(n, k, p, seed):
        # Build the graph: C is a clique, I is independent, cross edges with prob p
        rng = Random(seed)
        adj = {v: set() for v in range(n)}
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)

        # Remaining edges as a set of frozensets of size 2
        remaining = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    remaining.add((u, v))

        cliques = []

        while remaining:
            if time.time() > deadline:
                break

            # Build adjacency among vertices that still have edges
            active = set()
            for (u, v) in remaining:
                active.add(u)
                active.add(v)

            # Greedy maximal clique via degree ordering
            # Order vertices by (degree in remaining, vertex id) descending
            deg = {v: 0 for v in active}
            for (u, v) in remaining:
                deg[u] += 1
                deg[v] += 1

            ordered = sorted(active, key=lambda x: (-deg[x], x))

            # Greedy: try to add each vertex if connected to all current clique members
            clique = []
            for v in ordered:
                if all(u in adj[v] for u in clique):
                    clique.append(v)

            if len(clique) < 2:
                # Shouldn't happen if remaining is non-empty, but safety
                # Take any remaining edge
                e = next(iter(remaining))
                clique = [e[0], e[1]]

            # Verify clique is valid (all pairs are edges in original graph)
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    if (clique[i], clique[j]) not in adj.get(clique[i], set()) and (clique[j], clique[i]) not in adj.get(clique[j], set()):
                        # Not a clique; fall back to just the first two
                        clique = [clique[0], clique[1]]
                        break

            if len(clique) < 2:
                clique = [clique[0], clique[1]]

            cliques.append(clique)

            # Remove all edges within this clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if a < b:
                        remaining.discard((a, b))
                    else:
                        remaining.discard((b, a))

        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        result.append(solve_split(*spec))

    return result
# EVOLVE-BLOCK-END
