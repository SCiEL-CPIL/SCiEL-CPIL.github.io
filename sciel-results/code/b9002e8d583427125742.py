import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        # Build full edge set
        all_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_edges.add((u, w))
        for v in range(k, n):
            for u in range(k):
                if u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))

        uncovered = set(all_edges)
        cliques = []

        # Precompute: for each vertex, its neighbor set
        # We'll use a global greedy: repeatedly find the clique that covers the most uncovered edges

        # For efficiency, we track which vertices are still "active" (have uncovered edges)
        # A vertex is active if it has at least one uncovered edge incident to it

        while uncovered:
            if time.time() > deadline:
                break

            # Find the best clique to add
            # Strategy: for each candidate "seed" vertex, try to build the maximal clique
            # that covers the most uncovered edges

            best_clique = None
            best_count = 0

            # Candidate seeds: vertices that appear in uncovered edges
            active_vertices = set()
            for e in uncovered:
                active_vertices.add(e[0])
                active_vertices.add(e[1])

            # For each active vertex, try to grow a clique from it
            for seed in active_vertices:
                # Start with seed, try to add other vertices
                # A valid clique containing seed: all vertices must be mutually adjacent
                # and all edges within must be uncovered (or at least we count uncovered ones)

                # Greedy growth: start with {seed}, add vertices that are adjacent to all current members
                current = {seed}
                # Candidates: neighbors of seed that are in active_vertices
                candidates = [v for v in neighbors[seed] if v in active_vertices and v != seed]

                # Sort candidates by how many uncovered edges they'd add
                # For a candidate v, it adds edges (v, u) for u in current
                # We want to maximize total uncovered edges covered

                # Simple greedy: repeatedly add the candidate that covers the most new uncovered edges
                changed = True
                while changed:
                    changed = False
                    best_candidate = None
                    best_gain = 0
                    for v in candidates:
                        if v in current:
                            continue
                        # Check if v is adjacent to all in current
                        valid = True
                        for u in current:
                            if u not in neighbors[v]:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Count uncovered edges that would be covered by adding v
                        gain = 0
                        for u in current:
                            e = (min(v, u), max(v, u))
                            if e in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = v
                    if best_candidate is not None and best_gain > 0:
                        current.add(best_candidate)
                        candidates.remove(best_candidate)
                        changed = True

                # Count total uncovered edges covered by current clique
                total_covered = 0
                for u1, u2 in combinations(sorted(current), 2):
                    if (u1, u2) in uncovered:
                        total_covered += 1

                if total_covered > best_count:
                    best_count = total_covered
                    best_clique = current

            if best_clique is None or len(best_clique) < 2:
                # Shouldn't happen if uncovered is non-empty, but safety
                break

            cliques.append(sorted(best_clique))
            for u1, u2 in combinations(sorted(best_clique), 2):
                uncovered.discard((u1, u2))

        # Safety net: cover any remaining uncovered edges with 2-cliques
        while uncovered:
            if time.time() > deadline:
                break
            e = next(iter(uncovered))
            cliques.append(sorted(e))
            uncovered.discard(e)

        results.append([[int(x) for x in cl] for cl in cliques])

    return results
