"""Clique partitioning of dense random graphs - greedy maximal clique extraction."""

import time
from random import Random

_DEADLINE = 140.0  # safety margin before 150s kill
_START = time.time()

def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def construct():
    global _START
    _START = time.time()

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def greedy_clique_partition(edges, n):
        # Build adjacency set for O(1) membership
        adj = [set() for _ in range(n)]
        remaining = set(edges)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        cliques = []

        while remaining:
            if _remaining_time() < 2.0:
                # Emergency fallback: one clique per edge
                for e in remaining:
                    cliques.append([e[0], e[1]])
                return cliques

            # Pick vertex with most remaining incident edges
            # Count remaining edges per vertex
            deg = [0] * n
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1

            # Select vertex with max degree
            start_v = max(range(n), key=lambda x: deg[x])
            if deg[start_v] == 0:
                break

            # Greedily grow a clique starting from start_v
            # We want a clique C such that all edges within C are in `remaining`
            # and we maximize |C| (or at least get a good clique)
            # Strategy: iteratively add vertices that connect to all current clique members
            # and have remaining edges to all current members

            clique = [start_v]
            # Candidate neighbors: must have remaining edge to start_v
            candidates = [v for v in adj[start_v] if (start_v, v) in remaining or (v, start_v) in remaining]

            # Sort candidates by degree (prefer high-degree to grow larger clique)
            candidates.sort(key=lambda v: deg[v], reverse=True)

            for cand in candidates:
                if _remaining_time() < 1.0:
                    break
                # Check if cand has remaining edge to every vertex in clique
                can_add = True
                for cv in clique:
                    if cv == cand:
                        continue
                    pair = (cv, cand) if cv < cand else (cand, cv)
                    if pair not in remaining:
                        can_add = False
                        break
                if can_add:
                    clique.append(cand)

            # Ensure clique has at least 2 vertices
            if len(clique) < 2:
                # Just take one edge
                e = next(iter(remaining))
                cliques.append([e[0], e[1]])
                remaining.discard(e)
                continue

            # Remove all edges within the clique from remaining
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    pair = (a, b) if a < b else (b, a)
                    remaining.discard(pair)

            cliques.append(clique)

        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = greedy_clique_partition(edges, n)
        results.append(cliques)

    return results
