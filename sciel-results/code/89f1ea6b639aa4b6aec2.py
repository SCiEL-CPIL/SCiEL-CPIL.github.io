"""Clique partition of two fixed dense random graphs via randomized greedy max-clique extraction."""
# EVOLVE-BLOCK-START
import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 135.0  # wall-clock guard (evaluator kills at 150s)


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]


def _greedy_partition(edges, n, rng=None):
    """Greedy clique partition: repeatedly extract a large clique from remaining edges."""
    if not edges:
        return []

    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    remaining = set(edge_set)
    cliques = []

    if rng is None:
        rng = _random.Random(0)

    while remaining:
        if time.time() > _DEADLINE - 3:
            for e in remaining:
                cliques.append([e[0], e[1]])
            break

        # Compute degrees in remaining graph
        deg = [0] * n
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1

        # Find vertex with max degree (with small random tie-breaking for diversity)
        max_deg = max(deg)
        candidates = [v for v in range(n) if deg[v] == max_deg]
        v = rng.choice(candidates) if len(candidates) > 1 else candidates[0]

        if deg[v] == 0:
            u, w = next(iter(remaining))
            cliques.append([u, w])
            remaining.discard((min(u, w), max(u, w)))
            continue

        # Get neighbors of v in remaining graph
        S = set()
        for u, w in remaining:
            if u == v:
                S.add(w)
            elif w == v:
                S.add(u)

        if not S:
            u, w = next(iter(remaining))
            cliques.append([u, w])
            remaining.discard((min(u, w), max(u, w)))
            continue

        # Build clique starting with v
        C = [v]
        current_S = set(S)

        # Greedily extend clique
        while current_S:
            # Find vertex in current_S with max degree within current_S
            best_u = -1
            best_deg = -1
            best_candidates = []
            for u in current_S:
                d = 0
                for w in current_S:
                    if u == w:
                        continue
                    if (min(u, w), max(u, w)) in remaining:
                        d += 1
                if d > best_deg:
                    best_deg = d
                    best_u = u
                    best_candidates = [u]
                elif d == best_deg:
                    best_candidates.append(u)

            if best_u == -1:
                break

            # Random tie-breaking among equally-good candidates
            if len(best_candidates) > 1:
                best_u = rng.choice(best_candidates)

            C.append(best_u)

            # Update current_S: keep only vertices connected to best_u
            new_S = set()
            for w in current_S:
                if w == best_u:
                    continue
                if (min(best_u, w), max(best_u, w)) in remaining:
                    new_S.add(w)
            current_S = new_S

        if len(C) >= 2:
            cliques.append(C)
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, w = C[i], C[j]
                    e = (min(u, w), max(u, w))
                    remaining.discard(e)
        else:
            u, w = next(iter(remaining))
            cliques.append([u, w])
            remaining.discard((min(u, w), max(u, w)))

    return cliques


def _solve_instance(edges, n):
    """Try multiple random restarts, keep best partition."""
    best_cliques = None
    best_count = float('inf')

    # Deterministic first pass
    cliques = _greedy_partition(edges, n, _random.Random(0))
    best_cliques = cliques
    best_count = len(cliques)

    # Randomized restarts with different seeds
    for seed in range(1, 50):
        if time.time() > _DEADLINE - 10:
            break
        cliques = _greedy_partition(edges, n, _random.Random(seed))
        if len(cliques) < best_count:
            best_count = len(cliques)
            best_cliques = cliques

    return best_cliques


def construct():
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)

    res1 = _solve_instance(edges1, 32)
    if time.time() > _DEADLINE - 5:
        # Not enough time for second instance with restarts; do deterministic
        res2 = _greedy_partition(edges2, 64, _random.Random(0))
    else:
        res2 = _solve_instance(edges2, 64)

    return [res1, res2]
# EVOLVE-BLOCK-END
