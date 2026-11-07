"""Rewritten program: vertex-centric clique packing for split graphs."""
# EVOLVE-BLOCK-START
import time

def construct():
    _t0 = time.time()
    _DEADLINE = _t0 + 145.0  # safety margin under 150s wall-clock

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        rng = Random(seed)
        # Build core clique and cross edges deterministically (matches instance())
        core_edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            cross[v] = nbrs

        # Forced cliques: each (u, v) with v independent and u a core neighbor
        forced = []
        covered_core = set()
        for v in range(k, n):
            for u in cross[v]:
                forced.append([u, v])
                covered_core.add((min(u, v), max(u, v)))

        # Remaining core edges must be covered by core-only cliques.
        # Covering all of K_k in one clique is valid (it's a clique) and minimal.
        cliques = forced + [[list(range(k))]]

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        if time.time() > _DEADLINE:
            # Fallback: return what we have plus trivial edge-clique partitions
            while len(result) < 3:
                n, k, p, seed = specs[len(result)]
                result.append([[u, v] for u, v in instance(n, k, p, seed)])
            break
        result.append(solve(*spec))

    # Type enforcement: ints as Python int, cliques as list
    out = []
    for cliques in result:
        cleaned = []
        for c in cliques:
            cleaned.append([int(x) for x in c])
        out.append(cleaned)
    return out
# EVOLVE-BLOCK-END
