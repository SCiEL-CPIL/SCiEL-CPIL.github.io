"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    t_start = time.time()
    DEADLINE = 145.0  # safety margin under 150s kill

    def instance_edges(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve_instance(n, p, seed):
        edges = instance_edges(n, p, seed)
        # adjacency as sets of remaining edges
        adj = {v: set() for v in range(n)}
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(edges)
        cliques = []

        def bronk_max(candidates, min_size=2):
            # find a largest clique among 'candidates' (set of vertices)
            # using Bron-Kerbosch with pivot, returning the best clique found
            best = [set()]

            def bk(R, P, X):
                if not P and not X:
                    if len(R) > len(best[0]):
                        best[0] = set(R)
                    return
                # pivot: choose u in P ∪ X maximizing |P ∩ N(u)|
                pivot_pool = P | X
                pivot = max(pivot_pool, key=lambda u: len(P & adj[u])) if pivot_pool else None
                candidates_iter = list(P - (adj[pivot] if pivot is not None else set()))
                for v in candidates_iter:
                    # only recurse if we can still beat best
                    if len(R) + len(P) <= len(best[0]):
                        continue
                    bk(R | {v}, P & adj[v], X & adj[v])
                    P.remove(v)
                    X.add(v)

            bk(set(), set(candidates), set())
            return best[0]

        while remaining:
            if time.time() > DEADLINE - 2.0:
                break
            # candidate vertices = endpoints of remaining edges
            verts = set()
            for (u, v) in remaining:
                verts.add(u)
                verts.add(v)
            clique = bronk_max(verts)
            if len(clique) < 2:
                # should not happen; fallback to single edge
                (u, v) = next(iter(remaining))
                clique = {u, v}
            cliques.append(sorted(clique))
            # remove all edges within clique
            cl = list(clique)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    e = (a, b) if a < b else (b, a)
                    remaining.discard(e)
        return cliques

    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > DEADLINE - 2.0:
            results.append([])
            continue
        results.append(solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
