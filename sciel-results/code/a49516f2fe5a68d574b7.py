"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_DEADLINE = time.time() + 115.0


def construct():
    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        m = n - k
        # cross adjacency
        cross = [set() for _ in range(k)]   # u -> set of j
        rev = [set() for _ in range(m)]     # j -> set of u
        for v in range(k, n):
            j = v - k
            for u in range(k):
                if rng.random() < p:
                    cross[u].add(j)
                    rev[j].add(u)

        # Bron-Kerbosch with pivot on the cross (bipartite) graph.
        # State: U (set of clique-verts still to add), J (set of indep-verts still to add).
        # A partial clique is (U_done, J_done); we recurse by adding one vertex at a time.
        # Here we use the standard formulation on the full vertex set but exploit structure.
        # Vertices are represented as ints: 0..k-1 (clique), k..k+m-1 (indep).
        # adjacency: clique-clique complete, indep-indep none, cross per data.
        N = k + m

        def adj(a):
            if a < k:
                # adjacent to all other clique verts + its cross indep verts
                s = set(range(k))
                s.discard(a)
                s |= (cross[a] | set()).union(k + j for j in cross[a])
                return s
            else:
                j = a - k
                return set(k + u for u in rev[j])

        # Precompute adjacency sets for speed
        adjset = [None] * N
        for a in range(k):
            s = set(range(k))
            s.discard(a)
            s |= {k + j for j in cross[a]}
            adjset[a] = s
        for j in range(m):
            adjset[k + j] = {k + u for u in rev[j]}

        maxcliques = []

        def bk(R, P, X):
            if time.time() > _DEADLINE:
                return
            if not P and not X:
                if len(R) >= 2:
                    maxcliques.append(frozenset(R))
                return
            # pivot
            if P or X:
                cand = P | X
                # choose pivot with max |cand ∩ adj(p)|
                best, bestdeg = None, -1
                for p in cand:
                    d = len(cand & adjset[p])
                    if d > bestdeg:
                        bestdeg, best = d, p
                if best is not None:
                    P = P - adjset[best]
                else:
                    return
            for v in list(P):
                P.remove(v)
                R2 = R | {v}
                bk(R2, P & adjset[v], X & adjset[v])
                X.add(v)

        bk(frozenset(), set(range(N)), frozenset())

        # Partition cross edges among maximal cliques; each cross edge lies in exactly one maximal clique.
        # Build owner map.
        owner = {}
        for cli in maxcliques:
            for a in cli:
                for b in cli:
                    if a < k and b >= k:
                        key = (a, b - k)
                        if key not in owner:
                            owner[key] = cli
                    elif a >= k and b < k:
                        key = (b, a - k)
                        if key not in owner:
                            owner[key] = cli

        # Group cross edges by their maximal clique
        from collections import defaultdict
        bycli = defaultdict(set)
        for (u, j), cli in owner.items():
            bycli[cli].add((u, j))

        cliques = [list(range(k))]  # the clique on first k vertices
        for cli, edset in bycli.items():
            if not edset:
                continue
            verts = set()
            for (u, j) in edset:
                verts.add(u)
                verts.add(k + j)
            # ensure it's a clique (it is, being subset of a maximal clique) and >=2 verts
            if len(verts) >= 2:
                cliques.append(sorted(verts))
        return cliques

    out = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        out.append(solve(*spec))
    return out
# EVOLVE-BLOCK-END
