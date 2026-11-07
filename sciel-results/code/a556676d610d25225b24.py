"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    # ---- helpers for one split instance ----
    def max_stars(adj, active_c, active_i, deadline):
        # greedy: repeatedly take the star (c, S) with max |S| among active sets
        # adj[c] = set of i in I adjacent to c
        best = []
        while True:
            if time.time() > deadline:
                return best
            best_c, best_s = -1, None
            for c in active_c:
                s = adj[c] & active_i
                if len(s) > (len(best_s) if best_s is not None else -1):
                    best_c, best_s = c, s
            if best_c < 0:
                break
            best.append((best_c, best_s))
            active_i -= best_s
            active_c.discard(best_c)
        return best

    def solve_one(n, k, p, seed, deadline):
        edges = instance(n, k, p, seed)
        E = set()
        for a, b in edges:
            E.add((a, b))
        C = set(range(k))
        I = set(range(k, n))
        # adjacency of bipartite part
        adj = {c: set() for c in C}
        for (a, b) in E:
            if a < k and b >= k:
                adj[a].add(b)
            elif b < k and a >= k:
                adj[b].add(a)
        # initial greedy stars
        stars = max_stars(adj, set(C), set(I), deadline)
        # local improvement: try to replace a small subset of stars with fewer
        # stars that cover the same edge set
        for _round in range(40):
            if time.time() > deadline:
                break
            improved = False
            m = len(stars)
            for i in range(m):
                if time.time() > deadline:
                    break
                for j in range(i + 1, m):
                    if time.time() > deadline:
                        break
                    # edge set covered by stars i and j
                    ei = {(stars[i][0], x) for x in stars[i][1]}
                    ej = {(stars[j][0], x) for x in stars[j][1]}
                    union = ei | ej
                    # candidate single stars that could cover union
                    covered = set()
                    cand = []
                    for (c, x) in union:
                        cand.append(c)
                    # try each candidate center c: S = {x : (c,x) in union}
                    # it must be a star: all (c,x) in union must be edges (they are, since in union)
                    # and it must be a subset of adj[c] (yes). A single star covers its edges.
                    # We need union to be exactly the star's edges (no extra), and single star covers all.
                    # So union must be a star itself.
                    for c in set(cand):
                        S = set()
                        ok = True
                        for (cc, x) in union:
                            if cc == c:
                                S.add(x)
                            else:
                                ok = False
                                break
                        if not ok:
                            continue
                        # this single star covers exactly union if every x in S has (c,x) in union
                        # (true by construction) and no extra edges. A star (c,S) covers {(c,x):x in S}=union.
                        # Replace stars i,j with this single star if it is valid (S subset adj[c]).
                        if S <= adj[c]:
                            new_stars = [s for t, s in enumerate(stars) if t != i and t != j]
                            new_stars.append((c, S))
                            # verify it covers union (it does) and nothing else is left uncovered
                            # Since we only removed i,j and added a star covering exactly their union, ok.
                            stars = new_stars
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        # build final clique list
        cliques = [sorted(C)]
        for (c, S) in stars:
            if S:
                cliques.append(sorted([c] + list(S)))
        # verify: every edge covered exactly once, every clique a clique
        cover = {}
        for cl in cliques:
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    key = (min(cl[a], cl[b]), max(cl[a], cl[b]))
                    cover[key] = cover.get(key, 0) + 1
        for key in cover:
            if cover[key] != 1:
                raise RuntimeError("overlap/miss")
        if set(cover.keys()) != E:
            raise RuntimeError("edge set mismatch")
        return cliques

    start = time.time()
    deadline = start + 120.0
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 5.0:
            # safety: fall back to seed for remaining
            n, k, p, s = spec
            results.append([list(e) for e in instance(n, k, p, s)])
            continue
        results.append(solve_one(*spec, deadline - 5.0))
    # convert to plain python ints / lists
    out = []
    for lst in results:
        nl = []
        for cl in lst:
            nl.append([int(x) for x in cl])
        out.append(nl)
    return out
# EVOLVE-BLOCK-END
