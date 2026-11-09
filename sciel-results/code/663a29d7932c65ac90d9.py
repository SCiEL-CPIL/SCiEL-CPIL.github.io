import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    from random import Random
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed, time_budget):
        edges = instance(n, k, p, seed)

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        I_set = set(I)

        # Internal C edges as bitmask per vertex
        c_adj_mask = [0] * k
        for u in range(k):
            for v in range(u + 1, k):
                if v in adj[u]:
                    c_adj_mask[u] |= (1 << v)
                    c_adj_mask[v] |= (1 << u)

        # I vertex shadows (bitmask over C)
        i_shadow = {}
        for v in I:
            mask = 0
            for u in range(k):
                if u in adj[v]:
                    mask |= (1 << u)
            i_shadow[v] = mask

        # Cross edge set: (v, u) for v in I, u in C, u in shadow(v)
        # Represent as list of (v, u)
        cross_edges = []
        for v in I:
            m = i_shadow[v]
            u = 0
            while m:
                if m & 1:
                    cross_edges.append((v, u))
                m >>= 1
                u += 1

        # C-C edges list
        cc_edges = []
        for u in range(k):
            m = c_adj_mask[u]
            v = u + 1
            mm = m >> (u + 1)
            while mm:
                if mm & 1:
                    cc_edges.append((u, v))
                mm >>= 1
                v += 1

        def is_clique_in_C(indices):
            # indices: list of C vertex ids
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]
                    if not (c_adj_mask[a] >> b) & 1:
                        return False
            return True

        def max_clique_greedy(available_mask, rng):
            # Greedy max clique on induced subgraph of C given by available_mask
            avail = []
            for u in range(k):
                if (available_mask >> u) & 1:
                    avail.append(u)
            if not avail:
                return []
            # order by degree within available
            def deg(u):
                cnt = 0
                for w in avail:
                    if u != w and (c_adj_mask[u] >> w) & 1:
                        cnt += 1
                return cnt
            avail.sort(key=lambda u: (-deg(u), rng.random()))
            clique = []
            for u in avail:
                ok = True
                for w in clique:
                    if not (c_adj_mask[u] >> w) & 1:
                        ok = False
                        break
                if ok:
                    clique.append(u)
            return clique

        def cover_cc_edges(cc_edge_list, rng, time_budget):
            # Peel max cliques from remaining C-C edges
            remaining = set(cc_edge_list)
            result = []
            while remaining:
                # build induced graph on remaining edges
                verts = set()
                for a, b in remaining:
                    verts.add(a); verts.add(b)
                # greedy: pick a vertex with high remaining degree, grow clique
                rem_adj = {u: set() for u in verts}
                for a, b in remaining:
                    rem_adj[a].add(b); rem_adj[b].add(a)
                # order vertices
                ordered = sorted(verts, key=lambda u: (-len(rem_adj[u]), rng.random()))
                best_clique = []
                for start in ordered:
                    clique = [start]
                    cands = sorted(rem_adj[start], key=lambda x: (-len(rem_adj[x]), rng.random()))
                    for w in cands:
                        ok = True
                        for u in clique:
                            if w not in rem_adj[u]:
                                ok = False
                                break
                        if ok:
                            clique.append(w)
                    if len(clique) > len(best_clique):
                        best_clique = clique
                if len(best_clique) < 2:
                    # fallback: single edge
                    a, b = next(iter(remaining))
                    best_clique = [a, b]
                result.append(sorted(best_clique))
                for i in range(len(best_clique)):
                    for j in range(i + 1, len(best_clique)):
                        a, b = best_clique[i], best_clique[j]
                        e = (min(a, b), max(a, b))
                        remaining.discard(e)
                if time_budget and time.time() > time_budget:
                    break
            # leftover edges as 2-cliques
            for a, b in remaining:
                result.append(sorted([a, b]))
            return result

        def cover_cross_with_bicliques(cross_edge_list, i_shadow, rng, time_budget):
            # Each biclique: (i_set, c_set) where c_set is a clique in C,
            # and for all v in i_set, c_set ⊆ shadow(v).
            # Covers edges (v, u) for v in i_set, u in c_set.
            remaining = set(cross_edge_list)
            bicliques = []  # list of (frozen_i_set, c_set_list)
            while remaining:
                if time_budget and time.time() > time_budget:
                    break
                # Try to find a large biclique
                # Pick a C vertex with high remaining degree
                c_deg = defaultdict(int)
                for (v, u) in remaining:
                    c_deg[u] += 1
                if not c_deg:
                    break
                # order C vertices
                c_order = sorted(c_deg.keys(), key=lambda u: (-c_deg[u], rng.random()))
                best_bi = None
                best_bi_size = -1
                for start_c in c_order[:min(len(c_order), 12)]:
                    # candidate C clique containing start_c, all with high remaining degree
                    cand = [start_c]
                    # grow C clique among remaining C neighbors
                    for u in c_order:
                        if u == start_c:
                            continue
                        if len(cand) >= 6:
                            break
                        # u must be adjacent to all in cand in C
                        ok = True
                        for w in cand:
                            if not (c_adj_mask[u] >> w) & 1:
                                ok = False
                                break
                        if ok:
                            cand.append(u)
                    # Now find I vertices connected to ALL of cand in remaining edges
                    i_cands = set()
                    for (v, u) in remaining:
                        if u in cand:
                            i_cands.add(v)
                    # filter: v must have all of cand in shadow
                    i_final = []
                    for v in i_cands:
                        ok = True
                        for u in cand:
                            if not (i_shadow[v] >> u) & 1:
                                ok = False
                                break
                            if (v, u) not in remaining:
                                ok = False
                                break
                        if ok:
                            i_final.append(v)
                    if not i_final or not cand:
                        continue
                    size = len(i_final) * len(cand)
                    if size > best_bi_size:
                        best_bi_size = size
                        best_bi = (i_final, cand)
                if best_bi is None:
                    # fallback: single edge
                    v, u = next(iter(remaining))
                    best_bi = ([v], [u])
                i_set, c_set = best_bi
                bicliques.append((frozenset(i_set), list(c_set)))
                # remove covered edges
                for v in i_set:
                    for u in c_set:
                        remaining.discard((v, u))
            # leftover as 2-cliques
            for (v, u) in remaining:
                bicliques.append((frozenset([v]), [u]))
            return bicliques

        def bicliques_to_cliques(bicliques):
            out = []
            for i_set, c_set in bicliques:
                if not i_set and not c_set:
                    continue
                verts = list(i_set) + list(c_set)
                if len(verts) >= 2:
                    out.append(sorted(verts))
            return out

        def local_improve_cliques(cliques, adj, rng, time_budget):
            # Try to merge two cliques if their union is a clique and reduces nothing? 
            # Merging two cliques that are disjoint in edges but union is a clique would
            # reduce count by 1. Try merging.
            changed = True
            while changed and (not time_budget or time.time() < time_budget):
                changed = False
                cliques = [list(c) for c in cliques]
                for i in range(len(cliques)):
                    for j in range(i + 1, len(cliques)):
                        A, B = cliques[i], cliques[j]
                        union = list(set(A) | set(B))
                        if len(union) < 2:
                            continue
                        # check union is clique
                        ok = True
                        for a in union:
                            for b in union:
                                if a >= b:
                                    continue
                                if b not in adj[a]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            # merging valid: replace A,B with union
                            cliques[i] = union
                            cliques.pop(j)
                            changed = True
                            break
                    if changed:
                        break
            # Also try splitting a clique into two cliques if it reduces count? No, splitting increases.
            # Try: for a large clique, see if we can remove a vertex and that vertex's edges are
            # covered elsewhere? Hard. Skip.
            return [sorted(c) for c in cliques if len(c) >= 2]

        def build_solution(bicliques, cc_cliques):
            all_cliques = bicliques_to_cliques(bicliques) + cc_cliques
            return all_cliques

        def validate(cliques, all_edges_set):
            # every clique is a clique
            for c in cliques:
                for a in c:
                    for b in c:
                        if a >= b:
                            continue
                        if b not in adj[a]:
                            return False
            # every edge in exactly one clique
            edge_count = defaultdict(int)
            for c in cliques:
                for a in c:
                    for b in c:
                        if a >= b:
                            continue
                        edge_count[(min(a, b), max(a, b))] += 1
            for e in all_edges_set:
                if edge_count[e] != 1:
                    return False
            for e, cnt in edge_count.items():
                if e not in all_edges_set:
                    return False
            return True

        all_edges_set = set(edges)

        best = None
        best_count = float('inf')

        n_restarts = 8
        per_restart_budget = time_budget / n_restarts if time_budget else None

        for r in range(n_restarts):
            rng = Random(seed * 1000 + r * 7 + 1)
            if time_budget and time.time() > time_budget:
                break
            # randomize ordering by shuffling cross edge processing implicitly via rng
            bicliques = cover_cross_with_bicliques(cross_edges, i_shadow, rng,
                                                   time_budget - 1.0 if time_budget else None)
            cc_cliques = cover_cc_edges(cc_edges, rng, time_budget - 1.0 if time_budget else None)
            sol = build_solution(bicliques, cc_cliques)
            # local improve
            sol = local_improve_cliques(sol, adj, rng, time_budget - 0.5 if time_budget else None)
            if validate(sol, all_edges_set):
                cnt = len(sol)
                if cnt < best_count:
                    best_count = cnt
                    best = sol
            if time_budget and time.time() > time_budget:
                break

        if best is None:
            # Fallback: trivial valid solution (every edge as 2-clique)
            best = []
            for a, b in edges:
                best.append(sorted([a, b]))
            best_count = len(best)

        cleaned = [[int(x) for x in c] for c in best]
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for idx, spec in enumerate(specs):
        if time.time() > deadline - 5.0:
            while len(results) < 3:
                results.append([])
            return results
        # allocate time per instance, more for larger
        remaining = deadline - time.time()
        frac = (3 - idx) / 3.0
        budget = remaining * frac * 0.95
        res = solve_instance(*spec, time_budget=deadline - 2.0)
        results.append(res)

    return results
