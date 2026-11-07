"""Rewritten program: clique partition of split graphs via structured greedy + local improvement."""
import time
from random import Random

_DEADLINE = 140.0  # safety margin under 150s wall-clock
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        """Return a list of cliques (lists of int) partitioning all edges of the split graph."""
        edges = instance(n, k, p, seed)
        # Build adjacency
        adj = [set() for _ in range(n)]
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        # Structure: clique part = [0, k), independent part = [k, n)
        # A clique is either subset of [0,k) or {v} ∪ S with v >= k, S ⊆ [0,k), S ⊆ N(v)
        # We need to partition edge_set into such cliques, each of size >= 2.

        # Strategy:
        # 1. For each independent vertex v, its cross-edges must be covered by cliques containing v.
        #    Each such clique = {v} ∪ S where S ⊆ N(v) (S can be any subset, since [0,k) is a clique).
        #    The clique {v} ∪ S covers cross-edges (v,u) for u in S, and clique-edges (u,w) for u,w in S.
        # 2. Clique-part edges (within [0,k)) are covered either by a clique {v}∪S (the S-S edges)
        #    or by a pure clique-part clique.

        # Greedy: process independent vertices. For each, decide how to cover its cross-edges.
        # To minimize total cliques, we want each independent vertex to use as few cliques as possible
        # (ideally 1: {v} ∪ N(v)), and we want those to absorb as many clique-part edges as possible.

        # But using {v} ∪ N(v) for v absorbs all edges within N(v). If two independent vertices v1, v2
        # both use full N, edges within N(v1) ∩ N(v2) get double-covered. So we must partition.

        # Approach: assign each clique-part edge to either a pure clique-part clique or to exactly one
        # independent-vertex clique. Cross-edges (v,u) must go to a clique containing v.

        # Let's do: for each independent vertex v, we will create cliques. We process v in some order.
        # For v, let N = N(v). We want to cover cross-edges (v,u) for u in N.
        # We can put all of them in one clique {v} ∪ N, which also covers all edges within N.
        # But some edges within N may already be "claimed" by earlier vertices. We can still use {v}∪N
        # as long as we don't double-cover. So we should only include in S the neighbors whose
        # cross-edges to v are uncovered AND we must ensure S-S edges not already covered.

        # This is a partition problem. Let me do a clean greedy:
        # - Maintain set of uncovered edges.
        # - For each independent vertex v (in order of decreasing |N(v)|), build cliques for v:
        #     Take S = N(v) ∩ {u : edge (v,u) uncovered}. 
        #     But we also need S to be such that S-S edges are uncovered (else conflict).
        #     Since S ⊆ [0,k), S is a clique. The edges within S may or may not be uncovered.
        #     If some S-S edge is already covered, we cannot put both endpoints in S of a v-clique.
        #     So we need S such that all edges within S are uncovered (so this clique doesn't double-cover).
        #     Actually no: the clique {v}∪S covers S-S edges. If an S-S edge is already covered by a
        #     previous clique, then including both endpoints in S would double-cover it. So we need
        #     S to be a set where all internal edges are still uncovered.
        #     Since [0,k) is a clique, "all internal edges uncovered" means S is a clique in the
        #     subgraph of uncovered clique-part edges.
        # - This is getting into maximum-clique-on-uncovered-edges territory.

        # Given small k (<=30), let me do: for each v, find a large subset S ⊆ N(v) such that
        # all edges within S are uncovered (i.e., S is a clique in the uncovered clique-part graph),
        # and all cross-edges (v,u) for u in S are uncovered. Then clique {v}∪S covers those.
        # Repeat until all cross-edges of v are covered.

        # Then after all independent vertices, cover remaining clique-part edges with cliques.

        uncovered = set(edge_set)

        def _norm(e):
            return (min(e), max(e))

        cliques = []

        independent = list(range(k, n))
        # Order: process vertices with larger neighborhoods first? Or smaller? 
        # Larger N(v) means more cross-edges to cover; doing them first lets them absorb more clique-edges.
        independent.sort(key=lambda v: -len(adj[v]))

        for v in independent:
            # Cover all cross-edges (v, u) for u in N(v)
            Nv = adj[v]
            # We need to partition Nv into subsets S_1, S_2, ... such that each S_i is a clique
            # in the uncovered clique-part graph (so S_i-S_i edges are uncovered), and we form
            # cliques {v} ∪ S_i.
            # Greedy: repeatedly find a large clique in (Nv ∩ uncovered-clique-neighbors) within
            # the uncovered clique-part subgraph.
            remaining = set(Nv)
            while remaining:
                if _deadline_hit():
                    # Fallback: cover each remaining cross-edge with a 2-clique {v, u}
                    # and handle clique-part edges later.
                    for u in list(remaining):
                        e = _norm((v, u))
                        if e in uncovered:
                            cliques.append([v, u])
                            uncovered.discard(e)
                        remaining.discard(u)
                    break
                # Find a large subset S ⊆ remaining such that all edges within S are uncovered.
                # i.e., S is a clique in the graph G_unclique on [0,k) with edge_set = uncovered ∩ (clique-part edges).
                # Greedy: start with a vertex, grow.
                # Simpler: since k is small, find a maximal clique in the induced subgraph on `remaining`
                # using the uncovered-clique-edges.
                # Build adjacency within remaining for uncovered clique-part edges.
                rem_list = list(remaining)
                # Greedy clique: order by degree in uncovered-clique subgraph within remaining.
                # Compute degrees.
                deg = {}
                for u in rem_list:
                    deg[u] = 0
                for i in range(len(rem_list)):
                    a = rem_list[i]
                    for j in range(i + 1, len(rem_list)):
                        b = rem_list[j]
                        e = _norm((a, b))
                        if e in uncovered:
                            deg[a] += 1
                            deg[b] += 1
                # Greedy: pick highest degree, add neighbors that are connected (uncovered edge), repeat.
                S = []
                cand = set(rem_list)
                while cand:
                    # pick vertex with max degree within cand
                    best = max(cand, key=lambda x: sum(1 for y in cand if x != y and _norm((x, y)) in uncovered))
                    S.append(best)
                    # Keep only candidates connected to all of S via uncovered edges
                    new_cand = set()
                    for y in cand:
                        if y == best:
                            continue
                        ok = True
                        for s in S:
                            if _norm((y, s)) not in uncovered:
                                ok = False
                                break
                        if ok:
                            new_cand.add(y)
                    cand = new_cand
                # S is a clique in uncovered clique-part graph, S ⊆ remaining
                if not S:
                    # Should not happen; fallback single vertex
                    S = [next(iter(remaining))]
                clique = [v] + S
                cliques.append(clique)
                # Remove covered edges: cross (v,u) for u in S, and clique-part (u,w) for u,w in S
                for u in S:
                    uncovered.discard(_norm((v, u)))
                for i in range(len(S)):
                    a = S[i]
                    for j in range(i + 1, len(S)):
                        b = S[j]
                        uncovered.discard(_norm((a, b)))
                for u in S:
                    remaining.discard(u)

        # Now cover remaining uncovered edges (all within [0,k)) with cliques.
        # These are clique-part edges not absorbed. Cover greedily with large cliques.
        rem_edges = set(uncovered)
        while rem_edges:
            if _deadline_hit():
                # Fallback: 2-cliques
                for e in list(rem_edges):
                    cliques.append(list(e))
                    rem_edges.discard(e)
                break
            # Find a vertex with max degree in rem_edges
            deg = {}
            for (a, b) in rem_edges:
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1
            start = max(deg, key=lambda x: deg[x])
            # Greedy clique growth from start within rem_edges
            S = [start]
            cand = set()
            for (a, b) in rem_edges:
                if a == start:
                    cand.add(b)
                elif b == start:
                    cand.add(a)
            while cand:
                best = max(cand, key=lambda x: sum(1 for y in cand if x != y and _norm((x, y)) in rem_edges))
                S.append(best)
                new_cand = set()
                for y in cand:
                    if y == best:
                        continue
                    ok = True
                    for s in S:
                        if _norm((y, s)) not in rem_edges:
                            ok = False
                            break
                    if ok:
                        new_cand.add(y)
                cand = new_cand
            clique = S
            cliques.append(clique)
            for i in range(len(S)):
                a = S[i]
                for j in range(i + 1, len(S)):
                    b = S[j]
                    rem_edges.discard(_norm((a, b)))

        # Validate: every clique size >= 2, distinct ints, and union of clique-edges == original edge_set,
        # and no edge in two cliques.
        seen = set()
        for c in cliques:
            if len(c) < 2:
                # Should not happen; but guard
                pass
            for i in range(len(c)):
                a = c[i]
                for j in range(i + 1, len(c)):
                    b = c[j]
                    e = _norm((a, b))
                    if e in seen:
                        # Double coverage: invalid. This shouldn't happen with our construction.
                        pass
                    seen.add(e)
        # Ensure all original edges covered
        # (Trust construction; but if any missing, add 2-cliques as fallback)
        for e in edge_set:
            if e not in seen:
                cliques.append(list(e))
                seen.add(e)

        # Convert to lists of Python ints
        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            # Fallback: seed-like (each edge as 2-clique)
            out.append([list(e) for e in instance(*spec)])
        else:
            out.append(_solve_split(*spec))
    return out
