"""Clique partition of split graphs via ILP for Type-B triangles + greedy K-cover."""
import time
from random import Random
from collections import defaultdict
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def construct():
    _T0 = time.time()
    _DEADLINE = 135.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _greedy_clique_cover(edges, n_vertices, seed=0, deadline=None):
        """Greedy maximal clique extraction for K-subgraph edges."""
        if not edges:
            return []
        rng = Random(seed)
        rem = set(edges)
        cliques = []
        while rem:
            if deadline and time.time() > deadline:
                for e in rem:
                    cliques.append([e[0], e[1]])
                break
            # Build adjacency from remaining edges
            adj = defaultdict(set)
            active = set()
            for (a, b) in rem:
                adj[a].add(b)
                adj[b].add(a)
                active.add(a)
                active.add(b)
            if not active:
                break
            # Greedy: pick vertex with max degree in remaining
            deg = {v: len(adj[v]) for v in active}
            max_v = max(active, key=lambda x: (deg[x], -x))
            # Expand clique greedily
            clique = [max_v]
            candidates = sorted(adj[max_v], key=lambda x: (-deg.get(x, 0), x))
            for cand in candidates:
                ok = True
                for cv in clique:
                    if (min(cv, cand), max(cv, cand)) not in rem:
                        ok = False
                        break
                if ok:
                    clique.append(cand)
            cliques.append(sorted(clique))
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    rem.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
        return cliques

    def _solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)

        # Identify K-edges and cross edges
        k_edges = []
        k_edge_set = set()
        cross_edges = []
        i_adj = {v: set() for v in range(k, n)}  # K-neighbors of each I-vertex
        for (u, v) in edges:
            if u < k and v < k:
                k_edges.append((u, v))
                k_edge_set.add((u, v))
            elif u < k and v >= k:
                cross_edges.append((u, v))
                i_adj[v].add(u)

        n_k = len(k_edges)
        n_i = n - k
        i_verts = list(range(k, n))

        # Build valid triangle pairs: (k_edge_idx, i_vert) where i_vert adjacent to both endpoints
        valid_tris = []  # list of (e_idx, i_idx)
        for ei, (u, v) in enumerate(k_edges):
            for ii, w in enumerate(i_verts):
                if u in i_adj[w] and v in i_adj[w]:
                    valid_tris.append((ei, ii))

        # Build valid pair assignments: (k_vert, i_vert) where edge exists
        valid_pairs = []
        for ii, w in enumerate(i_verts):
            for u in i_adj[w]:
                valid_pairs.append((u, ii))

        # ILP formulation:
        # Variables:
        #   x[0..n_valid_tris-1]: triangle selection
        #   y[0..n_valid_pairs-1]: pair selection
        # Objective: maximize sum(x) + sum(y)  [each saves 1 clique vs baseline]
        # Baseline: all cross edges as pairs + K-cover of all K-edges
        # With triangles: saves 2 cross edges per triangle, covers 1 K-edge
        # With pairs: saves 0 cross edges (still need pair), but... wait.
        #
        # Actually, let me reconsider the cost model:
        # Total cliques = (K-cover of uncovered K-edges) + (cross edges not in triangles)
        # A triangle covers 1 K-edge + 2 cross edges -> saves 2 from cross, 1 from K-cover
        # A pair covers 1 cross edge -> saves 1 from cross, 0 from K-cover
        # But K-cover is nonlinear. Let's use: minimize K_cover(remaining_K) + remaining_cross
        #
        # For the ILP, let's maximize: 2*sum(x) + sum(y) subject to:
        #   Each K-edge in at most 1 triangle: sum_{(ei,ii):ei=e} x[e,ii] <= 1
        #   Each I-vertex in at most 1 triangle: sum_{(ei,ii):ii=i} x[e,ii] <= 1
        #   Each (K-vert, I-vert) pair used at most once: trivially satisfied
        #   Each I-vertex in at most 1 pair: sum_{(u,ii):ii=i} y[u,ii] <= 1
        #   If I-vertex in a triangle, it can't also be in a pair (but it's in 2 pairs via triangle)
        #   Actually, if I-vertex is in a triangle with K-edge (u,v), then cross edges (u,w) and (v,w) 
        #   are covered by the triangle. We don't need separate pair cliques for them.
        #   So: if x[e,ii]=1, then y[u,ii]=0 and y[v,ii]=0 for the two K-verts in edge e.
        #
        # Let me simplify: the triangle already covers the cross edges, so we just need:
        #   For each I-vertex ii: sum of x over triangles involving ii <= 1
        #   For each K-vert u: if u is in a triangle with ii, then y[u,ii] = 0
        #   This coupling makes it harder. Let's just do:
        #   Each I-vertex in at most 1 clique total (triangle or pair)
        #   Each K-vert-I-vert pair in at most 1 clique
        #
        # Simpler approach: just optimize triangle selection, then handle pairs greedily.
        # The main savings come from triangles (save 2 cross + 1 K-edge cover).
        # Pairs are just the remaining cross edges.

        if not valid_tris:
            # No triangles possible, just greedy K-cover + all cross as pairs
            k_cliques = _greedy_clique_cover(k_edges, k, seed=42, deadline=_T0 + _DEADLINE)
            cliques = k_cliques + [[u, v] for (u, v) in cross_edges]
            return cliques

        n_tris = len(valid_tris)
        # Objective: maximize sum(x) -> minimize -sum(x)
        c = np.full(n_tris, -1.0)

        constraints = []

        # Constraint 1: Each K-edge in at most 1 triangle
        # Group by k_edge index
        tris_by_kedge = defaultdict(list)
        for ti, (ei, ii) in enumerate(valid_tris):
            tris_by_kedge[ei].append(ti)
        for ei, tis in tris_by_kedge.items():
            if len(tis) > 1:
                row = np.zeros(n_tris)
                for ti in tis:
                    row[ti] = 1.0
                constraints.append(LinearConstraint(row, -np.inf, 1.0))

        # Constraint 2: Each I-vertex in at most 1 triangle
        tris_by_i = defaultdict(list)
        for ti, (ei, ii) in enumerate(valid_tris):
            tris_by_i[ii].append(ti)
        for ii, tis in tris_by_i.items():
            if len(tis) > 1:
                row = np.zeros(n_tris)
                for ti in tis:
                    row[ti] = 1.0
                constraints.append(LinearConstraint(row, -np.inf, 1.0))

        bounds = Bounds(lb=np.zeros(n_tris), ub=np.ones(n_tris))
        integrality = np.ones(n_tris)

        if _time_up():
            # Fallback: no triangles
            k_cliques = _greedy_clique_cover(k_edges, k, seed=42)
            return k_cliques + [[u, v] for (u, v) in cross_edges]

        try:
            res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality,
                       options={"time_limit": max(1.0, _DEADLINE - (time.time() - _T0) - 5)})
            if res.status != 0:
                raise ValueError(f"ILP status {res.status}")
            x_vals = res.x
            selected_tris = [ti for ti in range(n_tris) if x_vals[ti] > 0.5]
        except Exception:
            selected_tris = []

        # Build final cliques
        cliques = []
        covered_k_edges = set()
        covered_i_verts = set()
        i_to_kedge = {}

        for ti in selected_tris:
            ei, ii = valid_tris[ti]
            u, v = k_edges[ei]
            w = i_verts[ii]
            cliques.append(sorted([u, v, w]))
            covered_k_edges.add((u, v))
            covered_i_verts.add(w)
            i_to_kedge[w] = (u, v)

        # Remaining K-edges
        rem_k_edges = [e for e in k_edges if e not in covered_k_edges]
        k_cliques = _greedy_clique_cover(rem_k_edges, k, seed=42, deadline=_T0 + _DEADLINE)
        cliques.extend(k_cliques)

        # Remaining cross edges (not covered by triangles)
        for (u, v) in cross_edges:
            if v in covered_i_verts:
                ku, kv = i_to_kedge[v]
                if u == ku or u == kv:
                    continue  # covered by triangle
            cliques.append([u, v])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if _time_up():
            # Emergency fallback
            edges = instance(n, k, p, seed)
            results.append([[u, v] for (u, v) in edges])
            continue
        cliques = _solve_instance(n, k, p, seed)
        results.append(cliques)

    return results
