"""Clique partition of split graphs via greedy absorption + C-only cover."""
import time
from random import Random


def construct():
    _T0 = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _cc_edge(a, b):
        return (a, b) if a < b else (b, a)

    def _cover_remaining_cc(remaining, k):
        """Cover the remaining C-C edges with C-only cliques.
        Greedy: repeatedly extract a maximal clique from the remaining graph.
        Since remaining is a subgraph of K_k, a maximal clique is a set of
        mutually-adjacent vertices in remaining. Use greedy: start from the
        vertex of highest degree in remaining, grow by adding vertices adjacent
        to all current members.
        """
        if not remaining:
            return []
        # adjacency within remaining
        adj = [set() for _ in range(k)]
        for (a, b) in remaining:
            adj[a].add(b)
            adj[b].add(a)
        active = set()
        for (a, b) in remaining:
            active.add(a)
            active.add(b)
        cliques = []
        remaining = set(remaining)
        while remaining:
            # pick a vertex with max degree in remaining among active
            best = None
            best_deg = -1
            for v in active:
                deg = len(adj[v] & active)
                if deg > best_deg:
                    best_deg = deg
                    best = v
            if best is None:
                break
            # grow a clique from best
            clique = {best}
            candidates = set(adj[best] & active)
            for u in list(candidates):
                # u can join if adjacent to all in clique
                if all(w in adj[u] for w in clique):
                    clique.add(u)
                    # remove u from candidates of others? keep simple
            # Actually do proper greedy: iteratively add
            clique = {best}
            frontier = set(adj[best] & active)
            while frontier:
                # pick u in frontier adjacent to all in clique
                added = None
                for u in sorted(frontier):
                    if all(w in adj[u] for w in clique):
                        added = u
                        break
                if added is None:
                    break
                clique.add(added)
                frontier = {x for x in frontier if x in adj[added]}
            # remove all edges within clique from remaining
            cl = sorted(clique)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    remaining.discard(_cc_edge(cl[i], cl[j]))
            # remove clique vertices from active
            active -= clique
            cliques.append(cl)
        return cliques

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add(_cc_edge(a, b))

        # N_C[v] = neighbors of I-vertex v in C
        N_C = {}
        for v in range(k, n):
            N_C[v] = frozenset(u for u in range(k) if _cc_edge(u, v) in edge_set)

        # all C-C edges
        all_cc = set(_cc_edge(u, v) for u in range(k) for v in range(u + 1, k))

        # Greedy: process I-vertices in order of descending d_v.
        # For each, if d_v >= 2 and the C-C edges within N_C[v] are all still
        # available (not consumed by a previously-chosen big clique), absorb them
        # and create one clique {v} ∪ N_C[v]. Otherwise, create d_v separate
        # 2-cliques {v, u}.
        used_cc = set()
        cliques = []

        order = sorted(range(k, n), key=lambda v: -len(N_C[v]))
        for v in order:
            S = N_C[v]
            d = len(S)
            if d == 0:
                continue
            if d >= 2:
                # check if all C-C edges within S are still free
                inner_edges = set(_cc_edge(a, b) for a in S for b in S if a < b)
                if not (inner_edges & used_cc):
                    # absorb
                    used_cc |= inner_edges
                    cliques.append([v] + sorted(S))
                    continue
            # fall back: separate 2-cliques
            for u in sorted(S):
                cliques.append([u, v])

        # cover remaining C-C edges
        remaining_cc = all_cc - used_cc
        c_only = _cover_remaining_cc(remaining_cc, k)
        for cl in c_only:
            cliques.append(cl)

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue
        results.append(solve_one(*spec))

    # convert to plain python int lists
    out = []
    for clist in results:
        nl = []
        for c in clist:
            nl.append([int(x) for x in c])
        out.append(nl)
    return out
