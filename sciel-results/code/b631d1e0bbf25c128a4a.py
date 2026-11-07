"""Clique partition of split graphs via Max-Bipartite-Matching for Type-B cliques + Max-Clique cover."""
import time
from random import Random
from collections import deque


def construct():
    _T0 = time.time()
    _DEADLINE = 140.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _edge(a, b):
        return (a, b) if a < b else (b, a)

    def _max_clique_bronk(adj, n):
        """Bron-Kerbosch with pivot to find a maximum clique."""
        best = set()

        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = set(R)
                return
            # Prune: even adding all P can't beat best
            if len(R) + len(P) <= len(best):
                return
            # Pivot
            pivot = max(P | X, key=lambda u: len(adj[u] & P)) if (P | X) else None
            candidates = list(P - adj[pivot]) if pivot is not None else list(P)
            for v in candidates:
                R.add(v)
                bk(R, P & adj[v], X & adj[v])
                R.discard(v)
                P.discard(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return best

    def _hopcroft_karp(adj, n_left, n_right):
        """
        Hopcroft-Karp algorithm for maximum bipartite matching.
        adj: list of lists, adj[i] contains indices of right nodes connected to left node i.
        n_left: number of left nodes.
        n_right: number of right nodes.
        Returns matching array: matchR[j] = i means right node j is matched to left node i.
        """
        matchL = [-1] * n_left
        matchR = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if matchL[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    mu = matchR[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == -1:
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found

        def dfs(u):
            for v in adj[u]:
                mu = matchR[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = -1
            return False

        while bfs():
            for u in range(n_left):
                if matchL[u] == -1:
                    dfs(u)
        return matchL, matchR

    def _cover_cc_edges(remaining_cc, k):
        """Cover C-C edges with C-only cliques using max clique extraction."""
        if not remaining_cc:
            return []
        cliques = []
        remaining = set(remaining_cc)
        while remaining:
            # Build adjacency on active vertices
            active = set()
            for (a, b) in remaining:
                active.add(a)
                active.add(b)
            if not active:
                break
            active_list = sorted(active)
            idx = {v: i for i, v in enumerate(active_list)}
            m = len(active_list)
            adj = [set() for _ in range(m)]
            for (a, b) in remaining:
                ia, ib = idx[a], idx[b]
                adj[ia].add(ib)
                adj[ib].add(ia)
            mc = _max_clique_bronk(adj, m)
            if not mc:
                # Fallback: pick any edge
                a, b = next(iter(remaining))
                mc = {idx[a], idx[b]}
            cl = sorted(active_list[i] for i in mc)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    remaining.discard(_edge(cl[i], cl[j]))
            cliques.append(cl)
        return cliques

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add(_edge(a, b))

        # All C-C edges
        all_cc = set(_edge(u, v) for u in range(k) for v in range(u + 1, k))

        # Build adjacency: for each I-vertex, which C-vertices it connects to
        adj_c = {}
        for w in range(k, n):
            adj_c[w] = set(u for u in range(k) if _edge(u, w) in edge_set)

        # Setup Bipartite Graph for Matching
        # Left: C-C edges. Right: I-vertices.
        # We need to map C-C edges to indices for the matching algorithm.
        cc_edges_list = list(all_cc)
        cc_index = {e: i for i, e in enumerate(cc_edges_list)}
        n_left = len(cc_edges_list)
        n_right = n - k
        
        # adj_match[left_idx] = list of right_idx (w - k) such that (u,v,w) is a triangle
        adj_match = [[] for _ in range(n_left)]
        
        # For each C-C edge (u,v), find all w such that w is connected to u and v
        for i, e in enumerate(cc_edges_list):
            u, v = e
            for w in range(k, n):
                if u in adj_c[w] and v in adj_c[w]:
                    # w is a valid partner for edge e
                    adj_match[i].append(w - k)

        # Run Hopcroft-Karp
        matchL, matchR = _hopcroft_karp(adj_match, n_left, n_right)
        
        # Extract matched triangles
        type_b_cliques = []
        covered_cc = set()
        covered_cross = set()
        
        for i, w_idx in enumerate(matchL):
            if w_idx != -1:
                e = cc_edges_list[i]
                u, v = e
                w = w_idx + k
                type_b_cliques.append([u, v, w])
                covered_cc.add(e)
                covered_cross.add(_edge(u, w))
                covered_cross.add(_edge(v, w))

        # Cover remaining C-C edges with C-only cliques
        remaining_cc = all_cc - covered_cc
        c_only_cliques = _cover_cc_edges(remaining_cc, k)

        # Cover remaining cross edges as 2-cliques
        cross_cliques = []
        for w in range(k, n):
            for u in range(k):
                e = _edge(u, w)
                if e in edge_set and e not in covered_cross:
                    cross_cliques.append([u, w])
                    covered_cross.add(e)

        cliques = type_b_cliques + c_only_cliques + cross_cliques

        # Validate: every clique must be a valid clique with at least 2 vertices
        validated = []
        for cl in cliques:
            if len(cl) < 2:
                continue
            valid = True
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    if _edge(cl[i], cl[j]) not in edge_set:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                validated.append(cl)

        return validated

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            edge_set = set()
            for (a, b) in edges:
                edge_set.add(_edge(a, b))
            cliques = [list(range(k))]
            for w in range(k, n):
                for u in range(k):
                    if _edge(u, w) in edge_set:
                        cliques.append([u, w])
            results.append(cliques)
            continue
        results.append(solve_one(*spec))

    out = []
    for clist in results:
        nl = []
        for c in clist:
            nl.append([int(x) for x in c])
        out.append(nl)
    return out
