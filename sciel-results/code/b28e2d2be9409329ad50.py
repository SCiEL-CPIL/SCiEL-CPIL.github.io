def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges + cross)

    def is_clique(vertices, adj):
        vs = list(vertices)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                if vs[j] not in adj[vs[i]]:
                    return False
        return True

    def greedy_clique_partition(edge_set, n, adj):
        if not edge_set:
            return []
        remaining = set(edge_set)
        cliques = []
        while remaining:
            vertices = set()
            for u, v in remaining:
                vertices.add(u)
                vertices.add(v)
            deg = {v: 0 for v in vertices}
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            start = max(vertices, key=lambda v: deg[v])
            clique = {start}
            candidates = set(adj[start]) & vertices
            while candidates:
                best = None
                best_score = -1
                for c in candidates:
                    score = sum(1 for x in clique if x in adj[c])
                    if score > best_score:
                        best_score = score
                        best = c
                if best is None:
                    break
                clique.add(best)
                candidates = {c for c in candidates if best in adj[c]}
            clique_list = sorted(clique)
            clique_edges = set()
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    e = (clique_list[i], clique_list[j])
                    if e in remaining:
                        clique_edges.add(e)
            if not clique_edges:
                e = next(iter(remaining))
                clique_edges = {e}
                clique_list = [e[0], e[1]]
            remaining -= clique_edges
            cliques.append(clique_list)
        return cliques

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        cliques = []
        covered = set()
        for v in range(k, n):
            neighbors = sorted(adj[v])
            if not neighbors:
                continue
            candidate = [v] + neighbors
            if len(candidate) < 2:
                continue
            if not is_clique(candidate, adj):
                continue
            cand_edges = set()
            for i in range(len(candidate)):
                for j in range(i + 1, len(candidate)):
                    e = (min(candidate[i], candidate[j]), max(candidate[i], candidate[j]))
                    if e in edge_set:
                        cand_edges.add(e)
            if cand_edges & covered:
                continue
            covered |= cand_edges
            cliques.append(candidate)
        remaining = edge_set - covered
        if remaining:
            rem_cliques = greedy_clique_partition(remaining, n, adj)
            cliques.extend(rem_cliques)
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_one(n, k, p, seed)
        results.append([list(c) for c in cliques])
    return results
