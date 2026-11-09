import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 135  # seconds, leave 15s margin for safety

    def random_graph_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, adj):
        cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return

            if P or X:
                pivot_set = P | X
                pivot = max(pivot_set, key=lambda v: len(P & adj[v]))
                candidates = P - adj[pivot]
            else:
                candidates = P.copy()

            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                P.remove(v)
                X.add(v)
                bk(R, P_new, X_new)
                R.remove(v)
                P.add(v)
                X.remove(v)

        bk(set(), set(range(n)), set())
        return cliques

    def greedy_hitting_set(cliques, n):
        if not cliques:
            return set()

        vertex_to_cliques = {v: [] for v in range(n)}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].append(i)

        uncovered = set(range(len(cliques)))
        selected = set()

        while uncovered:
            if time.time() - start_time > DEADLINE:
                break
            best_v = None
            best_count = 0
            for v in range(n):
                if not vertex_to_cliques[v]:
                    continue
                count = sum(1 for ci in vertex_to_cliques[v] if ci in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v

            if best_v is None or best_count == 0:
                break

            selected.add(best_v)
            for ci in vertex_to_cliques[best_v]:
                uncovered.discard(ci)

        return selected

    def try_remove_redundant(selected, cliques, n):
        if not selected:
            return selected

        selected = set(selected)
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                redundant = True
                for clique in cliques:
                    if v in clique:
                        if not (clique & (selected - {v})):
                            redundant = False
                            break
                if redundant:
                    selected.discard(v)
                    changed = True

        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > DEADLINE:
            results.append(list(range(n)))
            continue

        edges = random_graph_edges(n, p, seed)

        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        cliques = bron_kerbosch_maximal_cliques(n, adj)

        if not cliques:
            results.append([])
            continue

        selected = greedy_hitting_set(cliques, n)
        selected = try_remove_redundant(selected, cliques, n)

        results.append(sorted(selected))

    return results
