"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill

    def random_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot u from P ∪ X to minimize branching
            union = P | X
            if union:
                # Pick pivot with max degree in P
                best_u = max(union, key=lambda u: len(adj[u] & P))
                candidates = P - adj[best_u]
            else:
                candidates = P
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_hitting_set(cliques, n):
        """Greedy: pick vertex covering most uncovered cliques."""
        remaining = [set(c) for c in cliques]  # cliques not yet hit
        selected = set()
        # Map vertex -> set of clique indices it belongs to
        while remaining:
            # Count how many remaining cliques each vertex hits
            counts = {}
            for i, c in enumerate(remaining):
                for v in c:
                    counts[v] = counts.get(v, 0) + 1
            # Pick vertex with max count
            best_v = max(counts, key=lambda v: (counts[v], -v))
            selected.add(best_v)
            # Remove all cliques hit by best_v
            remaining = [c for c in remaining if best_v not in c]
        return selected

    def local_search(selected, cliques, n, time_limit):
        """Improve via deletions and 2-opt swaps."""
        selected = set(selected)
        start = time.time()

        def is_valid(sel):
            return all(any(v in sel for v in c) for c in cliques)

        def try_delete(sel):
            """Try removing one vertex at a time."""
            for v in list(sel):
                new_sel = sel - {v}
                if is_valid(new_sel):
                    return new_sel, True
            return sel, False

        def try_swap(sel):
            """Try swapping one vertex out for another."""
            for v_out in list(sel):
                new_sel_no = sel - {v_out}
                # Find which cliques are now uncovered
                uncovered = [c for c in cliques if not any(x in new_sel_no for x in c)]
                if not uncovered:
                    return new_sel_no, True
                # Find a vertex that covers all uncovered cliques
                for v_in in range(n):
                    if v_in in sel:
                        continue
                    if all(v_in in c for c in uncovered):
                        return new_sel_no | {v_in}, True
            return sel, False

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try deletions
            new_sel, ok = try_delete(selected)
            if ok:
                selected = new_sel
                improved = True
                continue
            # Try swaps
            new_sel, ok = try_swap(selected)
            if ok:
                selected = new_sel
                improved = True
                continue
        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # Fallback: return all vertices as safety
            results.append(list(range(n)))
            continue

        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)

        if not cliques:
            results.append([])
            continue

        # Greedy initial solution
        greedy = greedy_hitting_set(cliques, n)

        # Local search with time budget
        per_instance_time = min(30, max(5, deadline - time.time() - 5))
        improved = local_search(greedy, cliques, n, time.time() + per_instance_time)

        results.append(sorted(improved))

    return results
# EVOLVE-BLOCK-END
