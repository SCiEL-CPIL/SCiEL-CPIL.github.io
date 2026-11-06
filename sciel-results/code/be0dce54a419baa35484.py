"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

def construct():
    sys.setrecursionlimit(100000)
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def generate_graph(n, p, seed):
        import random
        rng = random.Random(seed)
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
            # Choose pivot
            if P:
                pivot = max(P, key=lambda v: len(P & adj[v]))
                candidates = P - adj[pivot]
            else:
                candidates = P

            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)

        bk(frozenset(), set(range(n)), frozenset())
        return cliques

    def find_min_transversal(adj, n, cliques, time_limit):
        """Find minimum set of vertices hitting all maximal cliques of size >= 2."""
        if not cliques:
            return set()

        # Greedy initialization: pick vertices that cover the most uncovered cliques
        # Build incidence: for each vertex, which cliques it's in
        vert_cliques = defaultdict(list)
        for ci, clique in enumerate(cliques):
            for v in clique:
                vert_cliques[v].append(ci)

        # Greedy set cover
        uncovered = set(range(len(cliques)))
        selected = set()

        while uncovered:
            # Pick vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = sum(1 for ci in vert_cliques[v] if ci in uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            selected.add(best_v)
            for ci in vert_cliques[best_v]:
                if ci in uncovered:
                    uncovered.discard(ci)

        # Local search: try to remove redundant vertices
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for v in list(selected):
                # Check if removing v still covers all cliques
                if all(any(u in selected for u in clique) for clique in cliques if v in clique):
                    selected.discard(v)
                    improved = True

        # Try swaps: replace one vertex with another that covers more
        for _ in range(50):  # limited iterations
            if time.time() > time_limit:
                break
            improved = False
            for v in list(selected):
                # For each vertex not in selected, try swapping
                for u in range(n):
                    if u in selected:
                        continue
                    new_sel = (selected - {v}) | {u}
                    # Check coverage
                    ok = True
                    for clique in cliques:
                        if not any(x in new_sel for x in clique):
                            ok = False
                            break
                    if ok:
                        # Only accept if same size (we're trying to maintain minimality)
                        # or if it enables further reduction
                        selected = new_sel
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        # Final redundancy removal
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for v in list(selected):
                if all(any(u in selected for u in clique) for clique in cliques if v in clique):
                    selected.discard(v)
                    improved = True

        return selected

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE - 10:
            # Fallback: return all non-isolated vertices
            adj = generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue

        adj = generate_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        remaining = DEADLINE - time.time()
        if remaining < 5:
            # Not enough time, use greedy only
            vert_cliques = defaultdict(list)
            for ci, clique in enumerate(cliques):
                for v in clique:
                    vert_cliques[v].append(ci)
            uncovered = set(range(len(cliques)))
            selected = set()
            while uncovered:
                best_v = None
                best_count = -1
                for v in range(n):
                    if v in selected:
                        continue
                    count = sum(1 for ci in vert_cliques[v] if ci in uncovered)
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_v is None or best_count == 0:
                    break
                selected.add(best_v)
                for ci in vert_cliques[best_v]:
                    if ci in uncovered:
                        uncovered.discard(ci)
            results.append(sorted(selected))
        else:
            sel = find_min_transversal(adj, n, cliques, time.time() + min(remaining - 5, 30))
            results.append(sorted(sel))

    return results
# EVOLVE-BLOCK-END
