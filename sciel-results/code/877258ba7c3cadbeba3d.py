"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill

    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        n = len(adj)
        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return

            # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
            candidates = list(P | X)
            if not candidates:
                return
            pivot = max(candidates, key=lambda u: len(P & adj[u]))

            # Recurse on P \ N(pivot)
            for v in list(P - adj[pivot]):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return max_cliques

    def greedy_set_cover(cliques, n):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        # Map each vertex to the set of clique indices it covers
        # Only consider cliques of size >= 2
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        if not relevant_cliques:
            return set()

        # For each vertex, which cliques does it hit?
        vertex_to_cliques = {v: set() for v in range(n)}
        for idx, clique in enumerate(relevant_cliques):
            for v in clique:
                vertex_to_cliques[v].add(idx)

        uncovered = set(range(len(relevant_cliques)))
        selected = set()

        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = len(vertex_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.add(best_v)
            uncovered -= vertex_to_cliques[best_v]

        return selected

    def local_search_remove(selected, cliques, n, iterations=200):
        """Try to remove vertices from selected if coverage is maintained."""
        selected = set(selected)
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        if not relevant_cliques:
            return selected

        # Precompute for each clique which vertices in selected hit it
        for _ in range(iterations):
            made_change = False
            for v in list(selected):
                # Check if removing v still covers all cliques
                temp_selected = selected - {v}
                all_covered = True
                for clique in relevant_cliques:
                    if not (clique & temp_selected):
                        all_covered = False
                        break
                if all_covered:
                    selected = temp_selected
                    made_change = True
            if not made_change:
                break
        return selected

    def local_search_swap(selected, cliques, n, iterations=100):
        """Try to swap vertices: remove one, add another if it reduces size or same size but different."""
        selected = set(selected)
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        if not relevant_cliques:
            return selected

        for _ in range(iterations):
            made_change = False
            for v_remove in list(selected):
                temp = selected - {v_remove}
                # Find uncovered cliques after removal
                uncovered = [c for c in relevant_cliques if not (c & temp)]
                if not uncovered:
                    # Can just remove
                    selected = temp
                    made_change = True
                    continue
                # Try to add one vertex that covers all uncovered cliques
                # Find intersection of all uncovered cliques
                if len(uncovered) == 1:
                    candidates = uncovered[0] - selected
                else:
                    candidates = set(uncovered[0])
                    for c in uncovered[1:]:
                        candidates &= c
                    candidates -= selected
                for v_add in candidates:
                    new_selected = temp | {v_add}
                    if len(new_selected) < len(selected):
                        # Verify full coverage
                        ok = all(c & new_selected for c in relevant_cliques)
                        if ok:
                            selected = new_selected
                            made_change = True
                            break
                if made_change:
                    break
            if not made_change:
                break
        return selected

    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch(adj)

        if not cliques:
            return []

        # Initial greedy solution
        selected = greedy_set_cover(cliques, n)
        selected = local_search_remove(selected, cliques, n, iterations=300)
        selected = local_search_swap(selected, cliques, n, iterations=200)

        # Multiple restarts with different randomization in greedy
        best = selected
        for restart in range(50):
            if time.time() > deadline:
                break
            # Randomized greedy: add small random perturbation
            rng = random.Random(seed + restart * 1000)
            relevant_cliques = [c for c in cliques if len(c) >= 2]
            if not relevant_cliques:
                break

            vertex_to_cliques = {v: set() for v in range(n)}
            for idx, clique in enumerate(relevant_cliques):
                for v in clique:
                    vertex_to_cliques[v].add(idx)

            uncovered = set(range(len(relevant_cliques)))
            sel = set()

            while uncovered:
                # Score each candidate vertex
                scores = {}
                for v in range(n):
                    if v in sel:
                        continue
                    count = len(vertex_to_cliques[v] & uncovered)
                    if count > 0:
                        # Add random tie-breaking
                        scores[v] = count + rng.random() * 0.1
                if not scores:
                    break
                best_v = max(scores, key=scores.get)
                sel.add(best_v)
                uncovered -= vertex_to_cliques[best_v]

            sel = local_search_remove(sel, cliques, n, iterations=200)
            sel = local_search_swap(sel, cliques, n, iterations=100)

            if len(sel) < len(best):
                best = sel

        return sorted(best)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Return best effort: empty lists for remaining
            while len(results) < 3:
                results.append([])
            break
        results.append(solve_instance(n, p, seed))

    while len(results) < 3:
        results.append([])

    return results
# EVOLVE-BLOCK-END
