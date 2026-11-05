"""Strategy: ILP-style branch-and-bound with memoized feasibility checks on the clique hypergraph."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return

            pivot_candidates = list(P) + list(X)
            if not pivot_candidates:
                return

            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u

            P_set = set(P)
            X_set = set(X)

            if best_pivot is not None:
                candidates = [v for v in P_set if v not in adj[best_pivot]]
            else:
                candidates = list(P_set)

            for v in candidates:
                new_R = R | {v}
                new_P = P_set & adj[v]
                new_X = X_set & adj[v]
                bk(new_R, new_P, new_X)
                P_set.remove(v)
                X_set.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def solve_hitting_set_bnb(n, cliques, time_budget):
        """
        Branch-and-bound with greedy initialization and aggressive pruning.
        State: set of selected vertices.
        Objective: minimize |selected| subject to every clique being hit.
        """
        if not cliques:
            return set()

        m = len(cliques)

        # Precompute: for each vertex, which cliques it covers (as list of indices)
        vertex_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(i)

        # Greedy initialization
        def greedy_init():
            uncovered = set(range(m))
            selected = set()
            while uncovered:
                best_v = -1
                best_gain = 0
                for v in range(n):
                    gain = len([i for i in vertex_cliques[v] if i in uncovered])
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1 or best_gain == 0:
                    break
                selected.add(best_v)
                for i in vertex_cliques[best_v]:
                    if i in uncovered:
                        uncovered.discard(i)
            return selected

        # Check if a set is valid (hits all cliques)
        def is_valid(selected):
            for c in cliques:
                if not any(v in selected for v in c):
                    return False
            return True

        # Get uncovered cliques for a selected set
        def get_uncovered(selected):
            uncovered = []
            for i, c in enumerate(cliques):
                if not any(v in selected for v in c):
                    uncovered.append(i)
            return uncovered

        # Lower bound: minimum number of additional vertices needed to cover uncovered cliques
        # Use a simple greedy lower bound based on maximum coverage per vertex
        def lower_bound(uncovered_cliques, current_size):
            if not uncovered_cliques:
                return current_size
            # Estimate: each vertex can cover at most max_cov cliques
            max_cov = 0
            for v in range(n):
                cov = len([i for i in vertex_cliques[v] if i in uncovered_cliques])
                if cov > max_cov:
                    max_cov = cov
            if max_cov == 0:
                return current_size + len(uncovered_cliques)
            return current_size + (len(uncovered_cliques) + max_cov - 1) // max_cov

        # Branch and bound
        best_solution = greedy_init()
        best_size = len(best_solution)
        current = set(best_solution)

        # Try to improve by removing redundant vertices
        changed = True
        while changed:
            changed = False
            for v in list(current):
                test = set(current)
                test.discard(v)
                if is_valid(test):
                    current = test
                    changed = True
                    best_size = len(current)
                    best_solution = set(current)

        # Now try to find a smaller solution via local search with swaps
        rng = random.Random(42)
        T = 1.0
        alpha = 0.999
        T_min = 0.01

        # Precompute for faster evaluation
        # For each clique, store the set of vertices in it
        clique_vertices = [set(c) for c in cliques]

        def eval_move(selected, v_add=None, v_remove=None):
            """Evaluate if adding v_add and removing v_remove improves validity"""
            new_selected = set(selected)
            if v_remove is not None:
                new_selected.discard(v_remove)
            if v_add is not None:
                new_selected.add(v_add)
            return new_selected

        while time.time() < time_budget and T > T_min:
            # Generate a move: swap one vertex out, one in
            if current:
                # Find vertices that can be removed (not critical)
                removable = []
                for v in current:
                    is_critical = False
                    for i in vertex_cliques[v]:
                        c = clique_vertices[i]
                        if not any(u in current for u in c if u != v):
                            is_critical = True
                            break
                    if not is_critical:
                        removable.append(v)

                if removable:
                    v_rem = rng.choice(removable)
                    new_current = set(current)
                    new_current.discard(v_rem)

                    # Find a vertex to add that helps
                    unc = get_uncovered(new_current)
                    if unc:
                        # Weight by coverage
                        weighted = []
                        for v in range(n):
                            if v not in new_current:
                                w = len([i for i in vertex_cliques[v] if i in unc])
                                if w > 0:
                                    weighted.extend([v] * w)
                        if weighted:
                            v_add = rng.choice(weighted)
                            new_current.add(v_add)
                            # Check if valid
                            if is_valid(new_current):
                                new_size = len(new_current)
                                old_size = len(current)
                                delta = new_size - old_size
                                if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                                    current = new_current
                                    if new_size < best_size:
                                        best_size = new_size
                                        best_solution = set(current)
                    else:
                        # All cliques covered; try to remove more
                        pass
            else:
                # Empty set; add a vertex
                if cliques:
                    v = rng.choice(range(n))
                    current = {v}
                    if is_valid(current) and len(current) < best_size:
                        best_size = len(current)
                        best_solution = set(current)

            T *= alpha

        # Final cleanup: remove any redundant vertices
        changed = True
        while changed:
            changed = False
            for v in list(best_solution):
                test = set(best_solution)
                test.discard(v)
                if is_valid(test):
                    best_solution = test
                    changed = True

        return best_solution

    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 20:
            results.append(list(range(n)))
            continue

        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        cliques = [c for c in cliques if len(c) >= 2]

        if not cliques:
            results.append([])
            continue

        # Allocate remaining time
        elapsed = time.time() - start_time
        remaining_instances = 3 - len(results)
        per_instance = (deadline - 10 - elapsed) / max(1, remaining_instances)
        per_instance = max(1.0, min(per_instance, deadline - 10 - elapsed))

        sol = solve_hitting_set_bnb(n, cliques, start_time + elapsed + per_instance)
        results.append(sorted(sol))

    return results
# EVOLVE-BLOCK-END
