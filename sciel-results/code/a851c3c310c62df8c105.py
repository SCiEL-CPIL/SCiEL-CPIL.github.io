"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def build_graph(n, p, seed):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []

        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            # Choose pivot from P ∪ X to minimize branching
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                R2 = R | {v}
                P2 = P & adj[v]
                X2 = X & adj[v]
                expand(R2, P2, X2)
                P -= {v}
                X |= {v}

        expand(frozenset(), frozenset(range(n)), frozenset())
        return maximal_cliques

    def greedy_hitting_set(cliques, n):
        """
        Greedy set cover: each vertex covers all cliques it belongs to.
        Returns a hitting set (list of vertices).
        """
        cliques = list(cliques)
        num_cliques = len(cliques)
        if num_cliques == 0:
            return []

        # For each vertex, find which cliques it hits
        vertex_to_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(i)

        uncovered = set(range(num_cliques))
        selected = []

        while uncovered:
            # Find vertex that covers most uncovered cliques
            best_v = None
            best_count = -1
            for v in range(n):
                count = len(vertex_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count <= 0:
                break
            selected.append(best_v)
            uncovered -= vertex_to_cliques[best_v]

        return selected

    def local_search_improve(hitting_set, cliques, n, time_budget=5.0):
        """
        Try to remove vertices from the hitting set if possible,
        and use 2-exchange moves to reduce size.
        """
        hitting_set = list(hitting_set)
        end_time = time.time() + time_budget
        improved = True

        # Precompute clique membership for fast checking
        # For each clique, store the set of vertices in it
        clique_sets = [set(c) for c in cliques]
        num_cliques = len(clique_sets)

        def is_valid(hs):
            hs_set = set(hs)
            for cs in clique_sets:
                if not (cs & hs_set):
                    return False
            return True

        while improved and time.time() < end_time:
            improved = False

            # Try removing each vertex
            for i in range(len(hitting_set)):
                if time.time() > end_time:
                    break
                candidate = hitting_set[:i] + hitting_set[i+1:]
                if is_valid(candidate):
                    hitting_set = candidate
                    improved = True
                    break

            if improved:
                continue

            # Try 2-exchange: remove two vertices, add one
            if len(hitting_set) >= 2:
                hs_set = set(hitting_set)
                # For each pair of vertices to remove, check if we can add one vertex to fix
                for i in range(len(hitting_set)):
                    if time.time() > end_time:
                        break
                    for j in range(i + 1, len(hitting_set)):
                        if time.time() > end_time:
                            break
                        removed = {hitting_set[i], hitting_set[j]}
                        remaining = hs_set - removed
                        # Find which cliques are now uncovered
                        uncovered = []
                        for idx, cs in enumerate(clique_sets):
                            if not (cs & remaining):
                                uncovered.append(idx)
                        if not uncovered:
                            # Both removals are valid, but we already tried single removals
                            continue
                        # Try adding each vertex not in removed
                        for v in range(n):
                            if v in removed:
                                continue
                            new_hs = remaining | {v}
                            if is_valid(new_hs):
                                hitting_set = list(new_hs)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

        return hitting_set

    def branch_and_bound_hitting_set(cliques, n, time_limit=10.0):
        """
        Branch and bound to find a minimal hitting set.
        Uses greedy solution as initial upper bound.
        """
        cliques = list(cliques)
        num_cliques = len(cliques)
        if num_cliques == 0:
            return []

        # Initial greedy solution
        best = greedy_hitting_set(cliques, n)
        best_size = len(best)
        if best_size <= 1:
            return best

        start_bb = time.time()
        best_solution = list(best)
        best_size = len(best)

        # Sort vertices by frequency in cliques (more frequent first)
        vertex_freq = [0] * n
        for clique in cliques:
            for v in clique:
                vertex_freq[v] += 1
        vertices_sorted = sorted(range(n), key=lambda v: -vertex_freq[v])

        # For each vertex, precompute which cliques it covers
        vertex_cliques = [set() for _ in range(n)]
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_cliques[v].add(i)

        # Convert cliques to frozensets for hashing
        cliques_fs = [frozenset(c) for c in cliques]

        def bb(uncovered, selected, depth):
            nonlocal best_solution, best_size
            if time.time() - start_bb > time_limit:
                return

            if not uncovered:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = list(selected)
                return

            # Lower bound: need at least ceil(|uncovered| / max_cover) more vertices
            max_cover = 0
            for v in vertices_sorted:
                cover = len(vertex_cliques[v] & uncovered)
                if cover > max_cover:
                    max_cover = cover
                if max_cover >= len(uncovered):
                    break
            if max_cover == 0:
                return
            lb = len(selected) + math.ceil(len(uncovered) / max_cover)
            if lb >= best_size:
                return

            # Branch on vertices that cover at least one uncovered clique
            # Pick a clique to branch on
            target_clique = next(iter(uncovered))
            # Vertices in this clique
            target_vertices = [v for v in cliques_fs[target_clique]
                               if len(vertex_cliques[v] & uncovered) > 0]
            if not target_vertices:
                return

            # Sort by coverage
            target_vertices.sort(key=lambda v: -len(vertex_cliques[v] & uncovered))

            for v in target_vertices:
                if time.time() - start_bb > time_limit:
                    return
                new_uncovered = uncovered - vertex_cliques[v]
                new_selected = selected | {v}
                bb(new_uncovered, new_selected, depth + 1)

        bb(set(range(num_cliques)), set(), 0)
        return best_solution

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]

    for n, p, seed in instances:
        if time.time() > deadline:
            # Return what we have so far with empty lists for remaining
            results.append([])
            continue

        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)

        if not cliques:
            results.append([])
            continue

        # Get a good initial solution via greedy
        initial = greedy_hitting_set(cliques, n)

        # Try branch and bound with time limit
        remaining = deadline - time.time()
        bb_time = min(25.0, max(1.0, remaining / 3))
        solution = branch_and_bound_hitting_set(cliques, n, time_limit=bb_time)

        # Local search to try to improve
        remaining = deadline - time.time()
        ls_time = min(10.0, max(0.5, remaining / 3))
        solution = local_search_improve(solution, cliques, n, time_budget=ls_time)

        results.append(sorted(solution))

    return results
# EVOLVE-BLOCK-END
