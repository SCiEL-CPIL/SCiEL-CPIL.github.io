"""Strategy: Branch-and-bound on the clique-hypergraph with dominance pruning."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

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

    def solve_hitting_set(n, cliques, time_budget):
        """
        Branch-and-bound: find minimum set of vertices hitting all cliques.
        State: frozenset of uncovered clique indices.
        Pruning: dominance (if vertex A's clique set ⊆ vertex B's, A is never needed first).
        """
        if not cliques:
            return set()

        m = len(cliques)

        # Precompute: for each vertex, bitmask of cliques it covers
        # Use Python int as bitmask for speed (m is small, ~5-50)
        vertex_mask = [0] * n
        for i, c in enumerate(cliques):
            for v in c:
                vertex_mask[v] |= (1 << i)

        # Dominance reduction: if mask[a] is a subset of mask[b] (and a != b),
        # then a is dominated by b — we can remove a from consideration.
        # Actually, we keep b and discard a.
        # But careful: if both are needed for different reasons? No — if a's coverage
        # is a subset of b's, any solution using a can replace a with b (or b already covers it).
        # So a is never strictly necessary. Remove dominated vertices.

        # Build list of (vertex, mask) for vertices with non-zero mask
        active = [(v, vertex_mask[v]) for v in range(n) if vertex_mask[v] != 0]

        # Dominance: sort by popcount descending, then check subsets
        # If mask_a & ~mask_b == 0, then a is dominated by b
        active.sort(key=lambda x: bin(x[1]).count('1'), reverse=True)

        kept = []
        kept_masks = []
        for v, mask in active:
            dominated = False
            for km in kept_masks:
                if mask & ~km == 0:  # mask is subset of km
                    dominated = True
                    break
            if not dominated:
                kept.append(v)
                kept_masks.append(mask)

        # Now solve on the reduced set
        k = len(kept)
        if k == 0:
            return set()

        # Full target mask
        full_mask = (1 << m) - 1

        best_solution = None
        best_size = k + 1  # upper bound: take all kept vertices

        # Branch and bound
        # State: (uncovered_mask, current_selected_size, current_selected_set)
        # We use recursion with memoization on uncovered_mask to store best additional cost

        memo = {}  # uncovered_mask -> min additional vertices needed

        def popcount(x):
            return bin(x).count('1')

        def branch(uncovered, selected, depth):
            nonlocal best_solution, best_size

            if time.time() > time_budget:
                return

            if uncovered == 0:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                return

            # Lower bound: we need at least ceil(popcount(uncovered) / max_coverage) more vertices
            # But simpler: if len(selected) >= best_size, prune
            if len(selected) >= best_size:
                return

            # Memoization: if we've seen this uncovered state with a solution no worse, skip
            if uncovered in memo:
                if memo[uncovered] <= best_size - len(selected):
                    return

            # Choose the vertex that covers the most uncovered cliques
            # Among kept vertices
            best_v = -1
            best_gain = 0
            for idx, v in enumerate(kept):
                gain = popcount(kept_masks[idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_v = idx

            if best_v == -1 or best_gain == 0:
                # Can't cover remaining cliques with kept vertices
                # This shouldn't happen if we started with all vertices
                return

            # Branch 1: include best_v
            v = kept[best_v]
            new_uncovered = uncovered & ~kept_masks[best_v]
            selected.append(v)
            branch(new_uncovered, selected, depth + 1)
            selected.pop()

            # Branch 2: exclude best_v (try other vertices)
            # To avoid exponential blowup, we only exclude if there's an alternative
            # Actually, for correctness we must explore exclusion too.
            # But we can order: try including the best first, then exclude.
            # For exclusion, we try the next-best vertex.
            # Simpler: iterate over all vertices, try each as the next choice.
            # But that's what the greedy selection above does partially.
            # Let's do a proper branch: at each step, for each candidate vertex,
            # branch on including it. The first branch (best) is tried first.
            # For the "exclude" branch, we skip this vertex and try the next.

            # Alternative approach: at each node, pick the vertex covering the most uncovered,
            # then branch: include it, OR skip it (and solve the subproblem without it).
            # Skipping means we must cover its cliques with other vertices.

            # Let's implement: after trying include, try skip.
            # Skip: remove best_v from consideration, solve remaining.
            # This is tricky with the current structure. Let's use a different formulation.

            # Better: recursive function that takes a set of available vertices and uncovered mask.
            # But for speed, let's just try all vertices as the next pick, in order of coverage.

            # Actually, let's restart with a cleaner B&B:

        # Cleaner B&B implementation
        def bb_solve():
            nonlocal best_solution, best_size

            # Sort kept vertices by mask popcount descending
            order = sorted(range(k), key=lambda i: popcount(kept_masks[i]), reverse=True)

            def recurse(idx, uncovered, selected):
                nonlocal best_solution, best_size

                if time.time() > time_budget:
                    return

                if uncovered == 0:
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_solution = set(selected)
                    return

                if len(selected) >= best_size:
                    return

                # Lower bound: each remaining vertex covers at most max_remaining_cover cliques
                # Simple bound: need at least 1 more vertex
                if idx >= k:
                    return

                # Option 1: skip vertex at order[idx]
                recurse(idx + 1, uncovered, selected)

                # Option 2: include vertex at order[idx]
                v = kept[order[idx]]
                mask = kept_masks[order[idx]]
                if mask & uncovered:  # only include if it helps
                    new_uncovered = uncovered & ~mask
                    selected.append(v)
                    recurse(idx + 1, new_uncovered, selected)
                    selected.pop()

            # Initial upper bound: greedy
            def greedy_upper_bound():
                unc = full_mask
                sol = set()
                while unc:
                    best_i = -1
                    best_gain = 0
                    for i in range(k):
                        gain = popcount(kept_masks[i] & unc)
                        if gain > best_gain:
                            best_gain = gain
                            best_i = i
                    if best_i == -1 or best_gain == 0:
                        break
                    sol.add(kept[best_i])
                    unc &= ~kept_masks[best_i]
                return sol

            greedy_sol = greedy_upper_bound()
            best_size = len(greedy_sol)
            best_solution = greedy_sol

            # Now B&B to try to improve
            recurse(0, full_mask, [])

        bb_solve()

        if best_solution is None:
            # Fallback: all vertices
            best_solution = set(range(n))

        return best_solution

    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 15:
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
        time_budget = min(deadline - 5, elapsed + (deadline - 5 - elapsed) * 0.9)
        # Simpler: give each instance a portion
        remaining_instances = 3 - results.__len__()  # approximate
        per_instance = (deadline - 5 - elapsed) / max(1, remaining_instances)
        per_instance = min(per_instance, deadline - 5 - elapsed)

        sol = solve_hitting_set(n, cliques, start_time + elapsed + per_instance)
        results.append(sorted(sol))

    return results
# EVOLVE-BLOCK-END
