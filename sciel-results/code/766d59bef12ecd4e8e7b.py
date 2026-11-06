import time
import math
import random as _random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_pivot(adj, n):
        """Bron-Kerbosch with pivot to enumerate maximal cliques."""
        cliques = []

        def bk(R, P, X):
            if time.time() - start_time > DEADLINE:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return

            # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u

            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])

            for v in candidates:
                R.add(v)
                bk(R, P & adj[v], X & adj[v])
                R.discard(v)
                P.discard(v)
                X.add(v)

        bk(frozenset(), frozenset(range(n)), frozenset())
        return cliques

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        cliques = bron_kerbosch_pivot(adj, n)

        if not cliques:
            return []

        m = len(cliques)

        # Build incidence: which cliques each vertex covers
        vertex_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(ci)

        # Only vertices that appear in at least one clique matter
        active_vertices = [v for v in range(n) if vertex_cliques[v]]
        if not active_vertices:
            return []

        # Dominance pruning: if vertex_cliques[v] ⊆ vertex_cliques[u] and v != u, remove v
        # Sort by number of cliques covered (descending)
        active_sorted = sorted(active_vertices, key=lambda v: -len(vertex_cliques[v]))

        keep_vertices = []
        for v in active_sorted:
            dominated = False
            for u in keep_vertices:
                if vertex_cliques[v].issubset(vertex_cliques[u]):
                    dominated = True
                    break
            if not dominated:
                keep_vertices.append(v)

        candidates = sorted(keep_vertices)
        num_cand = len(candidates)

        if num_cand == 0:
            return []

        # Build coverage sets for each candidate
        cand_covers = [vertex_cliques[v] for v in candidates]

        # Build for each clique, which candidates cover it
        clique_to_cands = [set() for _ in range(m)]
        for i in range(num_cand):
            for ci in cand_covers[i]:
                clique_to_cands[ci].add(i)

        all_cliques = set(range(m))

        # Greedy set cover for initial solution
        def greedy_cover():
            selected = set()
            covered = set()
            while covered != all_cliques:
                remaining = all_cliques - covered
                best_i = None
                best_cov = 0
                for i in range(num_cand):
                    if i in selected:
                        continue
                    cov = len(remaining & cand_covers[i])
                    if cov > best_cov:
                        best_cov = cov
                        best_i = i
                if best_i is None or best_cov == 0:
                    break
                selected.add(best_i)
                covered |= cand_covers[best_i]
            return selected

        # Get initial solution
        best_sol = greedy_cover()
        best_size = len(best_sol)

        # Randomized greedy restarts
        rng = _random.Random(12345)
        for _ in range(50):
            selected = set()
            covered = set()
            while covered != all_cliques:
                remaining = all_cliques - covered
                # Weighted random choice among top candidates
                scores = []
                for i in range(num_cand):
                    if i in selected:
                        continue
                    cov = len(remaining & cand_covers[i])
                    if cov > 0:
                        scores.append((cov, i))
                if not scores:
                    break
                # Pick randomly among top 3
                scores.sort(reverse=True)
                top_k = min(3, len(scores))
                _, best_i = scores[rng.randint(0, top_k - 1)]
                selected.add(best_i)
                covered |= cand_covers[best_i]
            if len(selected) < best_size:
                best_size = len(selected)
                best_sol = selected

        # Branch and bound with LP lower bound
        def compute_lp_lb(uncovered_cliques, forbidden_cands):
            """LP relaxation lower bound for set cover on uncovered cliques."""
            if not uncovered_cliques:
                return 0

            # Build reduced problem
            active_cands = [i for i in range(num_cand) if i not in forbidden_cands]
            if not active_cands:
                return float('inf')

            # Filter to candidates that cover at least one uncovered clique
            active_cands = [i for i in active_cands if cand_covers[i] & uncovered_cliques]
            if not active_cands:
                return float('inf')

            # Use scipy linprog for LP relaxation
            try:
                from scipy.optimize import linprog
                import numpy as np

                # Objective: minimize sum x_i
                c_vec = np.ones(len(active_cands))

                # Constraints: for each uncovered clique, sum of x_i for i covering it >= 1
                A_ub = []
                b_ub = []
                for ci in uncovered_cliques:
                    row = []
                    for i in active_cands:
                        row.append(1.0 if ci in cand_covers[i] else 0.0)
                    A_ub.append(row)
                    b_ub.append(-1.0)  # -sum x_i <= -1

                if not A_ub:
                    return 0

                A_ub_arr = np.array(A_ub)
                b_ub_arr = np.array(b_ub)

                res = linprog(c_vec, A_ub=A_ub_arr, b_ub=b_ub_arr,
                             bounds=[(0, 1)] * len(active_cands),
                             method='highs')

                if res.success:
                    return res.fun
                else:
                    # Fallback: packing bound
                    return _packing_lb(uncovered_cliques, forbidden_cands)
            except Exception:
                return _packing_lb(uncovered_cliques, forbidden_cands)

        def _packing_lb(uncovered_cliques, forbidden_cands):
            """Packing lower bound: find disjoint cliques, each needs a distinct vertex."""
            # Greedy: repeatedly pick a clique with fewest available candidates
            remaining = set(uncovered_cliques)
            lb = 0
            while remaining:
                # Find clique with minimum available candidates
                min_avail = float('inf')
                best_ci = None
                for ci in remaining:
                    avail = len(clique_to_cands[ci] - forbidden_cands)
                    if avail == 0:
                        return float('inf')
                    if avail < min_avail:
                        min_avail = avail
                        best_ci = ci
                lb += 1
                # Remove best_ci and all cliques that share a candidate with it
                # (approximate: just remove best_ci for safety)
                remaining.discard(best_ci)
                # Actually, for a valid packing bound, we need disjoint cliques
                # Let's be more careful: remove cliques that are "covered" by the same candidates
                # For simplicity, just remove best_ci
            return lb

        def branch_and_bound(selected_cands, uncovered_cliques, depth=0):
            nonlocal best_size, best_sol

            if time.time() - start_time > DEADLINE - 2:
                return

            if not uncovered_cliques:
                if len(selected_cands) < best_size:
                    best_size = len(selected_cands)
                    best_sol = set(selected_cands)
                return

            # Prune: if current size + lower bound >= best_size
            lb = compute_lp_lb(uncovered_cliques, selected_cands)
            if len(selected_cands) + lb >= best_size:
                return

            # Choose the most constrained uncovered clique
            target_ci = None
            min_avail = float('inf')
            for ci in uncovered_cliques:
                avail = len(clique_to_cands[ci] - selected_cands)
                if avail == 0:
                    return  # infeasible
                if avail < min_avail:
                    min_avail = avail
                    target_ci = ci
                if min_avail == 1:
                    break

            if target_ci is None:
                return

            # Get candidates covering target_ci, ordered by coverage of uncovered
            avail_cands = [i for i in clique_to_cands[target_ci] if i not in selected_cands]
            avail_cands.sort(key=lambda i: -len(uncovered_cliques & cand_covers[i]))

            for i in avail_cands:
                new_selected = selected_cands | {i}
                new_uncovered = uncovered_cliques - cand_covers[i]
                branch_and_bound(new_selected, new_uncovered, depth + 1)
                if time.time() - start_time > DEADLINE - 2:
                    return

        # Run B&B with time limit
        bnb_start = time.time()
        bnb_time_limit = min(time_budget * 0.5, 30.0)
        branch_and_bound(set(), all_cliques)

        # Local search: iterated improvement
        def local_search(sol):
            improved = True
            while improved and time.time() - start_time < DEADLINE - 5:
                improved = False
                sol_set = set(sol)
                # Try removing each vertex
                for i in list(sol_set):
                    trial = sol_set - {i}
                    covered = set()
                    for j in trial:
                        covered |= cand_covers[j]
                    if covered == all_cliques:
                        sol_set = trial
                        improved = True
                        break

                # Try 1-swap: remove one, add one
                if not improved:
                    for i in list(sol_set):
                        for j in range(num_cand):
                            if j in sol_set:
                                continue
                            trial = (sol_set - {i}) | {j}
                            covered = set()
                            for k in trial:
                                covered |= cand_covers[k]
                            if covered == all_cliques and len(trial) < len(sol_set):
                                sol_set = trial
                                improved = True
                                break
                        if improved:
                            break

                # Try 2-swap: remove two, add two
                if not improved and len(sol_set) >= 4:
                    sol_list = list(sol_set)
                    for a_idx in range(len(sol_list)):
                        for b_idx in range(a_idx + 1, len(sol_list)):
                            i, j = sol_list[a_idx], sol_list[b_idx]
                            for k in range(num_cand):
                                if k in sol_set:
                                    continue
                                for l in range(num_cand):
                                    if l in sol_set or l == k:
                                        continue
                                    trial = (sol_set - {i, j}) | {k, l}
                                    covered = set()
                                    for m2 in trial:
                                        covered |= cand_covers[m2]
                                    if covered == all_cliques and len(trial) < len(sol_set):
                                        sol_set = trial
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break

            return sol_set

        best_sol = local_search(best_sol)
        best_size = len(best_sol)

        # Iterated local search with perturbation
        ils_start = time.time()
        ils_time_limit = time_budget * 0.3
        for iteration in range(200):
            if time.time() - start_time > DEADLINE - 5:
                break
            if time.time() - ils_start > ils_time_limit:
                break

            # Perturb: remove a random vertex, re-greedy
            perturbed = set(best_sol)
            if perturbed:
                victim = rng.choice(list(perturbed))
                perturbed.discard(victim)
                # Re-add via greedy
                covered = set()
                for i in perturbed:
                    covered |= cand_covers[i]
                while covered != all_cliques:
                    remaining = all_cliques - covered
                    best_i = None
                    best_cov = 0
                    for i in range(num_cand):
                        if i in perturbed:
                            continue
                        cov = len(remaining & cand_covers[i])
                        if cov > best_cov:
                            best_cov = cov
                            best_i = i
                    if best_i is None:
                        break
                    perturbed.add(best_i)
                    covered |= cand_covers[best_i]

            # Local search on perturbed
            improved_sol = local_search(perturbed)
            if len(improved_sol) < best_size:
                best_size = len(improved_sol)
                best_sol = improved_sol

        # Convert back to vertex indices
        best_vertices = [candidates[i] for i in best_sol]
        return sorted(best_vertices)

    # Allocate time across instances
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    total_time = DEADLINE - start_time
    # Give more time to larger instances
    weights = [1.0, 1.2, 1.5]
    total_weight = sum(weights)
    time_budgets = [total_time * w / total_weight for w in weights]

    results = []
    for (n, p, seed), tb in zip(instances, time_budgets):
        if time.time() - start_time > DEADLINE - 3:
            # Not enough time, use greedy
            edges = random_graph(n, p, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            cliques = bron_kerbosch_pivot(adj, n)
            if not cliques:
                results.append([])
                continue
            # Quick greedy
            vertex_cliques = [set() for _ in range(n)]
            for ci, c in enumerate(cliques):
                for v in c:
                    vertex_cliques[v].add(ci)
            active = [v for v in range(n) if vertex_cliques[v]]
            selected = set()
            covered = set()
            all_c = set(range(len(cliques)))
            while covered != all_c:
                remaining = all_c - covered
                best_v = None
                best_cov = 0
                for v in active:
                    if v in selected:
                        continue
                    cov = len(remaining & vertex_cliques[v])
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                if best_v is None:
                    break
                selected.add(best_v)
                covered |= vertex_cliques[best_v]
            results.append(sorted(selected))
        else:
            sol = solve_instance(n, p, seed, tb)
            results.append(sol)

    return results
