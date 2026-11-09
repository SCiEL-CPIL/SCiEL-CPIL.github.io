import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=np.bool_)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u, v] = True
                    adj[v, u] = True
        return adj

    def bron_kerbosch(adj, n):
        cliques = []
        neighbors = [0] * n
        for u in range(n):
            for v in range(n):
                if u != v and adj[u, v]:
                    neighbors[u] |= (1 << v)

        def expand(R, P, X):
            if P == 0 and X == 0:
                if R:
                    cliques.append(frozenset(i for i in range(n) if (R >> i) & 1))
                return
            union = P | X
            if union:
                max_count = -1
                pivot = -1
                u = 0
                while u < n:
                    if (union >> u) & 1:
                        count = bin(P & neighbors[u]).count('1')
                        if count > max_count:
                            max_count = count
                            pivot = u
                    u += 1
                candidates = P & ~neighbors[pivot] if pivot >= 0 else P
            else:
                candidates = P

            v = 0
            while v < n:
                if (candidates >> v) & 1:
                    bit = 1 << v
                    new_R = R | bit
                    new_P = P & neighbors[v]
                    new_X = X & neighbors[v]
                    expand(new_R, new_P, new_X)
                    P &= ~bit
                    X |= bit
                v += 1

        expand(0, (1 << n) - 1, 0)
        return cliques

    def is_valid(sol, cliques):
        sol_set = set(sol)
        for c in cliques:
            if not sol_set & c:
                return False
        return True

    def greedy_hitting_set(cliques, n):
        vertex_counts = [0] * n
        for c in cliques:
            for v in c:
                vertex_counts[v] += 1

        vertices = sorted(range(n), key=lambda v: (-vertex_counts[v], v))

        uncovered = [True] * len(cliques)
        num_uncovered = len(cliques)
        selected = []

        while num_uncovered > 0:
            best_v = -1
            best_hits = 0
            for v in vertices:
                if v in selected:
                    continue
                hits = 0
                for i, c in enumerate(cliques):
                    if uncovered[i] and v in c:
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_v = v

            if best_v == -1 or best_hits == 0:
                break

            selected.append(best_v)
            for i, c in enumerate(cliques):
                if uncovered[i] and best_v in c:
                    uncovered[i] = False
                    num_uncovered -= 1

        # Post-processing: remove unnecessary vertices
        sol_set = set(selected)
        for v in list(selected):
            test_set = sol_set - {v}
            ok = True
            for c in cliques:
                if not test_set & c:
                    ok = False
                    break
            if ok:
                selected.remove(v)
                sol_set = test_set

        return selected

    def greedy_init_with_perturbation(cliques, n, rng, perturb_prob=0.3):
        """Greedy hitting set with random perturbation for diversity."""
        vertex_counts = [0] * n
        for c in cliques:
            for v in c:
                vertex_counts[v] += 1

        # Add noise to counts for perturbation
        noisy_counts = []
        for v in range(n):
            noise = rng.gauss(0, 0.5 * max(1, vertex_counts[v]))
            noisy_counts.append(vertex_counts[v] + noise)

        vertices = sorted(range(n), key=lambda v: (-noisy_counts[v], v))

        uncovered = [True] * len(cliques)
        num_uncovered = len(cliques)
        selected = []

        while num_uncovered > 0:
            # Find all vertices that cover at least one uncovered clique
            best_v = -1
            best_hits = 0
            for v in vertices:
                if v in selected:
                    continue
                hits = 0
                for i, c in enumerate(cliques):
                    if uncovered[i] and v in c:
                        hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_v = v

            if best_v == -1 or best_hits == 0:
                break

            # With some probability, pick a random vertex that covers at least one uncovered clique
            if rng.random() < perturb_prob:
                candidates = [v for v in vertices if v not in selected]
                valid_candidates = []
                for v in candidates:
                    for i, c in enumerate(cliques):
                        if uncovered[i] and v in c:
                            valid_candidates.append(v)
                            break
                if valid_candidates:
                    best_v = rng.choice(valid_candidates)

            selected.append(best_v)
            for i, c in enumerate(cliques):
                if uncovered[i] and best_v in c:
                    uncovered[i] = False
                    num_uncovered -= 1

        # Post-processing: remove unnecessary vertices
        sol_set = set(selected)
        for v in list(selected):
            test_set = sol_set - {v}
            ok = True
            for c in cliques:
                if not test_set & c:
                    ok = False
                    break
            if ok:
                selected.remove(v)
                sol_set = test_set

        return selected

    def local_search(sol, cliques, n, max_iters=5000, time_budget=None, rng_seed=42):
        best_sol = list(sol)
        best_size = len(best_sol)
        cliques_list = list(cliques)

        num_cliques = len(cliques_list)
        if num_cliques == 0:
            return best_sol

        vertex_clique_mask = [0] * n
        for i, c in enumerate(cliques_list):
            for v in c:
                vertex_clique_mask[v] |= (1 << i)

        all_cliques_mask = (1 << num_cliques) - 1

        def covers_all(sol_set):
            mask = 0
            for v in sol_set:
                mask |= vertex_clique_mask[v]
            return mask == all_cliques_mask

        if not covers_all(best_sol):
            sol_set = set(best_sol)
            for c in cliques_list:
                if not sol_set & c:
                    best_sol.append(min(c))
            best_size = len(best_sol)

        rng = random.Random(rng_seed)
        ls_start = time.time()
        if time_budget is None:
            time_budget = 30
        ls_deadline = ls_start + time_budget

        def get_unique_mask(v, sol):
            """Get cliques covered only by v in sol."""
            other_mask = 0
            for u in sol:
                if u != v:
                    other_mask |= vertex_clique_mask[u]
            return vertex_clique_mask[v] & ~other_mask

        for it in range(max_iters):
            if time.time() > ls_deadline:
                break
            improved = False
            sol_set = set(best_sol)

            # Phase 1: Try removing a vertex (if it's redundant)
            for v in best_sol:
                new_mask = 0
                for u in best_sol:
                    if u != v:
                        new_mask |= vertex_clique_mask[u]
                if new_mask == all_cliques_mask:
                    best_sol.remove(v)
                    best_size -= 1
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: Try 1-swap (remove v, add w)
            found_swap = False
            for v in best_sol:
                v_unique = get_unique_mask(v, best_sol)
                if v_unique == 0:
                    continue

                # Find candidates w that cover all cliques in v_unique
                candidates = [w for w in range(n) if w not in sol_set and (vertex_clique_mask[w] & v_unique) == v_unique]
                for w in candidates:
                    new_mask = 0
                    for u in best_sol:
                        if u != v:
                            new_mask |= vertex_clique_mask[u]
                    new_mask |= vertex_clique_mask[w]
                    if new_mask == all_cliques_mask:
                        best_sol.remove(v)
                        best_sol.append(w)
                        found_swap = True
                        break
                if found_swap:
                    break

            if found_swap:
                # After swap, try to remove redundant vertices
                for v in list(best_sol):
                    new_mask = 0
                    for u in best_sol:
                        if u != v:
                            new_mask |= vertex_clique_mask[u]
                    if new_mask == all_cliques_mask:
                        best_sol.remove(v)
                        best_size -= 1
                        break
                continue

            # Phase 3: Try 2-swap (remove v1, v2; add w1) - replace 2 with 1
            found_2swap = False
            sol_list = list(best_sol)
            for i in range(len(sol_list)):
                v1 = sol_list[i]
                for j in range(i + 1, len(sol_list)):
                    v2 = sol_list[j]
                    # Cliques covered only by v1 or v2
                    other_mask = 0
                    for u in sol_list:
                        if u != v1 and u != v2:
                            other_mask |= vertex_clique_mask[u]
                    need_mask = (vertex_clique_mask[v1] | vertex_clique_mask[v2]) & ~other_mask

                    if need_mask == 0:
                        # Both v1 and v2 are redundant
                        best_sol.remove(v1)
                        best_sol.remove(v2)
                        best_size -= 2
                        improved = True
                        found_2swap = True
                        break

                    # Find a single vertex w not in sol that covers need_mask
                    candidates = [w for w in range(n) if w not in sol_set]
                    for w in candidates:
                        if (vertex_clique_mask[w] & need_mask) == need_mask:
                            best_sol.remove(v1)
                            best_sol.remove(v2)
                            best_sol.append(w)
                            best_size -= 1
                            found_2swap = True
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break

            if found_2swap:
                # After 2-swap, try to remove redundant vertices
                for v in list(best_sol):
                    new_mask = 0
                    for u in best_sol:
                        if u != v:
                            new_mask |= vertex_clique_mask[u]
                    if new_mask == all_cliques_mask:
                        best_sol.remove(v)
                        best_size -= 1
                        break
                continue

            # Phase 4: Try 2-swap (remove v1, v2; add w1, w2)
            found_2swap = False
            sol_list = list(best_sol)
            for i in range(len(sol_list)):
                v1 = sol_list[i]
                for j in range(i + 1, len(sol_list)):
                    v2 = sol_list[j]
                    other_mask = 0
                    for u in sol_list:
                        if u != v1 and u != v2:
                            other_mask |= vertex_clique_mask[u]
                    need_mask = (vertex_clique_mask[v1] | vertex_clique_mask[v2]) & ~other_mask

                    if need_mask == 0:
                        best_sol.remove(v1)
                        best_sol.remove(v2)
                        best_size -= 2
                        improved = True
                        found_2swap = True
                        break

                    # Find pairs (w1, w2) not in sol that cover need_mask
                    candidates = [w for w in range(n) if w not in sol_set]
                    for w1, w2 in combinations(candidates, 2):
                        if (vertex_clique_mask[w1] | vertex_clique_mask[w2]) & need_mask == need_mask:
                            best_sol.remove(v1)
                            best_sol.remove(v2)
                            best_sol.append(w1)
                            best_sol.append(w2)
                            found_2swap = True
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break

            if found_2swap:
                for v in list(best_sol):
                    new_mask = 0
                    for u in best_sol:
                        if u != v:
                            new_mask |= vertex_clique_mask[u]
                    if new_mask == all_cliques_mask:
                        best_sol.remove(v)
                        best_size -= 1
                        break
                continue

            # Phase 5: Random perturbation if stuck
            if rng.random() < 0.1:
                candidates = [v for v in range(n) if v not in best_sol]
                if candidates:
                    v = rng.choice(candidates)
                    new_sol = best_sol + [v]
                    if covers_all(new_sol):
                        best_sol = new_sol
                        best_size = len(best_sol)
                        for rv in list(best_sol):
                            new_mask = 0
                            for u in best_sol:
                                if u != rv:
                                    new_mask |= vertex_clique_mask[u]
                            if new_mask == all_cliques_mask:
                                best_sol.remove(rv)
                                best_size -= 1
                                break

        return best_sol

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        all_cliques = bron_kerbosch(adj, n)
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]

        if not relevant_cliques:
            return []

        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left < 2:
            return greedy_hitting_set(relevant_cliques, n)

        # Multiple restarts with perturbation
        best_sol = greedy_hitting_set(relevant_cliques, n)
        best_size = len(best_sol)

        # Try a few perturbed initializations
        rng_main = random.Random(seed)
        num_restarts = 3
        for r in range(num_restarts):
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            sol_r = greedy_init_with_perturbation(relevant_cliques, n, rng_main, perturb_prob=0.3)
            if len(sol_r) < best_size:
                best_sol = sol_r
                best_size = len(sol_r)

        # Local search with multiple passes
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 5:
            ls_iters = max(500, min(3000, int(time_left * 40)))
            sol2 = local_search(best_sol, relevant_cliques, n, max_iters=ls_iters, rng_seed=seed)
            if len(sol2) < best_size:
                best_sol = sol2
                best_size = len(sol2)

        # Second local search pass with different seed
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 5 and len(best_sol) > 0:
            ls_iters2 = max(300, min(2000, int(time_left * 30)))
            sol3 = local_search(best_sol, relevant_cliques, n, max_iters=ls_iters2, rng_seed=seed + 1)
            if len(sol3) < best_size:
                best_sol = sol3
                best_size = len(sol3)

        # Third local search pass
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 5 and len(best_sol) > 0:
            ls_iters3 = max(300, min(1500, int(time_left * 25)))
            sol4 = local_search(best_sol, relevant_cliques, n, max_iters=ls_iters3, rng_seed=seed + 2)
            if len(sol4) < best_size:
                best_sol = sol4
                best_size = len(sol4)

        # Try MILP if time permits
        elapsed = time.time() - start_time
        time_left = deadline - elapsed
        if time_left > 10:
            try:
                m = len(relevant_cliques)
                rows = []
                cols = []
                data = []
                for i, clique in enumerate(relevant_cliques):
                    for v in clique:
                        rows.append(i)
                        cols.append(v)
                        data.append(1)

                A = sp.csr_matrix((data, (rows, cols)), shape=(m, n))

                c_obj = np.ones(n, dtype=np.float64)
                constraints = LinearConstraint(A, 1, np.inf)
                bounds_obj = Bounds(0, 1)
                integrality = np.ones(n, dtype=np.int8)

                incumbent = np.zeros(n)
                for v in best_sol:
                    incumbent[v] = 1

                instance_time = min(time_left - 5, 20)
                res = milp(c=c_obj, constraints=constraints, integrality=integrality,
                           bounds=bounds_obj,
                           options={"time_limit": instance_time, "presolve": True,
                                    "MIP_start": incumbent})

                if res.x is not None:
                    milp_sol = [i for i in range(n) if res.x[i] > 0.5]
                    if is_valid(milp_sol, relevant_cliques):
                        if len(milp_sol) < best_size:
                            best_sol = milp_sol
                            best_size = len(milp_sol)
            except Exception:
                pass

        # Final aggressive reduction pass
        changed = True
        while changed:
            changed = False
            sol_set = set(best_sol)
            for v in list(best_sol):
                test_set = sol_set - {v}
                ok = True
                for c in relevant_cliques:
                    if not test_set & c:
                        ok = False
                        break
                if ok:
                    best_sol.remove(v)
                    sol_set = test_set
                    changed = True

        return best_sol

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            adj = random_graph(n, p, seed)
            all_cliques = bron_kerbosch(adj, n)
            relevant_cliques = [c for c in all_cliques if len(c) >= 2]
            sol = greedy_hitting_set(relevant_cliques, n) if relevant_cliques else []
            results.append(sorted(sol))
            continue

        sol = solve_instance(n, p, seed)
        results.append(sorted(sol))

    while len(results) < 3:
        results.append([])

    return results[:3]
