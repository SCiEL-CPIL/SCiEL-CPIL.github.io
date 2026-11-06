import time
import math
import random as _random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the verifier."""
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj_bitmask(edges, n):
        """Build adjacency as integer bitmasks."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bron_kerbosch_bitmask(adj, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting,
        using integer bitmasks for speed.
        """
        cliques = []
        full_mask = (1 << n) - 1

        def popcount(x):
            return x.bit_count()

        def bk(R_mask, P_mask, X_mask):
            if P_mask == 0 and X_mask == 0:
                if R_mask.bit_count() >= 2:
                    # Convert bitmask to list of vertex indices
                    verts = [i for i in range(n) if (R_mask >> i) & 1]
                    cliques.append(verts)
                return

            P_union_X = P_mask | X_mask
            if P_union_X == 0:
                return

            # Choose pivot u in P ∪ X to maximize |P ∩ N(u)|
            max_inter = -1
            pivot = -1
            temp = P_union_X
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                inter = P_mask & adj[u]
                cnt = inter.bit_count()
                if cnt > max_inter:
                    max_inter = cnt
                    pivot = u
                temp ^= lsb

            candidates = P_mask & ~adj[pivot] if pivot >= 0 else P_mask

            if candidates == 0:
                return

            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1

                new_R = R_mask | lsb
                new_P = P_mask & adj[v]
                new_X = X_mask & adj[v]

                bk(new_R, new_P, new_X)

                P_mask &= ~lsb
                X_mask |= lsb
                temp ^= lsb

        bk(0, full_mask, 0)
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        adj = build_adj_bitmask(edges, n)

        # Enumerate maximal cliques
        cliques = bron_kerbosch_bitmask(adj, n)

        if not cliques:
            return []

        m = len(cliques)

        # Build vertex -> clique coverage bitmask
        vertex_clique_mask = [0] * n
        for ci, c in enumerate(cliques):
            bit = 1 << ci
            for v in c:
                vertex_clique_mask[v] |= bit

        # Identify candidate vertices
        candidates = [v for v in range(n) if vertex_clique_mask[v] != 0]
        if not candidates:
            return []

        # Dominance Pruning
        active = sorted(candidates, key=lambda v: -vertex_clique_mask[v].bit_count())
        keep = set()
        for v in active:
            dominated = False
            for u in keep:
                if (vertex_clique_mask[v] & ~vertex_clique_mask[u]) == 0:
                    dominated = True
                    break
            if not dominated:
                keep.add(v)
        candidates = list(keep)
        if not candidates:
            return []

        # Precompute clique intersection masks for lower bound
        # clique_inter_mask[ci] = bitmask of cliques that share at least one vertex with clique[ci]
        clique_inter_mask = [0] * m
        for ci, c in enumerate(cliques):
            mask = 0
            for v in c:
                mask |= vertex_clique_mask[v]
            clique_inter_mask[ci] = mask

        # Precompute clique sizes
        clique_sizes = [len(c) for c in cliques]

        # Greedy Set Cover
        def greedy_cover():
            selected = set()
            unc = (1 << m) - 1
            while unc:
                best_v = -1
                best_cov = 0
                for v in candidates:
                    if v in selected:
                        continue
                    cov = (unc & vertex_clique_mask[v]).bit_count()
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                if best_v == -1:
                    break
                selected.add(best_v)
                unc &= ~vertex_clique_mask[best_v]
            return selected

        init_sol = greedy_cover()
        best_size = len(init_sol)
        best_solution = set(init_sol)

        # Multiple greedy restarts
        np.random.seed(42)
        for restart in range(10):
            if time.time() - start_time > DEADLINE:
                break
            rng = np.random.RandomState(42 + restart)
            perm = rng.permutation(len(candidates))
            order = [candidates[i] for i in perm]
            
            selected = set()
            unc = (1 << m) - 1
            for v in order:
                if unc == 0:
                    break
                if v in selected:
                    continue
                cov = (unc & vertex_clique_mask[v]).bit_count()
                if cov > 0:
                    selected.add(v)
                    unc &= ~vertex_clique_mask[v]
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = set(selected)

        # Precompute candidate list per clique for branching
        clique_cands = []
        for ci, c in enumerate(cliques):
            cc = [v for v in c if v in keep]
            clique_cands.append(cc)

        # Lower Bound: Disjoint Clique Packing
        def packing_bound(unc_mask):
            count = 0
            temp = unc_mask
            while temp:
                # Find clique in temp with minimum size
                min_size = float('inf')
                target_ci = -1
                t = temp
                while t:
                    lsb = t & (-t)
                    ci = lsb.bit_length() - 1
                    if clique_sizes[ci] < min_size:
                        min_size = clique_sizes[ci]
                        target_ci = ci
                    t ^= lsb
                
                if target_ci == -1:
                    break
                
                count += 1
                # Remove this clique and all intersecting cliques
                temp &= ~clique_inter_mask[target_ci]
            return count

        # Branch and Bound
        def branch_and_bound(selected, uncovered_mask):
            nonlocal best_size, best_solution

            if uncovered_mask == 0:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                return

            if len(selected) >= best_size:
                return

            if time.time() - start_time > DEADLINE:
                return

            # Lower Bound 1: Greedy Packing
            lb_pack = packing_bound(uncovered_mask)
            
            # Lower Bound 2: Max Coverage
            max_cov = 0
            for v in candidates:
                if v in selected:
                    continue
                cov = (uncovered_mask & vertex_clique_mask[v]).bit_count()
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return

            lb_cov = math.ceil(uncovered_mask.bit_count() / max_cov)
            
            lb = len(selected) + max(lb_pack, lb_cov)
            if lb >= best_size:
                return

            # Branching heuristic: choose uncovered clique with fewest available candidates
            target_ci = -1
            min_avail = float('inf')
            temp = uncovered_mask
            while temp:
                lsb = temp & (-temp)
                ci = lsb.bit_length() - 1
                avail = 0
                for v in clique_cands[ci]:
                    if v not in selected:
                        avail += 1
                if avail < min_avail:
                    min_avail = avail
                    target_ci = ci
                    if avail == 1:
                        break
                temp ^= lsb

            if target_ci == -1:
                return

            # Branch on vertices in this specific clique
            verts_in_target = [v for v in clique_cands[target_ci] if v not in selected]
            # Sort by coverage descending
            verts_in_target.sort(key=lambda v: -(uncovered_mask & vertex_clique_mask[v]).bit_count())

            for v in verts_in_target:
                new_selected = selected | {v}
                new_uncovered = uncovered_mask & ~vertex_clique_mask[v]
                branch_and_bound(new_selected, new_uncovered)

        initial_uncovered = (1 << m) - 1
        branch_and_bound(set(), initial_uncovered)

        # Local Search Polish
        def local_search_polish(sol):
            nonlocal best_size, best_solution
            sol = set(sol)
            improved = True
            while improved:
                improved = False
                for v in list(sol):
                    if len(sol) - 1 >= best_size:
                        continue
                    new_sol = sol - {v}
                    covered = 0
                    for u in new_sol:
                        covered |= vertex_clique_mask[u]
                    if covered == (1 << m) - 1:
                        sol = new_sol
                        improved = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_solution = set(sol)
                        break
            return sol

        if best_solution:
            polished = local_search_polish(best_solution)
            if len(polished) < best_size:
                best_size = len(polished)
                best_solution = set(polished)

        # 2-swap local search
        def two_swap_search(sol):
            nonlocal best_size, best_solution
            sol = set(sol)
            improved = True
            while improved:
                improved = False
                if time.time() - start_time > DEADLINE:
                    break
                for v_remove in list(sol):
                    covered_without = 0
                    for u in sol:
                        if u != v_remove:
                            covered_without |= vertex_clique_mask[u]
                    
                    missing = (1 << m) - 1 & ~covered_without
                    if missing == 0:
                        sol.discard(v_remove)
                        improved = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_solution = set(sol)
                        break
                    
                    for v_add in candidates:
                        if v_add in sol:
                            continue
                        if (missing & vertex_clique_mask[v_add]) == missing:
                            new_sol = sol - {v_remove} | {v_add}
                            # Check if we can now remove some vertex
                            for v_check in new_sol:
                                covered_check = 0
                                for u in new_sol:
                                    if u != v_check:
                                        covered_check |= vertex_clique_mask[u]
                                if covered_check == (1 << m) - 1:
                                    sol = new_sol - {v_check}
                                    improved = True
                                    if len(sol) < best_size:
                                        best_size = len(sol)
                                        best_solution = set(sol)
                                    break
                            if improved:
                                break
                    if improved:
                        break
            return sol

        if best_solution:
            swapped = two_swap_search(best_solution)
            if len(swapped) < best_size:
                best_size = len(swapped)
                best_solution = set(swapped)
            polished = local_search_polish(best_solution)
            if len(polished) < best_size:
                best_size = len(polished)
                best_solution = set(polished)

        if best_solution is None:
            return sorted(init_sol)

        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
