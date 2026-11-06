import time
import math
import random as _random
from collections import defaultdict

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
        Returns a list of lists (each a list of vertex indices).
        """
        cliques = []
        full_mask = (1 << n) - 1

        def popcount(x):
            return bin(x).count('1')

        def bk(R_mask, P_mask, X_mask):
            if P_mask == 0 and X_mask == 0:
                if popcount(R_mask) >= 2:
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
                cnt = popcount(inter)
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
        # vertex_clique_mask[v] = bitmask of cliques containing vertex v
        vertex_clique_mask = [0] * n
        for ci, c in enumerate(cliques):
            bit = 1 << ci
            for v in c:
                vertex_clique_mask[v] |= bit

        # Identify candidate vertices (those in at least one clique)
        candidates = [v for v in range(n) if vertex_clique_mask[v] != 0]

        if not candidates:
            return []

        # Dominance Pruning using bitmasks
        # If C_u ⊆ C_v (as bitmasks), then u is dominated by v
        active = sorted(candidates, key=lambda v: -bin(vertex_clique_mask[v]).count('1'))

        keep = set()
        for v in active:
            dominated = False
            for u in keep:
                # Check if vertex_clique_mask[v] is subset of vertex_clique_mask[u]
                if (vertex_clique_mask[v] & ~vertex_clique_mask[u]) == 0:
                    dominated = True
                    break
            if not dominated:
                keep.add(v)

        candidates = list(keep)

        if not candidates:
            return []

        # Precompute coverage counts for all candidate vertices
        cov_counts = [bin(vertex_clique_mask[v]).count('1') for v in candidates]

        # Greedy Set Cover using bitmasks
        def greedy_cover():
            selected = set()
            unc = (1 << m) - 1  # all cliques uncovered
            while unc:
                best_v = -1
                best_cov = 0
                for v in candidates:
                    if v in selected:
                        continue
                    cov = bin(unc & vertex_clique_mask[v]).count('1')
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

        # Precompute clique candidate lists
        clique_cands = []
        for ci, c in enumerate(cliques):
            cc = [v for v in c if v in keep]
            clique_cands.append(cc)

        # Precompute clique membership bitmasks for available candidates
        clique_avail_mask = [0] * m
        for ci, c in enumerate(cliques):
            mask = 0
            for v in c:
                if v in keep:
                    mask |= (1 << v)
            clique_avail_mask[ci] = mask

        # Branch and Bound using bitmasks for uncovered
        def branch_and_bound(selected_set, selected_mask, uncovered_mask, depth):
            nonlocal best_size, best_solution

            if uncovered_mask == 0:
                if len(selected_set) < best_size:
                    best_size = len(selected_set)
                    best_solution = set(selected_set)
                return

            if len(selected_set) >= best_size:
                return

            if time.time() - start_time > DEADLINE:
                return

            # Compute max coverage among unselected candidates
            max_cov = 0
            for v in candidates:
                if not (selected_mask >> v) & 1:
                    cov = bin(uncovered_mask & vertex_clique_mask[v]).count('1')
                    if cov > max_cov:
                        max_cov = cov

            if max_cov == 0:
                return

            # Improved lower bound using greedy packing
            lb_extra = 0
            temp_unc = uncovered_mask
            while temp_unc:
                best_v = -1
                best_c = 0
                for v in candidates:
                    if not (selected_mask >> v) & 1:
                        c = bin(temp_unc & vertex_clique_mask[v]).count('1')
                        if c > best_c:
                            best_c = c
                            best_v = v
                if best_v == -1 or best_c == 0:
                    break
                lb_extra += 1
                temp_unc &= ~vertex_clique_mask[best_v]
                if len(selected_set) + lb_extra >= best_size:
                    break

            if len(selected_set) + lb_extra >= best_size:
                return

            # Also use the simpler bound
            simple_lb = len(selected_set) + math.ceil(bin(uncovered_mask).count('1') / max_cov)
            if simple_lb >= best_size:
                return

            # Branching heuristic: choose uncovered clique with fewest available candidates
            target_ci = -1
            min_avail = float('inf')
            temp = uncovered_mask
            while temp:
                lsb = temp & (-temp)
                ci = lsb.bit_length() - 1
                avail = bin(clique_avail_mask[ci] & ~selected_mask).count('1')
                if avail < min_avail:
                    min_avail = avail
                    target_ci = ci
                    if avail == 1:
                        break
                temp ^= lsb

            if target_ci == -1:
                return

            # Branch on vertices in this specific clique
            avail_mask = clique_avail_mask[target_ci] & ~selected_mask
            verts_in_target = []
            temp = avail_mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                verts_in_target.append(v)
                temp ^= lsb

            # Sort by coverage descending
            verts_in_target.sort(key=lambda v: -bin(uncovered_mask & vertex_clique_mask[v]).count('1'))

            for v in verts_in_target:
                new_selected = selected_set | {v}
                new_selected_mask = selected_mask | (1 << v)
                new_uncovered = uncovered_mask & ~vertex_clique_mask[v]
                branch_and_bound(new_selected, new_selected_mask, new_uncovered, depth + 1)

        initial_uncovered = (1 << m) - 1
        branch_and_bound(set(), 0, initial_uncovered, 0)

        # Local search polish: try to remove vertices, then 2-swap
        def local_search_polish(sol):
            nonlocal best_size, best_solution
            sol = set(sol)
            full = (1 << m) - 1

            # Phase 1: Remove redundant vertices
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
                    if covered == full:
                        sol = new_sol
                        improved = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_solution = set(sol)
                        break

            # Phase 2: 2-swap - replace one vertex with another to enable removal
            improved = True
            while improved:
                improved = False
                for v in list(sol):
                    # Coverage lost if we remove v
                    lost = 0
                    for u in sol:
                        if u != v:
                            lost |= vertex_clique_mask[u]
                    lost = full & ~lost

                    if lost == 0:
                        # Already handled in phase 1
                        continue

                    # Try adding a vertex w not in sol that covers lost
                    for w in candidates:
                        if w in sol:
                            continue
                        if (vertex_clique_mask[w] & lost) == lost:
                            # Adding w covers all lost
                            new_sol = (sol - {v}) | {w}
                            # Check if we can now remove w or v
                            # Actually we replaced v with w, so size is same
                            # But maybe we can remove more
                            if len(new_sol) < len(sol):
                                sol = new_sol
                                improved = True
                                if len(sol) < best_size:
                                    best_size = len(sol)
                                    best_solution = set(sol)
                                break
                if improved:
                    # Re-run phase 1
                    improved2 = True
                    while improved2:
                        improved2 = False
                        for v in list(sol):
                            if len(sol) - 1 >= best_size:
                                continue
                            new_sol = sol - {v}
                            covered = 0
                            for u in new_sol:
                                covered |= vertex_clique_mask[u]
                            if covered == full:
                                sol = new_sol
                                improved2 = True
                                if len(sol) < best_size:
                                    best_size = len(sol)
                                    best_solution = set(sol)
                                break

            return sol

        # Polish the best solution found
        if best_solution:
            polished = local_search_polish(best_solution)
            if len(polished) < best_size:
                best_size = len(polished)
                best_solution = set(polished)

        # Multiple restarts with different greedy initializations
        if time.time() - start_time < DEADLINE - 10:
            for restart in range(5):
                # Perturb the greedy choice
                def greedy_cover_perturbed(perturb=0):
                    selected = set()
                    unc = (1 << m) - 1
                    while unc:
                        best_v = -1
                        best_cov = 0
                        best_covs = []
                        for v in candidates:
                            if v in selected:
                                continue
                            cov = bin(unc & vertex_clique_mask[v]).count('1')
                            best_covs.append((cov, v))
                        best_covs.sort(reverse=True)
                        # Pick from top few with perturbation
                        top_k = min(4, len(best_covs))
                        if perturb > 0 and top_k > 1:
                            idx = perturb % top_k
                        else:
                            idx = 0
                        best_cov, best_v = best_covs[idx]
                        if best_cov == 0:
                            break
                        selected.add(best_v)
                        unc &= ~vertex_clique_mask[best_v]
                    return selected

                init_sol = greedy_cover_perturbed(restart)
                if len(init_sol) < best_size:
                    best_size = len(init_sol)
                    best_solution = set(init_sol)

                # Quick B&B from this start
                branch_and_bound(set(init_sol), 0, (1 << m) - 1, 0)

                if best_solution:
                    polished = local_search_polish(best_solution)
                    if len(polished) < best_size:
                        best_size = len(polished)
                        best_solution = set(polished)

                if time.time() - start_time > DEADLINE - 5:
                    break

        if best_solution is None:
            return sorted(init_sol)

        # Final verification: ensure all cliques are covered
        final_mask = 0
        for v in best_solution:
            final_mask |= vertex_clique_mask[v]
        if final_mask != (1 << m) - 1:
            # Fallback to greedy
            return sorted(init_sol)

        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
