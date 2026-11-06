import time
import math
import random as _random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

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

        # Precompute popcount for faster coverage calculations
        # Use a lookup table for 16-bit chunks
        popcount_table = [0] * 65536
        for i in range(65536):
            popcount_table[i] = bin(i).count('1')
        
        def popcount_fast(x):
            if x == 0:
                return 0
            count = 0
            while x:
                count += popcount_table[x & 0xFFFF]
                x >>= 16
            return count

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
                    cov = popcount_fast(unc & vertex_clique_mask[v])
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

        # Branch and Bound using bitmasks for uncovered
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

            # Lower Bound: max coverage of any single remaining vertex
            max_cov = 0
            for v in candidates:
                if v in selected:
                    continue
                cov = popcount_fast(uncovered_mask & vertex_clique_mask[v])
                if cov > max_cov:
                    max_cov = cov

            if max_cov == 0:
                return

            lb = len(selected) + math.ceil(popcount_fast(uncovered_mask) / max_cov)
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
            verts_in_target.sort(key=lambda v: -popcount_fast(uncovered_mask & vertex_clique_mask[v]))

            for v in verts_in_target:
                new_selected = selected | {v}
                new_uncovered = uncovered_mask & ~vertex_clique_mask[v]
                branch_and_bound(new_selected, new_uncovered)

        initial_uncovered = (1 << m) - 1
        branch_and_bound(set(), initial_uncovered)

        # Local search polish: try to remove vertices from the solution
        def local_search_polish(sol):
            nonlocal best_size, best_solution
            sol = set(sol)
            improved = True
            while improved:
                improved = False
                for v in list(sol):
                    if len(sol) - 1 >= best_size:
                        continue
                    # Check if removing v still covers all cliques
                    new_sol = sol - {v}
                    # Check coverage
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

        # Polish the best solution found
        if best_solution:
            polished = local_search_polish(best_solution)
            if len(polished) < best_size:
                best_size = len(polished)
                best_solution = set(polished)

        # 2-swap local search: try replacing one vertex with another
        def two_swap_search(sol):
            nonlocal best_size, best_solution
            sol = set(sol)
            improved = True
            while improved:
                improved = False
                if time.time() - start_time > DEADLINE:
                    break
                # Try removing each vertex in sol
                for v_remove in list(sol):
                    # Compute coverage without v_remove
                    covered_without = 0
                    for u in sol:
                        if u != v_remove:
                            covered_without |= vertex_clique_mask[u]
                    
                    # Find cliques not covered
                    missing = (1 << m) - 1 & ~covered_without
                    if missing == 0:
                        # Can remove v_remove entirely
                        sol.discard(v_remove)
                        improved = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_solution = set(sol)
                        break
                    
                    # Try replacing v_remove with a vertex not in sol
                    # that covers all missing cliques
                    for v_add in candidates:
                        if v_add in sol:
                            continue
                        # Check if v_add covers all missing cliques
                        if (missing & vertex_clique_mask[v_add]) == missing:
                            # Check if this swap reduces size (it doesn't, but might enable further reductions)
                            # Only do it if it helps with subsequent removals
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
            # Polish again after swaps
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
