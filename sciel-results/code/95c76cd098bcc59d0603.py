import time
import math
import random as _random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        cliques = []
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
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
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                bk(R, P & adj[v], X & adj[v])
                R.discard(v)
                P.discard(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return cliques

    def build_clique_membership(cliques, n):
        vert_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vert_cliques[v].add(ci)
        return vert_cliques

    def eliminate_dominated(vert_cliques, n):
        dominated_by = {}
        for v in range(n):
            if v in dominated_by:
                continue
            sv = vert_cliques[v]
            if not sv:
                continue
            for w in range(n):
                if v == w or w in dominated_by:
                    continue
                if sv < vert_cliques[w]:
                    dominated_by[v] = w
                    break
        return dominated_by

    def greedy_hit(cliques, n, vert_cliques, order_seed=None, dominated_by=None):
        m = len(cliques)
        if m == 0:
            return set()
        uncovered = set(range(m))
        result = set()
        if order_seed is not None:
            rng = _random.Random(order_seed)
            order = list(range(n))
            rng.shuffle(order)
        else:
            order = sorted(range(n), key=lambda v: -len(vert_cliques[v]))
        if dominated_by:
            order = [v for v in order if v not in dominated_by]
        while uncovered:
            if time.time() - start_time > DEADLINE:
                break
            best_v = -1
            best_covered = 0
            for v in order:
                if v in result:
                    continue
                covered = len(uncovered & vert_cliques[v])
                if covered > best_covered:
                    best_covered = covered
                    best_v = v
            if best_v == -1 or best_covered == 0:
                break
            result.add(best_v)
            uncovered -= vert_cliques[best_v]
        return result

    def local_search_reduce(result, cliques, vert_cliques):
        result = set(result)
        m = len(cliques)
        cover_count = [0] * m
        for v in result:
            for ci in vert_cliques[v]:
                cover_count[ci] += 1
        improved = True
        while improved:
            improved = False
            for v in list(result):
                can_remove = True
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for ci in vert_cliques[v]:
                        cover_count[ci] -= 1
                    result.discard(v)
                    improved = True
        return result

    def local_search_swap(result, cliques, vert_cliques, n):
        result = set(result)
        m = len(cliques)
        cover_count = [0] * m
        for v in result:
            for ci in vert_cliques[v]:
                cover_count[ci] += 1
        improved = True
        iters = 0
        while improved and iters < 200 and time.time() - start_time < DEADLINE:
            improved = False
            iters += 1
            for v in list(result):
                can_remove = True
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for ci in vert_cliques[v]:
                        cover_count[ci] -= 1
                    result.discard(v)
                    improved = True
                    break
            if improved:
                continue
            best_swap = None
            for v in result:
                unique_cliques = set()
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        unique_cliques.add(ci)
                if not unique_cliques:
                    continue
                for w in range(n):
                    if w in result:
                        continue
                    w_cliques = vert_cliques[w]
                    if all(ci in w_cliques for ci in unique_cliques):
                        if len(w_cliques) >= len(vert_cliques[v]):
                            best_swap = (v, w)
                            break
                if best_swap:
                    break
            if best_swap:
                v, w = best_swap
                for ci in vert_cliques[v]:
                    cover_count[ci] -= 1
                result.discard(v)
                for ci in vert_cliques[w]:
                    cover_count[ci] += 1
                result.add(w)
                improved = True
        return result

    def iterated_local_search(cliques, n, vert_cliques, dominated_by=None, max_iters=60):
        m = len(cliques)
        if m == 0:
            return set()
        best = None
        best_size = float('inf')
        for seed in [None, 42, 123, 456, 789, 1000, 2000, 3000]:
            if time.time() - start_time > DEADLINE:
                break
            init = greedy_hit(cliques, n, vert_cliques, order_seed=seed, dominated_by=dominated_by)
            sol = local_search_reduce(init, cliques, vert_cliques)
            sol = local_search_swap(sol, cliques, vert_cliques, n)
            if len(sol) < best_size:
                best_size = len(sol)
                best = set(sol)
        if best is not None:
            for it in range(max_iters):
                if time.time() - start_time > DEADLINE:
                    break
                perturbed = set(best)
                if len(perturbed) <= 1:
                    break
                to_remove = list(perturbed)
                _random.shuffle(to_remove)
                perturbed.discard(to_remove[0])
                cover_count = [0] * m
                for v in perturbed:
                    for ci in vert_cliques[v]:
                        cover_count[ci] += 1
                uncovered = [ci for ci in range(m) if cover_count[ci] == 0]
                for ci in uncovered:
                    candidates = [v for v in cliques[ci] if v not in perturbed]
                    if candidates:
                        bv = max(candidates, key=lambda v: len(vert_cliques[v]))
                        perturbed.add(bv)
                        for c2 in vert_cliques[bv]:
                            cover_count[c2] += 1
                sol = local_search_reduce(perturbed, cliques, vert_cliques)
                sol = local_search_swap(sol, cliques, vert_cliques, n)
                if len(sol) < best_size:
                    best_size = len(sol)
                    best = set(sol)
        return best if best is not None else set()

    def packing_lb(clique_verts, uncovered_mask, m, n):
        """Combinatorial lower bound: greedy maximal set of pairwise disjoint uncovered cliques."""
        used = 0
        count = 0
        for ci in range(m):
            if not (uncovered_mask >> ci) & 1:
                continue
            mask = 0
            for v in clique_verts[ci]:
                mask |= (1 << v)
            if mask & used:
                continue
            used |= mask
            count += 1
        return count

    def branch_and_bound(cliques, n, vert_cliques, time_limit, initial_ub):
        m = len(cliques)
        if m == 0:
            return set()
        clique_verts = [list(c) for c in cliques]
        vert_mask = [0] * n
        for ci, verts in enumerate(clique_verts):
            for v in verts:
                vert_mask[v] |= (1 << ci)
        all_mask = (1 << m) - 1

        initial_sol, ub = initial_ub
        best_sol = set(initial_sol)
        best_size = ub
        best_mask = 0
        for v in best_sol:
            best_mask |= (1 << v)

        # Smooth surrogate lower bound: log-sum-exp soft-max of weighted clique packing.
        # For any weights w>=0 with sum_{cliques containing v} w_ci <= 1 (for all v),
        # sum_{ci in U} w_ci is a valid LP-dual lower bound on the hitting number of U.
        # We build w by greedy weighted packing (each clique's weight limited by the
        # remaining capacity of its vertices) and evaluate the soft-max (>= sum) over U.
        vert_capacity = [1.0] * n
        w = [0.0] * m
        for ci in range(m):
            if vert_capacity is None:
                break
            min_cap = 1.0
            for v in clique_verts[ci]:
                if vert_capacity[v] < min_cap:
                    min_cap = vert_capacity[v]
            if min_cap <= 0:
                continue
            w[ci] = min_cap
            for v in clique_verts[ci]:
                vert_capacity[v] -= min_cap

        wsum_all = sum(w)
        logsumexp_all = None
        if wsum_all > 0:
            mx = max(w)
            s = 0.0
            for x in w:
                s += math.exp(x - mx)
            logsumexp_all = mx + math.log(s)

        def dual_lb(uncovered_mask):
            # combinatorial packing
            lb = packing_lb(clique_verts, uncovered_mask, m, n)
            # smooth surrogate (log-sum-exp) over uncovered clique weights
            if logsumexp_all is not None and wsum_all > 0:
                if uncovered_mask == all_mask:
                    soft = logsumexp_all
                else:
                    mx = 0.0
                    s = 0.0
                    anyw = False
                    for ci in range(m):
                        if (uncovered_mask >> ci) & 1:
                            x = w[ci]
                            if x > mx:
                                mx = x
                            s += math.exp(x - mx)
                            anyw = True
                    if anyw:
                        soft = mx + math.log(s)
                    else:
                        soft = 0.0
                if soft > lb:
                    lb = int(math.floor(soft - 1e-9)) + 1
            return lb

        def bb(selected_mask, uncovered_mask, depth):
            nonlocal best_sol, best_size, best_mask
            if time.time() - start_time > time_limit:
                return
            if uncovered_mask == 0:
                sz = bin(selected_mask).count('1')
                if sz < best_size:
                    best_size = sz
                    best_mask = selected_mask
                return
            lb = dual_lb(uncovered_mask)
            if bin(selected_mask).count('1') + lb >= best_size:
                return
            # pick smallest uncovered clique to branch on
            min_ci = -1
            min_len = 10**9
            for ci in range(m):
                if (uncovered_mask >> ci) & 1:
                    L = len(clique_verts[ci])
                    if L < min_len:
                        min_len = L
                        min_ci = ci
                        if L <= 2:
                            break
            if min_ci < 0:
                return
            cverts = clique_verts[min_ci]
            cverts_sorted = sorted(cverts, key=lambda v: -((vert_mask[v] & uncovered_mask).bit_count()))
            for v in cverts_sorted:
                if time.time() - start_time > time_limit:
                    return
                nm = vert_mask[v]
                bb(selected_mask | (1 << v), uncovered_mask & ~nm, depth + 1)

        bb(0, all_mask, 0)

        if best_mask != 0 or best_size < ub:
            sol = set()
            for v in range(n):
                if (best_mask >> v) & 1:
                    sol.add(v)
            return sol
        return best_sol

    def final_polish(sol, cliques, n, vert_cliques):
        m = len(cliques)
        result = set(sol)
        cover_count = [0] * m
        for v in result:
            for ci in vert_cliques[v]:
                cover_count[ci] += 1
        improved = True
        iters = 0
        while improved and iters < 200 and time.time() - start_time < DEADLINE:
            improved = False
            iters += 1
            # remove one
            for v in list(result):
                ok = True
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        ok = False
                        break
                if ok:
                    for ci in vert_cliques[v]:
                        cover_count[ci] -= 1
                    result.discard(v)
                    improved = True
                    break
            if improved:
                continue
            # swap one
            for v in result:
                unique = set()
                for ci in vert_cliques[v]:
                    if cover_count[ci] <= 1:
                        unique.add(ci)
                if not unique:
                    continue
                for w in range(n):
                    if w in result:
                        continue
                    wc = vert_cliques[w]
                    if all(ci in wc for ci in unique):
                        for ci in vert_cliques[v]:
                            cover_count[ci] -= 1
                        result.discard(v)
                        for ci in wc:
                            cover_count[ci] += 1
                        result.add(w)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # remove two, add one (net -1)
            if len(result) <= 25:
                rl = list(result)
                done = False
                for i in range(len(rl)):
                    for j in range(i + 1, len(rl)):
                        v1, v2 = rl[i], rl[j]
                        tc = cover_count[:]
                        for ci in vert_cliques[v1]:
                            tc[ci] -= 1
                        for ci in vert_cliques[v2]:
                            tc[ci] -= 1
                        unc = [ci for ci in range(m) if tc[ci] == 0]
                        if not unc:
                            for ci in vert_cliques[v1]:
                                cover_count[ci] -= 1
                            for ci in vert_cliques[v2]:
                                cover_count[ci] -= 1
                            result.discard(v1)
                            result.discard(v2)
                            improved = True
                            done = True
                            break
                        if unc:
                            uncs = set(unc)
                            for w in range(n):
                                if w in result:
                                    continue
                                wc = vert_cliques[w]
                                if all(ci in wc for ci in uncs):
                                    for ci in vert_cliques[v1]:
                                        cover_count[ci] -= 1
                                    for ci in vert_cliques[v2]:
                                        cover_count[ci] -= 1
                                    result.discard(v1)
                                    result.discard(v2)
                                    for ci in wc:
                                        cover_count[ci] += 1
                                    result.add(w)
                                    improved = True
                                    done = True
                                    break
                                if done:
                                    break
                    if done:
                        break
        return result

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        cliques = bron_kerbosch(adj, n)
        if not cliques:
            return []
        vert_cliques = build_clique_membership(cliques, n)
        dominated_by = eliminate_dominated(vert_cliques, n)

        heuristic_sol = iterated_local_search(cliques, n, vert_cliques, dominated_by=dominated_by)
        heuristic_sol = local_search_reduce(heuristic_sol, cliques, vert_cliques)
        heuristic_sol = local_search_swap(heuristic_sol, cliques, vert_cliques, n)

        remaining = DEADLINE - (time.time() - start_time)
        bb_time_limit = min(remaining * 0.7, 70.0)
        initial_ub = (set(heuristic_sol), len(heuristic_sol))
        bb_sol = branch_and_bound(cliques, n, vert_cliques,
                                  time_limit=start_time + bb_time_limit,
                                  initial_ub=initial_ub)

        best = bb_sol if len(bb_sol) < len(heuristic_sol) else heuristic_sol
        best = final_polish(best, cliques, n, vert_cliques)
        best = local_search_reduce(best, cliques, vert_cliques)
        best = local_search_swap(best, cliques, vert_cliques, n)
        return sorted(best)

    results = []
    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        remaining_instances = 3 - idx
        time_per_instance = (DEADLINE - (time.time() - start_time)) / remaining_instances
        if time_per_instance < 8:
            results.append([])
            continue
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
