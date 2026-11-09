import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 120.0
    margin = 5.0

    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]

    results = []

    for n, p, seed in instances:
        # 1. Generate Graph
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)

        # 2. Enumerate Maximal Cliques using bitmask Bron-Kerbosch
        # Precompute adjacency as bitmask
        adj_mask = [0] * n
        for u in range(n):
            for v in adj[u]:
                adj_mask[u] |= (1 << v)

        maximal_cliques = []

        def bron_kerbosch(R_mask, P_mask, X_mask):
            if P_mask == 0 and X_mask == 0:
                # Count bits in R_mask
                if bin(R_mask).count('1') >= 2:
                    maximal_cliques.append(R_mask)
                return

            # Pivot selection: pick u from P ∪ X with max |P ∩ N(u)|
            candidates_pool = P_mask | X_mask
            if candidates_pool != 0:
                best_u = -1
                max_count = -1
                # Iterate over set bits in candidates_pool
                tmp = candidates_pool
                while tmp:
                    lb = tmp & (-tmp)
                    u = lb.bit_length() - 1
                    tmp ^= lb
                    count = bin(P_mask & adj_mask[u]).count('1')
                    if count > max_count:
                        max_count = count
                        best_u = u

                if best_u != -1:
                    # Candidates: vertices in P not adjacent to best_u
                    candidates_mask = P_mask & ~(adj_mask[best_u])
                else:
                    candidates_mask = P_mask
            else:
                candidates_mask = P_mask

            # Iterate over candidates
            tmp = candidates_mask
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                tmp ^= lb

                R_new = R_mask | lb
                P_new = P_mask & adj_mask[v]
                X_new = X_mask & adj_mask[v]
                bron_kerbosch(R_new, P_new, X_new)

                P_mask &= ~lb
                X_mask |= lb

        P_initial = (1 << n) - 1
        bron_kerbosch(0, P_initial, 0)

        # 3. Prepare Hitting Set Problem using bitmasks
        # cliques is list of int masks
        cliques = [c for c in maximal_cliques if bin(c).count('1') >= 2]
        num_cliques = len(cliques)

        if num_cliques == 0:
            results.append([])
            continue

        # For each vertex, compute bitmask of cliques it belongs to
        # clique_mask[v] = bitmask over cliques where v appears
        clique_mask = [0] * n
        for i, c in enumerate(cliques):
            bit = 1 << i
            tmp = c
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                tmp ^= lb
                clique_mask[v] |= bit

        # For fast uncovered tracking, precompute for each vertex the list of clique indices
        vertex_to_cliques = []
        for v in range(n):
            lst = []
            cm = clique_mask[v]
            while cm:
                lb = cm & (-cm)
                idx = lb.bit_length() - 1
                lst.append(idx)
                cm ^= lb
            vertex_to_cliques.append(lst)

        # Helper: given a set S (as bitmask over vertices), compute cover_count and uncovered
        def get_cover_state(S_mask):
            cover_count = [0] * num_cliques
            tmp = S_mask
            while tmp:
                lb = tmp & (-tmp)
                v = lb.bit_length() - 1
                tmp ^= lb
                for c_idx in vertex_to_cliques[v]:
                    cover_count[c_idx] += 1
            uncovered = 0
            for i in range(num_cliques):
                if cover_count[i] == 0:
                    uncovered |= (1 << i)
            return cover_count, uncovered

        def repair(S_mask):
            """Greedy repair: add vertices to S until all cliques covered. Returns (S_mask, cover_count, uncovered_mask)"""
            cover_count, uncovered = get_cover_state(S_mask)

            while uncovered != 0:
                best_v = -1
                max_gain = -1
                for v in range(n):
                    if S_mask & (1 << v):
                        continue
                    gain = 0
                    for c_idx in vertex_to_cliques[v]:
                        if uncovered & (1 << c_idx):
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_v = v

                if best_v == -1:
                    break

                S_mask |= (1 << best_v)
                for c_idx in vertex_to_cliques[best_v]:
                    if uncovered & (1 << c_idx):
                        uncovered &= ~(1 << c_idx)
                    cover_count[c_idx] += 1

            return S_mask, cover_count, uncovered

        def local_search(S_mask, cover_count, uncovered):
            """Remove redundant vertices. Returns (S_mask, cover_count, uncovered)"""
            changed = True
            while changed:
                changed = False
                tmp = S_mask
                while tmp:
                    lb = tmp & (-tmp)
                    v = lb.bit_length() - 1
                    tmp ^= lb

                    is_critical = False
                    for c_idx in vertex_to_cliques[v]:
                        if cover_count[c_idx] == 1:
                            is_critical = True
                            break
                    if not is_critical:
                        S_mask &= ~lb
                        for c_idx in vertex_to_cliques[v]:
                            cover_count[c_idx] -= 1
                            if cover_count[c_idx] == 0:
                                uncovered |= (1 << c_idx)
                        changed = True
                        break  # restart

            return S_mask, cover_count, uncovered

        def swap_local_search(S_mask, cover_count, uncovered):
            """Try 1-swap: remove one vertex, add another to reduce size.
            Returns (S_mask, cover_count, uncovered, improved)"""
            improved = False
            tmp = S_mask
            while tmp and not improved:
                lb = tmp & (-tmp)
                v_out = lb.bit_length() - 1
                tmp ^= lb

                # Try removing v_out
                new_cover = list(cover_count)
                for c_idx in vertex_to_cliques[v_out]:
                    new_cover[c_idx] -= 1
                new_uncovered = 0
                for i in range(num_cliques):
                    if new_cover[i] == 0:
                        new_uncovered |= (1 << i)

                if new_uncovered == 0:
                    # v_out is redundant, just remove it
                    S_mask &= ~lb
                    cover_count = new_cover
                    improved = True
                    continue

                # Need to add a vertex to cover new_uncovered
                best_v_in = -1
                best_gain = -1
                for v_in in range(n):
                    if v_in == v_out:
                        continue
                    if S_mask & (1 << v_in):
                        continue
                    gain = 0
                    for c_idx in vertex_to_cliques[v_in]:
                        if new_uncovered & (1 << c_idx):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_v_in = v_in

                if best_v_in != -1 and best_gain == bin(new_uncovered).count('1'):
                    # Full coverage with swap
                    S_mask &= ~lb
                    S_mask |= (1 << best_v_in)
                    cover_count = new_cover
                    for c_idx in vertex_to_cliques[best_v_in]:
                        cover_count[c_idx] += 1
                    uncovered = 0
                    improved = True

            # Also try pure additions if uncovered (shouldn't happen after repair)
            return S_mask, cover_count, uncovered, improved

        # 4. Greedy Initialization
        S_init = 0
        S_init, cover_count_init, uncovered_init = repair(S_init)
        S_init, cover_count_init, uncovered_init = local_search(S_init, cover_count_init, uncovered_init)

        best_S = S_init
        best_size = bin(best_S).count('1')

        # 5. Iterated Local Search with bitmask operations
        rng_ils = random.Random(seed + 500)

        def time_left():
            elapsed = time.time() - start_time
            remaining_instances = len(instances) - instances.index((n, p, seed))
            return time_limit - elapsed - margin - (remaining_instances - 1) * 2.0

        # Precompute list of vertices in S for fast random selection
        def get_vertex_list(S_mask):
            lst = []
            tmp = S_mask
            while tmp:
                lb = tmp & (-tmp)
                lst.append(lb.bit_length() - 1)
                tmp ^= lb
            return lst

        while time_left() > 0.1:
            # Perturb: Remove k random vertices from best_S
            k = rng_ils.randint(2, 6)

            S_perturbed = best_S
            vlist = get_vertex_list(S_perturbed)
            for _ in range(min(k, len(vlist))):
                if not vlist:
                    break
                idx = rng_ils.randint(0, len(vlist) - 1)
                v = vlist[idx]
                S_perturbed &= ~(1 << v)
                vlist.pop(idx)

            # Repair
            S_repaired, cc, unc = repair(S_perturbed)

            # Local Search (deletions)
            S_final, cc, unc = local_search(S_repaired, cc, unc)

            # Swap local search
            S_final, cc, unc, _ = swap_local_search(S_final, cc, unc)

            # Another deletion pass after swaps
            S_final, cc, unc = local_search(S_final, cc, unc)

            final_size = bin(S_final).count('1')
            if final_size < best_size:
                best_size = final_size
                best_S = S_final

        # Convert best_S bitmask to sorted list
        result_list = []
        tmp = best_S
        while tmp:
            lb = tmp & (-tmp)
            result_list.append(lb.bit_length() - 1)
            tmp ^= lb
        results.append(sorted(result_list))

    return results
