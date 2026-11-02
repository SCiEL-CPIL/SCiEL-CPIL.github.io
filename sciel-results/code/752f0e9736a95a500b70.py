import numpy as np
import itertools
import time
from collections import deque

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Generate candidates with squared norm exactly 4
    candidates = []

    # Type 1: One ±2, rest 0 (squared norm = 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four ±1s, rest 0 (squared norm = 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix
    M = candidates @ candidates.T

    # Conflict: for norm-4 vectors, need dist_sq >= 4
    # dist_sq = 4 + 4 - 2*M[i,j] = 8 - 2*M[i,j] >= 4
    # => M[i,j] <= 2
    # Conflict if M[i,j] > 2, i.e., M[i,j] >= 4 (since dot products are even here)
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)

    # Try to find bipartite partition
    # Partition by: sum of coordinates mod 2? All have even sum (one 2 or four 1s)
    # Try: number of negative coordinates mod 2
    neg_counts = np.sum(candidates < 0, axis=1)
    parity = neg_counts % 2

    # Verify bipartite: no edge within same parity group
    left_mask = (parity == 0)
    right_mask = (parity == 1)
    # Check: Conflict should be zero within left and within right
    left_indices = np.where(left_mask)[0]
    right_indices = np.where(right_mask)[0]

    # Verify bipartiteness
    is_bipartite = True
    if len(left_indices) > 0:
        if Conflict[np.ix_(left_indices, left_indices)].any():
            is_bipartite = False
    if len(right_indices) > 0:
        if Conflict[np.ix_(right_indices, right_indices)].any():
            is_bipartite = False

    if is_bipartite:
        # Use Hopcroft-Karp for maximum matching, then König's theorem
        # for minimum vertex cover, then complement is maximum independent set

        def hopcroft_karp(adj, n_left, n_right):
            """
            adj: list of lists, adj[u] = list of right nodes adjacent to left node u
            n_left, n_right: number of nodes in each partition
            Returns: matching dict {left_node: right_node}
            """
            pair_u = [-1] * n_left
            pair_v = [-1] * n_right
            dist = [0] * n_left

            def bfs():
                queue = deque()
                for u in range(n_left):
                    if pair_u[u] == -1:
                        dist[u] = 0
                        queue.append(u)
                    else:
                        dist[u] = -1
                found = False
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        pu = pair_v[v]
                        if pu == -1:
                            found = True
                        elif dist[pu] == -1:
                            dist[pu] = dist[u] + 1
                            queue.append(pu)
                return found

            def dfs(u):
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                        pair_u[u] = v
                        pair_v[v] = u
                        return True
                dist[u] = -1
                return False

            matching = 0
            while bfs():
                for u in range(n_left):
                    if pair_u[u] == -1:
                        if dfs(u):
                            matching += 1
            return pair_u, pair_v

        # Build adjacency from left to right
        # left_indices maps to 0..n_left-1, right_indices maps to 0..n_right-1
        n_left = len(left_indices)
        n_right = len(right_indices)

        # Map original indices to partition-local indices
        left_local = {orig: loc for loc, orig in enumerate(left_indices)}
        right_local = {orig: loc for loc, orig in enumerate(right_indices)}

        # Build adjacency list
        adj = [[] for _ in range(n_left)]
        for loc_l, orig_l in enumerate(left_indices):
            for loc_r, orig_r in enumerate(right_indices):
                if Conflict[orig_l, orig_r]:
                    adj[loc_l].append(loc_r)

        pair_u, pair_v = hopcroft_karp(adj, n_left, n_right)

        # Minimum vertex cover via König's theorem
        # Unmatched left nodes, then BFS alternating paths
        visited_left = [False] * n_left
        visited_right = [False] * n_right

        # Start BFS from unmatched left nodes
        queue = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                visited_left[u] = True
                queue.append(u)

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited_right[v]:
                    visited_right[v] = True
                    pu = pair_v[v]
                    if pu != -1 and not visited_left[pu]:
                        visited_left[pu] = True
                        queue.append(pu)

        # Minimum vertex cover:
        # Left nodes NOT visited + Right nodes visited
        min_cover_left = set()
        min_cover_right = set()
        for u in range(n_left):
            if not visited_left[u]:
                min_cover_left.add(u)
        for v in range(n_right):
            if visited_right[v]:
                min_cover_right.add(v)

        # Maximum independent set = all nodes - minimum vertex cover
        independent_left = set(range(n_left)) - min_cover_left
        independent_right = set(range(n_right)) - min_cover_right

        # Convert back to original indices
        best_set = []
        for loc in independent_left:
            best_set.append(left_indices[loc])
        for loc in independent_right:
            best_set.append(right_indices[loc])

    else:
        # Fallback: greedy + local search
        def run_greedy(seed):
            if time.time() > deadline:
                return []

            rng = np.random.default_rng(seed)
            active = np.ones(N, dtype=bool)
            deg = Conflict.sum(axis=1).astype(np.int64)
            selected = []

            while active.any():
                if time.time() > deadline:
                    break

                active_indices = np.where(active)[0]
                if len(active_indices) == 0:
                    break

                active_deg = deg[active_indices]
                min_deg = np.min(active_deg)
                candidates_for_pick = active_indices[active_deg == min_deg]
                idx = rng.choice(candidates_for_pick)

                selected.append(idx)

                neighbors_mask = Conflict[idx].astype(bool)
                removed_mask = np.zeros(N, dtype=bool)
                removed_mask[idx] = True
                removed_mask[neighbors_mask & active] = True

                removed_indices = np.where(removed_mask)[0]

                if len(removed_indices) > 0:
                    removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                    deg[active] -= removal_counts[active]

                active[removed_mask] = False

            return selected

        def local_search_fast(selected_indices):
            if time.time() > deadline:
                return selected_indices

            selected_set = set(selected_indices)
            selected_arr = np.array(list(selected_set))

            improved = True
            max_iters = 30
            iters = 0

            while improved and iters < max_iters and time.time() < deadline:
                improved = False
                iters += 1

                rng_ls = np.random.default_rng()
                order = rng_ls.permutation(len(selected_arr))
                selected_arr = selected_arr[order]

                for i_idx in range(len(selected_arr)):
                    if time.time() > deadline:
                        break

                    i = selected_arr[i_idx]

                    neighbors = np.where(Conflict[i])[0]
                    valid_js = [j for j in neighbors if j not in selected_set]

                    if len(valid_js) < 2:
                        continue

                    valid_js_arr = np.array(valid_js)

                    rest_mask = np.zeros(N, dtype=bool)
                    rest_mask[list(selected_set)] = True
                    rest_mask[i] = False

                    conflict_with_rest = Conflict[valid_js_arr, :].sum(axis=1)
                    compatible_js = valid_js_arr[conflict_with_rest == 0]

                    if len(compatible_js) < 2:
                        continue

                    found = False
                    for p in range(len(compatible_js)):
                        if found:
                            break
                        j1 = compatible_js[p]
                        rest_compatible = compatible_js[p+1:]
                        if len(rest_compatible) == 0:
                            continue
                        comp_with_j1 = Conflict[j1, rest_compatible] == 0
                        if comp_with_j1.any():
                            j2 = rest_compatible[np.where(comp_with_j1)[0][0]]
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            improved = True
                            found = True
                            break

                    if found:
                        break

            return list(selected_set)

        best_set = []
        best_size = 0

        num_restarts = 100
        for s in range(num_restarts):
            if time.time() > deadline:
                break

            res = run_greedy(s)
            if not res:
                continue

            res_improved = local_search_fast(res)

            if len(res_improved) > best_size:
                best_size = len(res_improved)
                best_set = res_improved

        if not best_set:
            best_set = list(range(min(22, N)))

    # Verify the solution is valid
    if len(best_set) >= 2:
        subset = candidates[best_set]
        norms_sq = np.sum(subset ** 2, axis=1)
        max_norm_sq = np.max(norms_sq)
        # Check all pairwise distances
        dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * (subset @ subset.T)
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
        if min_dist_sq < max_norm_sq:
            # Invalid, fall back to greedy
            def run_greedy_fallback(seed):
                if time.time() > deadline:
                    return []
                rng = np.random.default_rng(seed)
                active = np.ones(N, dtype=bool)
                deg = Conflict.sum(axis=1).astype(np.int64)
                selected = []
                while active.any():
                    if time.time() > deadline:
                        break
                    active_indices = np.where(active)[0]
                    if len(active_indices) == 0:
                        break
                    active_deg = deg[active_indices]
                    min_deg = np.min(active_deg)
                    candidates_for_pick = active_indices[active_deg == min_deg]
                    idx = rng.choice(candidates_for_pick)
                    selected.append(idx)
                    neighbors_mask = Conflict[idx].astype(bool)
                    removed_mask = np.zeros(N, dtype=bool)
                    removed_mask[idx] = True
                    removed_mask[neighbors_mask & active] = True
                    removed_indices = np.where(removed_mask)[0]
                    if len(removed_indices) > 0:
                        removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                        deg[active] -= removal_counts[active]
                    active[removed_mask] = False
                return selected

            best_set = run_greedy_fallback(42)

    return candidates[best_set].astype(np.int64)
