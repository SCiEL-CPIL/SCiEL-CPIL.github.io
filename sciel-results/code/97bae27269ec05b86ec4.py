import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    best_n = 30
    best_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]

    def get_edges_from_adj_bitmask(adj_list, n):
        edges = []
        for u in range(n):
            mask = adj_list[u]
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append([u, v])
                m ^= lsb
        return edges

    def is_c4_free_bitmask(adj_list, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_list[u] & adj_list[v]
                if common.bit_count() > 1:
                    return False
        return True

    def has_independent_set_exact(adj_list, n, k):
        all_mask = (1 << n) - 1
        comp_adj = []
        for i in range(n):
            comp_mask = all_mask ^ (1 << i) ^ adj_list[i]
            comp_adj.append(comp_mask)

        def greedy_color_bound(candidates_mask):
            if candidates_mask == 0:
                return 0
            colors = 0
            remaining = candidates_mask
            color_class = 0
            color_mask = 0
            while remaining:
                v_bit = remaining & -remaining
                v = v_bit.bit_length() - 1
                remaining ^= v_bit
                if (comp_adj[v] & color_mask) == 0:
                    color_mask |= v_bit
                else:
                    colors += 1
                    color_class = v_bit
                    color_mask = v_bit
            colors += 1
            return colors

        def backtrack(candidates_mask, clique_size):
            if clique_size >= k:
                return True
            if candidates_mask.bit_count() < k - clique_size:
                return False
            bound = greedy_color_bound(candidates_mask)
            if clique_size + bound < k:
                return False

            cand_list = []
            temp = candidates_mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                cand_list.append(v)
                temp ^= lsb

            if not cand_list:
                return False

            cand_list.sort(key=lambda v: (comp_adj[v] & candidates_mask).bit_count(), reverse=True)

            v = cand_list[0]
            v_bit = 1 << v

            new_mask = candidates_mask & comp_adj[v]
            new_mask &= ~v_bit

            if backtrack(new_mask, clique_size + 1):
                return True

            candidates_mask &= ~v_bit
            if backtrack(candidates_mask, clique_size):
                return True

            return False

        initial_mask = (1 << n) - 1
        return backtrack(initial_mask, 0)

    def try_greedy_construction(n, seed):
        random.seed(seed)
        adj_list = [0] * n

        order = list(range(n))
        random.shuffle(order)

        for v in order:
            candidates = [u for u in order if u in set(order[:order.index(v)])]
            random.shuffle(candidates)

            for u in candidates:
                common = adj_list[v] & adj_list[u]
                if common.bit_count() >= 2:
                    continue

                valid = True
                temp_mask = adj_list[u]
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    w = lsb.bit_length() - 1
                    if (adj_list[w] & adj_list[v]) != 0:
                        valid = False
                        break
                    temp_mask ^= lsb

                if not valid:
                    continue

                adj_list[v] |= (1 << u)
                adj_list[u] |= (1 << v)

        if not is_c4_free_bitmask(adj_list, n):
            return None

        if has_independent_set_exact(adj_list, n, 11):
            return None

        return get_edges_from_adj_bitmask(adj_list, n)

    # Try to construct for n=43 down to 39
    for n in range(43, 38, -1):
        if time.time() > deadline - 10:
            break

        found = False
        for seed in range(200):
            if time.time() > deadline - 5:
                break

            edges = try_greedy_construction(n, seed)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                if n == 43:
                    return {'n': 43, 'edges': edges}

        if found:
            break

    return {'n': best_n, 'edges': best_edges}
