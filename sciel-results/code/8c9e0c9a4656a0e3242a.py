import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17
    best_edges = edges_17

    adj17 = np.zeros((17, 17), dtype=np.int8)
    for u, v in edges_17:
        adj17[u, v] = 1
        adj17[v, u] = 1

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks

    def build_comp_masks(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        return comp_masks

    def count_k4_through_pair(n, masks, u, v):
        if not (masks[u] & (1 << v)):
            return 0
        count = 0
        common = masks[u] & masks[v]
        common &= ~((1 << (v + 1)) - 1)
        w_mask = common
        while w_mask:
            lsb_w = w_mask & -w_mask
            wi = lsb_w.bit_length() - 1
            w_mask ^= lsb_w
            x_mask = common & ~((1 << (wi + 1)) - 1)
            while x_mask:
                lsb_x = x_mask & -x_mask
                xi = lsb_x.bit_length() - 1
                x_mask ^= lsb_x
                if masks[wi] & (1 << xi):
                    count += 1
        return count

    def count_ind6_through_pair(n, comp_masks, u, v):
        if not (comp_masks[u] & (1 << v)):
            return 0
        count = 0
        common = comp_masks[u] & comp_masks[v]
        common &= ~((1 << (v + 1)) - 1)
        w_mask = common
        while w_mask:
            lsb_w = w_mask & -w_mask
            wi = lsb_w.bit_length() - 1
            w_mask ^= lsb_w
            x_mask = common & ~((1 << (wi + 1)) - 1)
            while x_mask:
                lsb_x = x_mask & -x_mask
                xi = lsb_x.bit_length() - 1
                x_mask ^= lsb_x
                if comp_masks[wi] & (1 << xi):
                    common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                    common2 &= ~((1 << (xi + 1)) - 1)
                    y_mask = common2
                    while y_mask:
                        lsb_y = y_mask & -y_mask
                        yi = lsb_y.bit_length() - 1
                        y_mask ^= lsb_y
                        z_mask = common2 & ~((1 << (yi + 1)) - 1)
                        while z_mask:
                            lsb_z = z_mask & -z_mask
                            zi = lsb_z.bit_length() - 1
                            z_mask ^= lsb_z
                            if comp_masks[yi] & (1 << zi):
                                count += 1
        return count

    def total_cost(n, masks, comp_masks):
        k4_total = 0
        for u in range(n):
            for v in range(u + 1, n):
                k4_total += count_k4_through_pair(n, masks, u, v)
        ind6_total = 0
        for u in range(n):
            for v in range(u + 1, n):
                ind6_total += count_ind6_through_pair(n, comp_masks, u, v)
        return k4_total + ind6_total

    def is_valid(n, masks, comp_masks):
        for u in range(n):
            for v in range(u + 1, n):
                if count_k4_through_pair(n, masks, u, v) > 0:
                    return False
        for u in range(n):
            for v in range(u + 1, n):
                if count_ind6_through_pair(n, comp_masks, u, v) > 0:
                    return False
        return True

    def try_find_graph(n, time_budget, seed_adj=None):
        end_time = time.time() + time_budget

        if seed_adj is not None and seed_adj.shape == (n, n):
            adj = seed_adj.copy()
        else:
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1

        masks = build_masks(n, adj)
        comp_masks = build_comp_masks(n, masks)
        current_cost = total_cost(n, masks, comp_masks)

        if current_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        T = 5.0
        min_T = 0.01
        cooling = 0.9998
        no_improve = 0
        max_no_improve = 10000
        full_mask = (1 << n) - 1

        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u

            old_state = adj[u, v]

            if old_state == 0:
                # Adding edge (u,v)
                # New K4s: {u,v,w,x} where w,x common neighbors, w-x edge
                common_nv = masks[u] & masks[v]
                common_nv &= ~(1 << u)
                common_nv &= ~(1 << v)
                new_k4 = 0
                w_mask = common_nv
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common_nv & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            new_k4 += 1
                # Destroyed ind6: ind6-sets containing u,v (they were non-adjacent)
                destroyed_ind6 = count_ind6_through_pair(n, comp_masks, u, v)
                delta = new_k4 - destroyed_ind6
            else:
                # Removing edge (u,v)
                destroyed_k4 = count_k4_through_pair(n, masks, u, v)
                # New ind6: need to count ind6-sets in new state containing u,v
                temp_comp = comp_masks[:]
                temp_comp[u] |= (1 << v)
                temp_comp[v] |= (1 << u)
                new_ind6 = count_ind6_through_pair(n, temp_comp, u, v)
                delta = -destroyed_k4 + new_ind6

            if delta < 0:
                adj[u, v] = 1
                adj[v, u] = 1
                if old_state == 0:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    comp_masks[u] &= ~(1 << v)
                    comp_masks[v] &= ~(1 << u)
                else:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    comp_masks[u] |= (1 << v)
                    comp_masks[v] |= (1 << u)
                current_cost += delta
                no_improve = 0
                if current_cost <= 0:
                    if is_valid(n, masks, comp_masks):
                        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                        return edges
            else:
                T = max(min_T, T * cooling)
                if random.random() < np.exp(-delta / T):
                    adj[u, v] = 1
                    adj[v, u] = 1
                    if old_state == 0:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        comp_masks[u] &= ~(1 << v)
                        comp_masks[v] &= ~(1 << u)
                    else:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                        comp_masks[u] |= (1 << v)
                        comp_masks[v] |= (1 << u)
                    current_cost += delta
                    no_improve += 1
                else:
                    no_improve += 1

        if is_valid(n, masks, comp_masks):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges
        return None

    for n in range(18, 36):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 3.0:
            break

        if n <= 22:
            budget = min(remaining * 0.4, 12.0)
        elif n <= 28:
            budget = min(remaining * 0.3, 8.0)
        else:
            budget = min(remaining * 0.2, 5.0)

        found = None
        num_restarts = 4
        per_restart = budget / num_restarts

        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            if restart == 0:
                seed_adj = np.zeros((n, n), dtype=np.int8)
                seed_adj[:17, :17] = adj17
                for i in range(17, n):
                    for j in range(17):
                        if random.random() < 0.5:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                for i in range(17, n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj)
            elif restart == 1:
                seed_adj = np.zeros((n, n), dtype=np.int8)
                seed_adj[:17, :17] = adj17
                for i in range(17, n):
                    for j in range(17):
                        if random.random() < 0.6:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                for i in range(17, n):
                    for j in range(i + 1, n):
                        if random.random() < 0.4:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj)
            else:
                found = try_find_graph(n, per_restart)

            if found is not None:
                break

        if found is not None:
            best_n = n
            best_edges = found
            if n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}
