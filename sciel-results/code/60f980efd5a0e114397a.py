import time
import random
import numpy as np
from collections import deque
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    # Best known seed: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17
    best_edges = edges_17

    # Build adjacency matrix for Paley(17)
    adj17 = np.zeros((17, 17), dtype=np.int8)
    for u, v in edges_17:
        adj17[u, v] = 1
        adj17[v, u] = 1

    # Precompute neighbor lists for faster operations
    def get_neighbors(adj, n):
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        return neighbors

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks

    def build_non_masks(n, masks):
        """Precompute non-neighbor masks (excluding self)."""
        full_mask = (1 << n) - 1
        non_masks = [0] * n
        for u in range(n):
            non_masks[u] = full_mask & ~(masks[u] | (1 << u))
        return non_masks

    def popcount(x):
        return bin(x).count('1')

    def has_k4(n, masks):
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
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
                            return True
        return False

    def has_ind6(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))

        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
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
                                        return True
        return False

    def is_valid(n, adj):
        masks = build_masks(n, adj)
        if has_k4(n, masks):
            return False
        if has_ind6(n, masks):
            return False
        return True

    def count_violations(n, masks):
        """Count total K4 + independent 6-sets."""
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))

        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
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
                            k4_count += 1

        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
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
                                        ind6_count += 1

        return k4_count + ind6_count

    def delta_k4(n, masks, u, v, adding):
        """Delta in K4 count when toggling edge (u,v).
        adding=True means we're adding the edge (it was absent).
        """
        delta = 0
        # New K4s formed: {u,v,w,x} where w,x are common neighbors of u,v
        # and w-x is an edge. Only count each K4 once by ordering.
        common = masks[u] & masks[v]
        # For each pair (w,x) in common with w<x, check if w-x is edge
        # But we need to be careful: the K4 {u,v,w,x} is only "new" if all 6 edges exist
        # Since u-v is the edge being toggled, we need u-w, u-x, v-w, v-x, w-x all present
        # u-w and v-w are guaranteed (w in common), same for x. So just need w-x.
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
                    delta += 1
        return delta if adding else -delta

    def delta_ind6_fast(n, masks, non_masks, u, v, adding):
        """Fast delta in independent 6-set count using precomputed non_masks."""
        # Common non-neighbors of u and v (excluding u,v themselves)
        common_non = non_masks[u] & non_masks[v]
        # Remove u and v from consideration (they shouldn't be there, but be safe)
        common_non &= ~(1 << u)
        common_non &= ~(1 << v)

        # Count independent 4-sets in common_non (vertices mutually non-adjacent in G)
        count = 0
        w_mask = common_non
        while w_mask:
            lsb_w = w_mask & -w_mask
            wi = lsb_w.bit_length() - 1
            w_mask ^= lsb_w
            x_mask = common_non & ~((1 << (wi + 1)) - 1)
            while x_mask:
                lsb_x = x_mask & -x_mask
                xi = lsb_x.bit_length() - 1
                x_mask ^= lsb_x
                if masks[wi] & (1 << xi):
                    continue
                y_mask = common_non & ~((1 << (xi + 1)) - 1)
                while y_mask:
                    lsb_y = y_mask & -y_mask
                    yi = lsb_y.bit_length() - 1
                    y_mask ^= lsb_y
                    if (masks[yi] & (1 << wi)) or (masks[yi] & (1 << xi)):
                        continue
                    z_mask = common_non & ~((1 << (yi + 1)) - 1)
                    while z_mask:
                        lsb_z = z_mask & -z_mask
                        zi = lsb_z.bit_length() - 1
                        z_mask ^= lsb_z
                        if (masks[zi] & (1 << wi)) or (masks[zi] & (1 << xi)) or (masks[zi] & (1 << yi)):
                            continue
                        count += 1

        return -count if adding else count

    def update_non_masks(non_masks, masks, n, u, v, adding):
        """Update non_masks after toggling edge (u,v)."""
        if adding:
            # u and v become neighbors, so they are no longer non-neighbors of each other
            non_masks[u] &= ~(1 << v)
            non_masks[v] &= ~(1 << u)
        else:
            non_masks[u] |= (1 << v)
            non_masks[v] |= (1 << u)

    def try_find_graph(n, time_budget, seed_adj=None, density=0.5, mode="anneal"):
        """Local search with incremental cost updates.
        mode='anneal': simulated annealing
        mode='repair': only accept improving moves (steepest descent)
        """
        end_time = time.time() + time_budget

        if seed_adj is not None and seed_adj.shape == (n, n):
            adj = seed_adj.copy()
        else:
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < density:
                        adj[i, j] = 1
                        adj[j, i] = 1

        masks = build_masks(n, adj)
        non_masks = build_non_masks(n, masks)
        current_cost = count_violations(n, masks)
        if current_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        T = 5.0
        min_T = 0.05
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 20000 if mode == "repair" else 10000
        steps_since_recheck = 0
        recheck_interval = 500
        best_cost = current_cost
        best_adj = adj.copy()
        best_masks = masks[:]

        # Track recent moves for diversification
        recent_moves = set()

        while time.time() < end_time and no_improve < max_no_improve:
            # Choose move strategy
            if mode == "repair":
                # In repair mode, prefer moves that fix violations
                if random.random() < 0.6:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                else:
                    # Targeted: pick a vertex involved in many violations
                    # Approximate by picking high-degree (for K4) or low-degree (for ind6)
                    if random.random() < 0.5:
                        degs = [masks[i].bit_count() for i in range(n)]
                        u = degs.index(max(degs))
                    else:
                        ndegs = [n - 1 - masks[i].bit_count() for i in range(n)]
                        u = ndegs.index(max(ndegs))
                    v = random.randint(0, n - 1)
            else:
                if random.random() < 0.7:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                else:
                    if random.random() < 0.5:
                        degs = [masks[i].bit_count() for i in range(n)]
                        u = degs.index(max(degs))
                    else:
                        ndegs = [n - 1 - masks[i].bit_count() for i in range(n)]
                        u = ndegs.index(max(ndegs))
                    v = random.randint(0, n - 1)
            
            if u == v:
                continue

            # Avoid repeating recent moves (limit set size)
            if len(recent_moves) > 200:
                recent_moves.clear()
            key = (min(u,v), max(u,v))
            if key in recent_moves:
                continue

            old_state = adj[u, v]
            adding = (old_state == 0)

            # Compute delta
            dk4 = delta_k4(n, masks, u, v, adding)
            din6 = delta_ind6_fast(n, masks, non_masks, u, v, adding)
            delta = dk4 + din6
            new_cost = current_cost + delta

            if mode == "repair" and delta >= 0:
                # In repair mode, skip non-improving moves
                no_improve += 1
                recent_moves.add(key)
                continue

            # Apply the toggle
            adj[u, v] = 1 - old_state
            adj[v, u] = 1 - old_state
            if adding:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            else:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
            update_non_masks(non_masks, masks, n, u, v, adding)

            recent_moves.add(key)

            if new_cost < current_cost:
                current_cost = new_cost
                no_improve = 0
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj.copy()
                    best_masks = masks[:]
                if new_cost == 0:
                    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                    return edges
            else:
                if mode == "anneal" and delta > 0:
                    T = max(min_T, T * cooling)
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                        no_improve += 1
                    else:
                        # Revert
                        adj[u, v] = old_state
                        adj[v, u] = old_state
                        if adding:
                            masks[u] &= ~(1 << v)
                            masks[v] &= ~(1 << u)
                        else:
                            masks[u] |= (1 << v)
                            masks[v] |= (1 << u)
                        update_non_masks(non_masks, masks, n, u, v, not adding)
                        recent_moves.discard(key)
                        no_improve += 1
                else:
                    current_cost = new_cost
                    no_improve += 1

            # Periodic recheck
            steps_since_recheck += 1
            if steps_since_recheck >= recheck_interval:
                steps_since_recheck = 0
                actual_cost = count_violations(n, masks)
                if actual_cost != current_cost:
                    current_cost = actual_cost
                if current_cost == 0:
                    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                    return edges

        # Final check with best state
        if best_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
            return edges

        if is_valid(n, adj):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        return None

    # Strategy: try to extend from n=18 upward
    for n in range(18, 36):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 3.0:
            break

        # Allocate time: more for larger n (harder but more valuable)
        if n <= 22:
            budget = min(remaining * 0.25, 6.0)
        elif n <= 28:
            budget = min(remaining * 0.35, 10.0)
        else:
            budget = min(remaining * 0.45, 15.0)

        # Try multiple restarts with different strategies
        found = None
        num_restarts = 8
        per_restart = budget / num_restarts

        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            if time.time() > start_time + deadline - 5.0:
                break

            # Determine mode: first half anneal, second half repair
            mode = "anneal" if restart < num_restarts // 2 else "repair"

            if restart == 0:
                seed_adj = np.zeros((n, n), dtype=np.int8)
                seed_adj[:17, :17] = adj17
                for i in range(17, n):
                    for j in range(17):
                        if random.random() < 0.45:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                for i in range(17, n):
                    for j in range(i + 1, n):
                        if random.random() < 0.45:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj, density=0.45, mode=mode)
            elif restart == 1:
                seed_adj = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.48:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj, density=0.48, mode=mode)
            elif restart == 2:
                seed_adj = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.42:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj, density=0.42, mode=mode)
            elif restart == 3:
                seed_adj = np.zeros((n, n), dtype=np.int8)
                seed_adj[:17, :17] = adj17
                for i in range(17, n):
                    for j in range(17):
                        if random.random() < 0.52:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                for i in range(17, n):
                    for j in range(i + 1, n):
                        if random.random() < 0.52:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj, density=0.52, mode=mode)
            elif restart == 4:
                # Dense random
                seed_adj = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.55:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj, density=0.55, mode=mode)
            elif restart == 5:
                # Very sparse
                seed_adj = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.35:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj, density=0.35, mode=mode)
            else:
                dens = random.uniform(0.38, 0.58)
                found = try_find_graph(n, per_restart, density=dens, mode=mode)

            if found is not None:
                break

        if found is not None:
            best_n = n
            best_edges = found
            if n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}
