import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def count_k4(n, masks):
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
        return k4_count

    def count_ind6(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
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
        return ind6_count

    def total_violations(n, adj):
        masks = build_masks(n, adj)
        return count_k4(n, masks) + count_ind6(n, masks)

    def edges_from_adj(n, adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    # Clebsch graph edges (16 vertices, K4-free, alpha=5)
    clebsch_edges = [
        (0, 6), (0, 7), (0, 10), (0, 11),
        (1, 5), (1, 8), (1, 10), (1, 13),
        (2, 4), (2, 9), (2, 11), (2, 13),
        (3, 4), (3, 7), (3, 9), (3, 12),
        (5, 6), (5, 8), (5, 12), (5, 15),
        (6, 7), (6, 9), (6, 14), (6, 15),
        (7, 8), (7, 11), (7, 14), (7, 15),
        (8, 9), (8, 12), (8, 15),
        (9, 10), (9, 14),
        (10, 11), (10, 13), (10, 14),
        (11, 12), (11, 14), (11, 15),
        (12, 13), (12, 15),
        (13, 14), (13, 15)
    ]
    
    best_n = 16
    best_edges = list(clebsch_edges)

    # Verify baseline
    adj_16 = np.zeros((16, 16), dtype=bool)
    for u, v in clebsch_edges:
        adj_16[u, v] = True
        adj_16[v, u] = True
    
    masks_16 = build_masks(16, adj_16)
    if count_k4(16, masks_16) > 0 or count_ind6(16, masks_16) > 0:
        best_n = 5
        best_edges = []

    def random_init(n, density, rng):
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    def degree_biased_edge_select(n, adj, rng):
        """Select an edge with probability biased toward high-degree vertices."""
        degrees = adj.sum(axis=1).astype(float)
        total = degrees.sum()
        if total == 0:
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            while u == v:
                v = rng.randint(0, n - 1)
            return u, v
        
        # Use degree as weight for selection
        weights = degrees / total
        u = rng.choice(range(n), p=weights)
        v = rng.randint(0, n - 1)
        while v == u:
            v = rng.randint(0, n - 1)
        return u, v

    def simulated_annealing(n, adj, budget, rng):
        """SA with degree-biased selection and better cooling schedule."""
        current_cost = total_violations(n, adj)
        T_init = 5.0
        T_min = 0.01
        max_steps = int(budget * 80)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            T = T_init * (T_min / T_init) ** (step / max(1, max_steps))
            
            # Choose move type
            if rng.random() < 0.6:
                # Single edge flip with degree bias
                u, v = degree_biased_edge_select(n, adj, rng)
                old = adj[u, v]
                adj[u, v] = not old
                adj[v, u] = not old
                new_cost = total_violations(n, adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if new_cost == 0:
                        return adj, 0
                else:
                    delta = new_cost - current_cost
                    if rng.random() < np.exp(-delta / max(T, 1e-10)):
                        current_cost = new_cost
                    else:
                        adj[u, v] = old
                        adj[v, u] = old
            else:
                # Double edge flip
                u1, v1 = degree_biased_edge_select(n, adj, rng)
                u2, v2 = degree_biased_edge_select(n, adj, rng)
                if u1 == v1 or u2 == v2:
                    continue
                old1 = adj[u1, v1]
                old2 = adj[u2, v2]
                adj[u1, v1] = not old1
                adj[v1, u1] = not old1
                adj[u2, v2] = not old2
                adj[v2, u2] = not old2
                new_cost = total_violations(n, adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if new_cost == 0:
                        return adj, 0
                else:
                    delta = new_cost - current_cost
                    if rng.random() < np.exp(-delta / max(T, 1e-10)):
                        current_cost = new_cost
                    else:
                        adj[u1, v1] = old1
                        adj[v1, u1] = old1
                        adj[u2, v2] = old2
                        adj[v2, u2] = old2
        
        return adj, current_cost

    # Try to find valid graphs for increasing n
    for target_n in range(17, 36):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        
        # Allocate more budget to larger n
        if target_n <= 22:
            budget_per_restart = min(remaining * 0.15, 4.0)
            num_restarts = max(3, int(remaining / 5.0))
        elif target_n <= 28:
            budget_per_restart = min(remaining * 0.12, 3.5)
            num_restarts = max(4, int(remaining / 4.0))
        else:
            budget_per_restart = min(remaining * 0.1, 3.0)
            num_restarts = max(5, int(remaining / 3.0))
        
        found = False
        densities = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        
        for restart in range(num_restarts):
            if time.time() > deadline or found:
                break
            
            rng = random.Random(restart * 7919 + target_n * 104729)
            density = densities[restart % len(densities)]
            
            adj = random_init(target_n, density, rng)
            current_cost = total_violations(target_n, adj)
            
            if current_cost == 0:
                found = True
                best_n = target_n
                best_edges = edges_from_adj(target_n, adj)
                break
            
            adj, final_cost = simulated_annealing(target_n, adj, budget_per_restart, rng)
            
            if final_cost == 0:
                found = True
                best_n = target_n
                best_edges = edges_from_adj(target_n, adj)
                break
        
        if found:
            if best_n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}
