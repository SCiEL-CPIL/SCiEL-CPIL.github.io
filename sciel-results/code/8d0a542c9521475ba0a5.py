import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Base seed: Paley(17)
    squares17 = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares17]
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

    def count_k4_delta(n, adj, u, v, old_state):
        count = 0
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[u, a] and adj[v, a]):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[u, b] and adj[v, b]):
                    continue
                if adj[a, b]:
                    count += 1
        return count

    def count_ind6_delta(n, adj, u, v, old_state):
        count = 0
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[u, a] or adj[v, a]:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if adj[u, b] or adj[v, b]:
                    continue
                if not adj[a, b]:
                    count += 1
        return count

    def cost_full(n, adj):
        masks = build_masks(n, adj)
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

        return k4_count + ind6_count

    def is_valid(n, adj):
        return cost_full(n, adj) == 0

    def spectral_init(n, rng):
        """Generate adjacency via spectral method using structured matrices."""
        approach = rng.integers(0, 3)

        if approach == 0:
            # Paley-like using a prime >= n
            p = n
            while True:
                is_prime = True
                for d in range(2, int(p**0.5) + 1):
                    if p % d == 0:
                        is_prime = False
                        break
                if is_prime:
                    break
                p += 1
            squares = set()
            for x in range(1, p):
                squares.add(x * x % p)
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % p
                    if diff in squares:
                        adj[i, j] = 1
                        adj[j, i] = 1
            return adj
        elif approach == 1:
            # Structured Hadamard-like: use a random symmetric matrix with
            # block structure, then binarize via top eigenvector signs
            A = rng.standard_normal((n, n))
            A = (A + A.T) / 2.0
            # Add a block structure to encourage spectral separation
            num_blocks = rng.integers(2, 5)
            block_size = max(1, n // num_blocks)
            for i in range(n):
                for j in range(n):
                    if (i // block_size) == (j // block_size):
                        A[i, j] += 1.0
                    elif i != j:
                        A[i, j] -= 0.3
            eigvals, eigvecs = np.linalg.eigh(A)
            top_vec = eigvecs[:, -1]
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if top_vec[i] * top_vec[j] > 0:
                        adj[i, j] = 1
                        adj[j, i] = 1
            return adj
        else:
            # Bipartite-like with random perturbation
            adj = np.zeros((n, n), dtype=np.int8)
            mid = n // 2
            for i in range(mid):
                for j in range(mid, n):
                    if rng.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
            for i in range(mid):
                for j in range(i + 1, mid):
                    if rng.random() < 0.2:
                        adj[i, j] = 1
                        adj[j, i] = 1
            for i in range(mid, n):
                for j in range(i + 1, n):
                    if rng.random() < 0.2:
                        adj[i, j] = 1
                        adj[j, i] = 1
            return adj

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

        current_cost = cost_full(n, adj)
        if current_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        T = 3.0
        min_T = 0.01
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 1000

        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue

            old_state = adj[u, v]
            adj[u, v] = 1 - old_state
            adj[v, u] = 1 - old_state

            if old_state == 0:
                k4_delta = count_k4_delta(n, adj, u, v, old_state)
                ind6_delta = -count_ind6_delta(n, adj, u, v, old_state)
            else:
                k4_delta = -count_k4_delta(n, adj, u, v, old_state)
                ind6_delta = count_ind6_delta(n, adj, u, v, old_state)

            new_cost = current_cost + k4_delta + ind6_delta

            if new_cost < current_cost:
                current_cost = new_cost
                no_improve = 0
                if current_cost <= 0:
                    if is_valid(n, adj):
                        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                        return edges
            else:
                delta = new_cost - current_cost
                T = max(min_T, T * cooling)
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    no_improve += 1
                else:
                    adj[u, v] = old_state
                    adj[v, u] = old_state
                    no_improve += 1

        if is_valid(n, adj):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        return None

    # Search from n=18 upward, with more restarts and spectral init
    for n in range(18, 36):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        if n <= 22:
            budget = min(remaining * 0.35, 12.0)
        elif n <= 28:
            budget = min(remaining * 0.30, 10.0)
        else:
            budget = min(remaining * 0.25, 8.0)

        found = None
        num_restarts = 8
        per_restart = budget / num_restarts

        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            if restart == 0:
                rng = np.random.default_rng(random.randint(0, 2**31))
                seed_adj = spectral_init(n, rng)
                found = try_find_graph(n, per_restart, seed_adj=seed_adj)
            elif restart == 1:
                rng = np.random.default_rng(random.randint(0, 2**31))
                seed_adj = spectral_init(n, rng)
                found = try_find_graph(n, per_restart, seed_adj=seed_adj)
            elif restart == 2:
                if n >= 18:
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
                else:
                    found = try_find_graph(n, per_restart)
            elif restart == 3:
                # Another Paley-style with different prime
                rng = np.random.default_rng(random.randint(0, 2**31))
                seed_adj = spectral_init(n, rng)
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
