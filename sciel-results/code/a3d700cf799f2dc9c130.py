"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit

    n = adjacency.shape[0]
    adjacency = adjacency.astype(bool)

    # Precompute neighbor lists for faster simulation
    neighbor_lists = [np.where(adjacency[i])[0] for i in range(n)]

    # Precompute neighbor mask arrays for vectorized simulation
    neighbor_masks = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if len(neighbor_lists[i]) > 0:
            neighbor_masks[i, neighbor_lists[i]] = True

    def simulate_greedy(order):
        """Simulate greedy decoder given an explicit vertex order (descending priority)."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                if len(neighbor_lists[v]) > 0:
                    selected[neighbor_lists[v]] = True
        return count

    def simulate_greedy_fast(order):
        """Faster simulation using precomputed masks."""
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                selected |= neighbor_masks[v]
        return count

    degrees = adjacency.sum(axis=1).astype(float)
    rng = np.random.RandomState(42)

    # Compute iterative relaxation scores (captures global structure)
    def compute_relaxation_scores(max_iters=50, lr=0.5):
        """Iterative relaxation: score_i = 1/(1+sum_j in N(i) score_j)"""
        scores = np.ones(n)
        for _ in range(max_iters):
            if time.time() > deadline - 30:
                break
            # sum of neighbor scores
            neighbor_score_sums = adjacency.astype(float) @ scores
            new_scores = 1.0 / (1.0 + neighbor_score_sums)
            scores = (1 - lr) * scores + lr * new_scores
        return scores

    relaxation_scores = compute_relaxation_scores(max_iters=80, lr=0.4)

    # Mean degree of neighbors (structural feature)
    sum_neighbor_degrees = adjacency.astype(float) @ degrees
    avg_neighbor_degree = np.zeros(n)
    nonzero_mask = degrees > 0
    avg_neighbor_degree[nonzero_mask] = sum_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]

    # Generate several initial orderings to start local search from
    initial_orders = []
    # 1. Ascending degree (low degree first)
    initial_orders.append(np.argsort(degrees))
    # 2. Descending degree (high degree first)
    initial_orders.append(np.argsort(-degrees))
    # 3. Random
    initial_orders.append(rng.permutation(n))
    # 4. Degree with small random noise for tie-breaking
    deg_noise = degrees + rng.uniform(0, 1, n) * 0.01
    initial_orders.append(np.argsort(deg_noise))
    # 5. Another random
    initial_orders.append(rng.permutation(n))
    # 6. Relaxation scores (higher score = more "independent" = pick first)
    initial_orders.append(np.argsort(-relaxation_scores))
    # 7. Combined: low degree + high relaxation score
    combined = degrees / (degrees.max() + 1e-9) - relaxation_scores / (relaxation_scores.max() + 1e-9)
    initial_orders.append(np.argsort(combined))
    # 8. Low avg neighbor degree (prefer vertices whose neighbors have low degree)
    initial_orders.append(np.argsort(avg_neighbor_degree))
    # 9. Random perturbation of relaxation
    relax_noise = relaxation_scores + rng.uniform(0, 1, n) * 0.05
    initial_orders.append(np.argsort(-relax_noise))
    # 10. Another random
    initial_orders.append(rng.permutation(n))

    best_order = initial_orders[0].copy()
    best_size = simulate_greedy_fast(best_order)

    for idx, init_order in enumerate(initial_orders):
        if time.time() > deadline - 15:
            break
        curr_order = init_order.copy()
        curr_size = simulate_greedy_fast(curr_order)

        # Phase 1: Adjacent swap local search (multiple passes until no improvement)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(n - 1):
                if time.time() > deadline - 10:
                    break
                new_order = curr_order.copy()
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                new_size = simulate_greedy_fast(new_order)
                if new_size > curr_size:
                    curr_size = new_size
                    curr_order = new_order
                    improved = True

        # Phase 2: Random swap local search to escape local optima
        no_improve = 0
        max_no_improve = 500
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            i, j = rng.randint(0, n, 2)
            if i == j:
                continue
            new_order = curr_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_size = simulate_greedy_fast(new_order)
            if new_size > curr_size:
                curr_size = new_size
                curr_order = new_order
                no_improve = 0
            else:
                no_improve += 1

        if curr_size > best_size:
            best_size = curr_size
            best_order = curr_order

    # Final polish: more thorough adjacent swap passes on the best order
    improved = True
    passes = 0
    while improved and time.time() < deadline - 2 and passes < 5:
        improved = False
        passes += 1
        for i in range(n - 1):
            if time.time() > deadline - 2:
                break
            new_order = best_order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            new_size = simulate_greedy_fast(new_order)
            if new_size > best_size:
                best_size = new_size
                best_order = new_order
                improved = True

    # Convert order to scores: earlier in order = higher score
    scores = np.zeros(n)
    for pos, v in enumerate(best_order):
        scores[v] = -pos

    return scores


# EVOLVE-BLOCK-END
