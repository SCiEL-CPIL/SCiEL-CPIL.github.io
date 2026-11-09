"""Priority function for Maximum Independent Set via greedy decoder optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    if n == 0:
        return np.array([])

    # Precompute neighbor lists
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    degrees = np.array([len(nei) for nei in neighbors])

    # Precompute neighbor masks for fast vectorized blocking
    neighbor_masks = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if len(neighbors[i]) > 0:
            neighbor_masks[i, neighbors[i]] = True

    def greedy_mis_size(order):
        """Evaluate the greedy independent set size for a given vertex order."""
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not blocked[v]:
                count += 1
                if len(neighbors[v]) > 0:
                    blocked |= neighbor_masks[v]
        return count

    # --- Initialization: Multi-heuristic score combination ---
    # Compute coreness (k-shell decomposition)
    coreness = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    deg_rem = degrees.copy().astype(float)
    k = 0
    while remaining.any():
        min_deg = np.inf
        for v in range(n):
            if remaining[v] and deg_rem[v] < min_deg:
                min_deg = deg_rem[v]
        if min_deg < 0:
            min_deg = 0
        candidates = [v for v in range(n) if remaining[v] and deg_rem[v] <= min_deg + 1e-9]
        for v in candidates:
            coreness[v] = k
            remaining[v] = False
        for u in range(n):
            if remaining[u]:
                for v in candidates:
                    if adj[u, v]:
                        deg_rem[u] -= 1
        k += 1

    max_core = coreness.max() if coreness.max() > 0 else 1
    core_score = 1.0 - (coreness.astype(float) / max_core)

    max_deg = degrees.max() if degrees.max() > 0 else 1
    deg_score = 1.0 - (degrees.astype(float) / max_deg)

    # Combined initial scores
    w_core = 0.5
    w_deg = 0.5
    scores = w_core * core_score + w_deg * deg_score

    # Initial order from combined scores (descending)
    current_order = list(np.argsort(-scores, kind='stable'))
    current_size = greedy_mis_size(current_order)

    best_order = current_order[:]
    best_size = current_size

    rng = np.random.default_rng(12345)

    # --- Simulated Annealing on the ordering ---
    num_restarts = 4
    steps_per_restart = 20000

    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break

        if r == 0:
            order = current_order[:]
            cur_size = current_size
        else:
            order = best_order[:]
            # Perturb best solution for restart
            for _ in range(8):
                i, j = rng.integers(0, n, 2)
                order[i], order[j] = order[j], order[i]
            cur_size = greedy_mis_size(order)

        if cur_size > best_size:
            best_size = cur_size
            best_order = order[:]

        temp = 3.0
        cooling = 0.99985
        min_temp = 0.05

        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break

            # Swap two random positions
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue

            vi = order[i]
            vj = order[j]
            order[i] = vj
            order[j] = vi

            new_size = greedy_mis_size(order)
            delta = cur_size - new_size

            if delta < 0:
                # Improvement
                cur_size = new_size
                if cur_size > best_size:
                    best_size = cur_size
                    best_order = order[:]
            elif delta == 0:
                # Neutral
                cur_size = new_size
            else:
                # Worsening: accept with probability exp(-delta/temp)
                if rng.random() < np.exp(-delta / max(temp, 1e-10)):
                    cur_size = new_size
                else:
                    # Reject: swap back
                    order[i] = vi
                    order[j] = vj

            temp *= cooling
            if temp < min_temp:
                break

    # Convert best_order to scores (first element gets highest score)
    final_scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        final_scores[v] = n - rank

    return final_scores.astype(float)

# EVOLVE-BLOCK-END
