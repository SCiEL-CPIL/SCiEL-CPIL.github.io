"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _aco_eval(dm, demands, capacity, H, n_ants, n_iterations, seed, time_limit):
    """
    Simulate the evaluator's ACO to estimate heuristic quality.
    """
    start_time = time.time()
    n = dm.shape[0]
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.RandomState(seed)
    best_cost = np.inf
    demands = np.asarray(demands, dtype=np.float64)

    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break

        iteration_costs = []

        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break

            current_node = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0

            while True:
                valid_mask = ~visited
                valid_mask[0] = False
                valid_mask &= (demands <= remaining_cap)

                if not np.any(valid_mask):
                    total_dist += dm[current_node, 0]
                    remaining_cap = capacity
                    current_node = 0
                    if np.all(visited):
                        break
                    continue

                row_tau = tau[current_node]
                row_eta = H[current_node]

                weights = np.zeros(n)
                weights[valid_mask] = row_tau[valid_mask] * row_eta[valid_mask]

                total_w = np.sum(weights)
                if total_w <= 0:
                    valid_indices = np.where(valid_mask)[0]
                    next_node = rng.choice(valid_indices)
                else:
                    probs = weights / total_w
                    next_node = rng.choice(n, p=probs)

                total_dist += dm[current_node, next_node]
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current_node = next_node

                if np.all(visited):
                    total_dist += dm[current_node, 0]
                    break

            iteration_costs.append(total_dist)

        if iteration_costs:
            current_best = min(iteration_costs)
            if current_best < best_cost:
                best_cost = current_best
            tau *= 0.9
            tau += 1.0 / current_best

    return best_cost


def _evaluate_heuristic(dm, demands, capacity, H, time_limit, n_seeds=3):
    """
    Evaluate a heuristic by running ACO simulations with multiple seeds.
    Returns the average best cost.
    """
    costs = []
    for s in range(n_seeds):
        cost = _aco_eval(dm, demands, capacity, H,
                         n_ants=8, n_iterations=8, seed=s,
                         time_limit=time_limit / n_seeds)
        if cost < np.inf:
            costs.append(cost)
    if costs:
        return np.mean(costs)
    return np.inf


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO using a search approach.
    Generates many varied candidate heuristics and keeps the best one found.
    Focus on demand-aware, multi-criteria heuristics.
    """
    start_time = time.time()
    deadline = start_time + 315.0  # Safety margin before 330s kill

    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)

    dm_safe = np.where(dm > 1e-10, dm, 1e10)

    # Precompute some useful quantities
    dm_mean = np.mean(dm_safe)
    dm_std = np.std(dm_safe) + 1e-10

    best_H = None
    best_score = np.inf

    def try_candidate(H_candidate, description=""):
        nonlocal best_H, best_score, deadline, start_time
        if time.time() > deadline - 15.0:
            return

        H_candidate = np.where(np.isfinite(H_candidate) & (H_candidate > 0),
                              H_candidate, 1e-9)
        H_candidate = H_candidate.astype(np.float64)

        # Quick check: is this candidate viable?
        sim_time = min(1.5, max(0.5, (deadline - 15.0 - time.time()) / 30))
        score = _evaluate_heuristic(dm, demands, capacity, H_candidate,
                                    time_limit=sim_time, n_seeds=3)
        if score < best_score:
            best_score = score
            best_H = H_candidate.copy()
            # print(f"  New best: {description}, score={score:.4f}, time={time.time()-start_time:.1f}s")

    # ---- Candidate generation strategy ----

    # Strategy 1: Power law 1/d^gamma
    # Coarse search
    gammas_coarse = np.linspace(0.3, 4.0, 15)
    for g in gammas_coarse:
        if time.time() > deadline - 20.0:
            break
        H_cand = 1.0 / (dm_safe ** g)
        try_candidate(H_cand, f"gamma={g:.2f}")

    # Strategy 2: Exponential decay exp(-alpha * d)
    alphas_coarse = np.linspace(0.5, 5.0, 12)
    for a in alphas_coarse:
        if time.time() > deadline - 20.0:
            break
        H_cand = np.exp(-a * dm_safe)
        try_candidate(H_cand, f"exp_alpha={a:.2f}")

    # Strategy 3: Combined: 1/(d^gamma) * exp(-alpha * d)
    for g in [0.5, 1.0, 1.5, 2.0]:
        for a in [0.5, 1.0, 2.0]:
            if time.time() > deadline - 20.0:
                break
            H_cand = (1.0 / (dm_safe ** g)) * np.exp(-a * dm_safe)
            try_candidate(H_cand, f"combo g={g}, a={a}")

    # Strategy 4: Demand-aware heuristic with multiple penalty forms
    # Penalize edges to high-demand customers to pack them early
    # Form: eta[i,j] = d(i,j)^-gamma * (1 / (1 + demand_penalty * demand[j]))
    for g in [0.8, 1.0, 1.2, 1.5, 2.0]:
        for dp in [0.1, 0.3, 0.5, 1.0, 2.0]:
            if time.time() > deadline - 20.0:
                break
            H_cand = 1.0 / (dm_safe ** g)
            demand_penalty = 1.0 / (1.0 + dp * demands)
            H_cand = H_cand * demand_penalty[np.newaxis, :]
            try_candidate(H_cand, f"demand_aware g={g}, dp={dp}")

    # Strategy 5: Use coordinates directly (Euclidean-based)
    # Compute Euclidean distances from coordinates
    eucl_dm = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
    eucl_safe = np.where(eucl_dm > 1e-10, eucl_dm, 1e10)

    for g in [0.5, 1.0, 1.5, 2.0]:
        if time.time() > deadline - 20.0:
            break
        H_cand = 1.0 / (eucl_safe ** g)
        try_candidate(H_cand, f"eucl_gamma={g}")

    # Strategy 6: Normalized distance-based
    for g in [0.5, 1.0, 1.5]:
        if time.time() > deadline - 20.0:
            break
        dm_norm = (dm_safe - dm_mean) / dm_std
        dm_norm = np.where(dm_norm > 0.1, dm_norm, 0.1)
        H_cand = 1.0 / (dm_norm ** g)
        try_candidate(H_cand, f"norm_gamma={g}")

    # Strategy 7: Rank-based heuristic
    # Use ranks of distances within each row
    for beta in [1.0, 2.0, 3.0]:
        if time.time() > deadline - 20.0:
            break
        # Rank distances in each row (lower rank = shorter distance)
        row_ranks = np.zeros_like(dm_safe)
        for i in range(n):
            order = np.argsort(dm_safe[i])
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1)
            row_ranks[i] = ranks
        H_cand = 1.0 / (row_ranks ** beta)
        try_candidate(H_cand, f"rank_beta={beta}")

    # Strategy 8: Mixed distance and rank
    for g in [0.5, 1.0]:
        for beta in [1.0, 2.0]:
            if time.time() > deadline - 20.0:
                break
            H_dist = 1.0 / (dm_safe ** g)
            row_ranks = np.zeros_like(dm_safe)
            for i in range(n):
                order = np.argsort(dm_safe[i])
                ranks = np.empty(n, dtype=np.float64)
                ranks[order] = np.arange(1, n + 1)
                row_ranks[i] = ranks
            H_rank = 1.0 / (row_ranks ** beta)
            # Normalize both
            H_dist = H_dist / np.max(H_dist)
            H_rank = H_rank / np.max(H_rank)
            H_cand = H_dist * H_rank
            try_candidate(H_cand, f"mixed g={g}, beta={beta}")

    # Strategy 9: Demand-aware with spatial clustering boost
    # Boost intra-cluster edges and penalize high-demand nodes
    # Use grid-based spatial clustering
    grid_size = 4
    cell_ids = np.zeros(n, dtype=int)
    for i in range(n):
        cx = int(coords[i, 0] * grid_size)
        cy = int(coords[i, 1] * grid_size)
        cell_ids[i] = cy * grid_size + cx

    for g in [1.0, 1.5, 2.0]:
        for dp in [0.3, 0.5, 1.0]:
            if time.time() > deadline - 20.0:
                break
            H_cand = 1.0 / (dm_safe ** g)
            
            # Boost intra-cell edges
            for i in range(n):
                for j in range(n):
                    if cell_ids[i] == cell_ids[j] and i != j:
                        H_cand[i, j] *= 1.5
            
            # Demand penalty
            demand_penalty = 1.0 / (1.0 + dp * demands)
            H_cand = H_cand * demand_penalty[np.newaxis, :]
            
            try_candidate(H_cand, f"cluster_demand g={g}, dp={dp}")

    # Strategy 10: Fine-tune around the best found so far
    if best_H is not None and time.time() < deadline - 30.0:
        # Identify what type of heuristic was best and fine-tune
        # For simplicity, do a local search around power law with demand penalty
        if best_score < np.inf:
            # Try perturbations
            for g in np.linspace(0.5, 2.5, 20):
                for dp in [0.2, 0.5, 1.0]:
                    if time.time() > deadline - 25.0:
                        break
                    H_cand = 1.0 / (dm_safe ** g)
                    demand_penalty = 1.0 / (1.0 + dp * demands)
                    H_cand = H_cand * demand_penalty[np.newaxis, :]
                    try_candidate(H_cand, f"fine_demand g={g:.2f}, dp={dp}")

    # Final: if no valid candidate was found, use a sensible default
    if best_H is None:
        best_H = 1.0 / dm_safe
        best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)

    # Ensure output shape matches input
    assert best_H.shape == dm.shape, "Shape mismatch"
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)

    return best_H

# EVOLVE-BLOCK-END
