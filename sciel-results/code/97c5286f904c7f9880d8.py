"""Ant colony orienteering heuristic: model-and-optimize with local prize density."""
import numpy as np
import time

# Deadline guard: evaluator kills at 330s wall-clock; keep safety margin
_DEADLINE = time.time() + 310.0  # 20s safety margin

_params_cache = {}


def _simulate_aco(prizes, dist, max_len, heuristic, n, rng):
    """Simulate ACO and return the highest total prize collected by any ant in any iteration."""
    num_ants = 20
    num_iters = 30
    pheromone_decay = 0.9
    pheromone_gain_divisor = 10.0

    pheromone = np.ones((n, n))
    dist_to_home = dist[:, 0].copy()

    global_best = 0.0

    for it in range(num_iters):
        if time.time() > _DEADLINE - 10:
            break
        iter_best = 0.0

        for ant in range(num_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path_len = 0.0
            total_prize = 0.0

            while True:
                # Find all unvisited nodes j such that path_len + d[i,j] + d[j,0] <= max_len
                feasible = []
                probs = []
                for j in range(n):
                    if not visited[j]:
                        if path_len + dist[current, j] + dist_to_home[j] <= max_len + 1e-9:
                            p = pheromone[current, j] * heuristic[current, j]
                            if p > 0:
                                feasible.append(j)
                                probs.append(p)

                if not feasible:
                    break

                total_p = sum(probs)
                if total_p <= 0:
                    break

                # Normalize
                probs = [p / total_p for p in probs]

                # Choose next node
                r = rng.random()
                cumsum = 0.0
                chosen = feasible[-1]
                for idx, p in enumerate(probs):
                    cumsum += p
                    if r <= cumsum:
                        chosen = feasible[idx]
                        break

                path_len += dist[current, chosen]
                total_prize += prizes[chosen]
                visited[chosen] = True
                current = chosen

            if total_prize > iter_best:
                iter_best = total_prize
            if total_prize > global_best:
                global_best = total_prize

        # Pheromone update
        pheromone *= pheromone_decay
        pheromone += iter_best / pheromone_gain_divisor

    return global_best


def _make_heuristic(prizes, dist, max_len, alpha, beta, gamma, delta_p, use_local_density):
    """Construct heuristic matrix with given parameters."""
    n = len(prizes)
    eps = 1e-9

    dist_to_home = dist[:, 0].copy()
    prize_dest = prizes[None, :]  # (1, n)
    d_ij = dist + eps  # (n, n)

    return_cost_ratio = dist_to_home[None, :] / max_len  # (1, n)
    feasibility_penalty = 1.0 + return_cost_ratio ** gamma  # (1, n)

    numerator = np.maximum(prize_dest, eps) ** alpha  # (1, n)
    denominator = (d_ij ** beta) * (feasibility_penalty)  # (n, n)

    eta = numerator / denominator  # (n, n)

    if use_local_density:
        # Local prize density: for each node j, compute average prize of neighbors within remaining budget
        # This is a node-specific factor that captures marginal utility
        local_density = np.ones(n)
        for j in range(n):
            # Nodes that could be visited after j given the budget
            remaining = max_len - dist_to_home[j]
            if remaining > 0:
                neighbors = []
                for k in range(n):
                    if k != j and dist[j, k] <= remaining:
                        neighbors.append(prizes[k])
                if neighbors:
                    local_density[j] = 1.0 + np.mean(neighbors) ** delta_p
                else:
                    local_density[j] = 1.0
            else:
                local_density[j] = 1.0

        # Multiply eta by local density at destination node
        eta = eta * local_density[None, :]

    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta


def _optimize_parameters(prizes, dist, max_len, n, rng, time_budget=180.0):
    """Optimize heuristic parameters by evaluating candidate heuristics via ACO simulation."""
    start_time = time.time()
    best_score = -1.0
    best_params = (2.0, 1.5, 1.0, 1.0, True)

    # Parameter grid
    alpha_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
    beta_vals = [0.5, 1.0, 1.5, 2.0]
    gamma_vals = [0.5, 1.0, 1.5, 2.0]
    delta_p_vals = [0.5, 1.0, 1.5]
    use_ld_vals = [True, False]

    # Start with a reasonable default
    params_list = []
    for a in alpha_vals:
        for b in beta_vals:
            for g in gamma_vals:
                for dp in delta_p_vals:
                    for ld in use_ld_vals:
                        params_list.append((a, b, g, dp, ld))

    # Shuffle for diversity
    rng.shuffle(params_list)

    # First try the default
    default_params = (2.0, 1.5, 1.0, 1.0, True)
    heur = _make_heuristic(prizes, dist, max_len, *default_params)
    score = _simulate_aco(prizes, dist, max_len, heur, n, rng)
    best_score = score
    best_params = default_params

    # Evaluate other candidates
    for params in params_list:
        if time.time() > start_time + time_budget:
            break
        heur = _make_heuristic(prizes, dist, max_len, *params)
        score = _simulate_aco(prizes, dist, max_len, heur, n, rng)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for Ant Colony Optimization on the
    Orienteering Problem.
    
    Uses a model-and-optimize approach: evaluates multiple heuristic forms
    with different parameters via ACO simulation and returns the best one.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)

    n = len(prizes)

    # Check cache
    cache_key = (n, tuple(prizes.tobytes()), tuple(distance_matrix.tobytes()), max_len)
    if cache_key in _params_cache:
        best_params = _params_cache[cache_key]
    else:
        # Seed RNG deterministically based on instance
        seed = int(np.sum(prizes * 1000) + np.sum(distance_matrix * 100)) % (2**32)
        rng = np.random.RandomState(seed)

        # Optimize parameters
        best_params = _optimize_parameters(prizes, distance_matrix, max_len, n, rng,
                                           time_budget=170.0)
        _params_cache[cache_key] = best_params

    alpha, beta, gamma, delta_p, use_ld = best_params
    eta = _make_heuristic(prizes, distance_matrix, max_len, alpha, beta, gamma, delta_p, use_ld)

    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)

    return eta
