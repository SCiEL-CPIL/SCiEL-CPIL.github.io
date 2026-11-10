"""
Ant colony orienteering heuristic: focused model-and-optimize approach.
Uses a small set of well-motivated heuristic forms with optimized parameters.
"""

import numpy as np
import time

_params_cache = None
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # 20s safety margin before 330s kill


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
        iter_best = 0.0

        for ant in range(num_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path_len = 0.0
            total_prize = 0.0

            while True:
                remaining = max_len - path_len
                feasible = (~visited) & ((dist[current, :] + dist_to_home) <= remaining)
                feasible[0] = False

                if not np.any(feasible):
                    break

                candidates = np.where(feasible)[0]
                weights = pheromone[current, candidates] * heuristic[current, candidates]

                total_w = weights.sum()
                if total_w <= 0:
                    break

                probs = weights / total_w
                idx = rng.choice(len(candidates), p=probs)
                next_node = candidates[idx]

                path_len += dist[current, next_node]
                total_prize += prizes[next_node]
                visited[next_node] = True
                current = next_node

            if total_prize > iter_best:
                iter_best = total_prize

        pheromone *= pheromone_decay
        pheromone += iter_best / pheromone_gain_divisor

        if iter_best > global_best:
            global_best = iter_best

    return global_best


def _make_heuristic(prizes, dist, max_len, params):
    """
    Generate heuristic matrix with budget-aware prize density.
    params: (a, b, c, d, e)
    Uses exact maximum collectible prize from node j within remaining budget
    as a key component, combined with distance decay terms.
    """
    n = len(prizes)
    a, b, c, d, e = params
    prize = prizes[None, :]  # (1, n)
    d_ij = dist + 1e-9
    d_j0 = dist[:, 0] + 1e-9
    d_j0_row = d_j0[None, :]  # (1, n)
    
    # Compute budget-aware prize density for each node j
    # For each node j, estimate max prize collectible from j within remaining budget
    # This is approximated by looking at neighbors and their prizes within feasible range
    
    # For node j, the remaining budget after arriving from i and returning home is:
    # budget[i,j] = max_len - dist[i,j] - dist[j,0]
    # We want to estimate how much additional prize can be collected from j's neighborhood
    
    # Fast approximation: for each node j, compute a "local prize density"
    # based on nearby high-prize nodes that could be visited after j
    
    # Precompute for each node j: sum of prizes of nodes k within some radius of j
    # where dist[j,k] + dist[k,0] is feasible
    local_prize_density = np.zeros(n)
    for j in range(n):
        # Nodes k that could be visited after j (roughly, ignoring path constraints)
        # dist[j,k] + dist[k,0] <= max_len - dist[j,0] (conservative)
        remaining_after_j = max_len - dist[j, 0]
        feasible_k = (dist[j, :] + dist[:, 0]) <= remaining_after_k if False else \
                     (dist[j, :] + dist[:, 0]) <= (max_len - dist[j, 0])
        feasible_k[j] = False  # Exclude j itself
        if np.any(feasible_k):
            local_prize_density[j] = np.sum(prizes[feasible_k])
    
    # Normalize local prize density
    max_ld = np.max(local_prize_density) if np.max(local_prize_density) > 0 else 1.0
    local_prize_density_norm = local_prize_density / max_ld
    
    # Budget: max_len - d_ij - d_j0 (how much length remains after visiting j)
    budget = np.maximum(0.0, max_len - dist - d_j0_row)  # (n, n)
    budget_norm = budget / max_len  # normalized to [0, 1]
    
    # Core: prize^power / (d_ij^b * d_j0^c)
    power = a + e
    h = (prize ** power) / (d_ij ** b * d_j0_row ** c)
    
    # Budget factor: reward nodes that leave more budget
    if d > 0:
        h = h * (budget_norm + 1e-6) ** d
    
    # Budget-aware prize density factor: reward nodes that have high local prize density
    # and are feasible (budget > 0)
    if e > 0:
        # For each (i,j), use local_prize_density[j] as a multiplier
        # Only apply if the move is feasible (budget > 0)
        lp_factor = local_prize_density_norm[None, :] ** e
        h = h * lp_factor
    
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _optimize_parameters():
    """Optimize heuristic parameters on representative instances."""
    global _params_cache
    if _params_cache is not None:
        return _params_cache

    # Use global deadline

    # Generate representative instances
    n_instances = 8
    instances = []
    for seed in range(n_instances):
        rng = np.random.RandomState(seed * 7 + 3)
        n = 50
        points = rng.uniform(0, 1, size=(n, 2))
        prizes = rng.uniform(0, 1, size=n)
        prizes[0] = 0.0
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        instances.append((prizes, dist, 3.0))

    best_params = (1.0, 1.0, 0.5, 0.0, 0.0)
    best_score = -1.0

    # Parameter grid: (a, b, c, d, e)
    # a: prize exponent base
    # b: inverse distance exponent
    # c: inverse distance-to-home exponent
    # d: budget factor exponent
    # e: local prize density exponent
    param_grid = []
    
    # Try various combinations
    for a in [0.5, 1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for c in [0.0, 0.5, 1.0, 1.5]:
                for d in [0.0, 0.5, 1.0]:
                    for e in [0.0, 0.5, 1.0]:
                        param_grid.append((a, b, c, d, e))
    
    # Evaluate candidates
    eval_rng = np.random.RandomState(42)
    
    for params in param_grid:
        if time.time() > _DEADLINE - 15.0:
            break

        total_score = 0.0
        for prizes, dist, max_len in instances:
            n = len(prizes)
            heuristic = _make_heuristic(prizes, dist, max_len, params)
            score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
            total_score += score

        mean_score = total_score / len(instances)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    _params_cache = best_params
    return _params_cache


def heuristics(prizes, distance_matrix, max_len):
    """Compute the heuristic matrix for ant colony optimization."""
    params = _optimize_parameters()
    heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)
    return heuristic
