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
                # Vectorized feasibility check
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
    Generate heuristic matrix.
    params: (a, b, c, d, e)
    h[i,j] = prize[j]^a * (1/d_ij)^b * (1/(d_j0+eps))^c * (budget_norm)^d * (prize[j])^e
    Simplified: h[i,j] = prize[j]^(a+e) / (d_ij^b * (d_j0+eps)^c) * budget_norm^d
    """
    n = len(prizes)
    a, b, c, d, e = params
    prize = prizes[None, :]  # (1, n)
    d_ij = dist + 1e-9
    d_j0 = dist[:, 0] + 1e-9
    d_j0_row = d_j0[None, :]  # (1, n)
    
    # budget: max_len - d_ij - d_j0 (how much length remains after visiting j)
    budget = np.maximum(0.0, max_len - dist - d_j0_row)  # (n, n)
    budget_norm = budget / max_len  # normalized to [0, 1]
    
    # Core: prize^power / (d_ij^b * d_j0^c)
    power = a + e
    h = (prize ** power) / (d_ij ** b * d_j0_row ** c)
    
    # Budget factor: reward nodes that leave more budget
    if d > 0:
        h = h * (budget_norm + 1e-6) ** d
    
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _optimize_parameters():
    """Optimize heuristic parameters on representative instances."""
    global _params_cache
    if _params_cache is not None:
        return _params_cache

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
    # e: additional prize exponent
    param_grid = []
    
    # Broader and finer grid
    a_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    b_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    c_vals = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    d_vals = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    e_vals = [0.0, 0.25, 0.5, 0.75]
    
    for a in a_vals:
        for b in b_vals:
            for c in c_vals:
                for d in d_vals:
                    for e in e_vals:
                        param_grid.append((a, b, c, d, e))
    
    # Evaluate candidates
    eval_rng = np.random.RandomState(42)
    
    for params in param_grid:
        if time.time() > _DEADLINE - 20.0:
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
    
    # Safety: ensure valid output
    n = len(prizes)
    if heuristic.shape != (n, n):
        heuristic = np.ones((n, n)) * 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
