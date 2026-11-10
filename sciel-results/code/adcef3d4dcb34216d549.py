"""Ant colony orienteering heuristic: regret-based flexibility + model-and-optimize."""
import numpy as np
import time

_DEADLINE = time.time() + 310.0  # 20s safety margin before 330s kill
_params_cache = None


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
        if time.time() > _DEADLINE - 15:
            break
        iter_best = 0.0

        for ant in range(num_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path_len = 0.0
            total_prize = 0.0

            while True:
                remaining = max_len - path_len
                feasible_mask = (~visited) & ((dist[current, :] + dist_to_home) <= remaining + 1e-9)
                feasible_mask[0] = False

                if not np.any(feasible_mask):
                    break

                candidates = np.where(feasible_mask)[0]
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


def _compute_regret_matrix(prizes, dist, max_len, n):
    """
    Compute regret-based flexibility for each node j.
    For node j, count how many other nodes k are reachable from j
    while still being able to return home within max_len.
    H_regret[i,j] is inversely proportional to this count (more flexibility = lower regret).
    """
    dist_to_home = dist[:, 0]
    regret = np.ones(n)
    
    for j in range(n):
        # After visiting j, remaining budget is max_len - dist[j,0] (conservative: ignore where we came from)
        remaining_after_j = max_len - dist_to_home[j]
        if remaining_after_j <= 0:
            regret[j] = n  # No flexibility at all
            continue
        
        # Count nodes k (k != j) such that dist[j,k] + dist[k,0] <= remaining_after_j
        # This means from j we can go to k and still return home
        feasible_from_j = (dist[j, :] + dist_to_home) <= remaining_after_j + 1e-9
        feasible_from_j[j] = False
        count = np.sum(feasible_from_j)
        # Inverse: more options = better = lower regret
        # Use (count + 1) to avoid division by zero
        regret[j] = (n - 1) / (count + 1)
    
    # Normalize to [1, n] range for stability
    regret_min = np.min(regret)
    regret_max = np.max(regret)
    if regret_max > regret_min:
        regret = 1.0 + (regret - regret_min) / (regret_max - regret_min) * (n - 1)
    else:
        regret = np.ones(n)
    
    return regret


def _make_heuristic(prizes, dist, max_len, params):
    """
    Generate heuristic matrix combining prize/distance with regret-based flexibility.
    params: (a, b, c, d, e, f)
    a: prize exponent
    b: inverse distance exponent  
    c: inverse distance-to-home exponent
    d: budget factor exponent
    e: regret flexibility exponent (higher = more weight on flexibility)
    f: local prize density exponent
    """
    n = len(prizes)
    a, b, c, d, e, f = params
    prize = prizes[None, :]  # (1, n)
    d_ij = dist + 1e-9
    d_j0 = dist[:, 0] + 1e-9
    d_j0_row = d_j0[None, :]  # (1, n)
    
    # Regret-based flexibility
    regret = _compute_regret_matrix(prizes, dist, max_len, n)
    
    # Local prize density
    local_prize_density = np.zeros(n)
    for j in range(n):
        remaining_after_j = max_len - dist[j, 0]
        if remaining_after_j > 0:
            feasible_k = (dist[j, :] + dist[:, 0]) <= remaining_after_j + 1e-9
            feasible_k[j] = False
            if np.any(feasible_k):
                local_prize_density[j] = np.sum(prizes[feasible_k])
    
    max_ld = np.max(local_prize_density) if np.max(local_prize_density) > 0 else 1.0
    local_prize_density_norm = local_prize_density / max_ld
    
    # Budget: max_len - d_ij - d_j0
    budget = np.maximum(0.0, max_len - dist - d_j0_row)
    budget_norm = budget / max_len
    
    # Core: prize^a / (d_ij^b * d_j0^c)
    h = (prize ** a) / (d_ij ** b * d_j0_row ** c)
    
    # Budget factor
    if d > 0:
        h = h * (budget_norm + 1e-6) ** d
    
    # Regret flexibility factor: reward nodes with high flexibility (low regret)
    # regret is in [1, n], higher = worse. We want to reward low regret.
    # Use (n+1 - regret) / n to get [0,1] where 1 = best flexibility
    if e > 0:
        flexibility = (n + 1 - regret) / n  # (n,)
        flexibility = np.maximum(flexibility, 1e-6)
        h = h * flexibility[None, :] ** e
    
    # Local prize density factor
    if f > 0:
        lp_factor = local_prize_density_norm[None, :] ** f
        h = h * lp_factor
    
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _optimize_parameters():
    """Optimize heuristic parameters on representative instances."""
    global _params_cache
    if _params_cache is not None:
        return _params_cache

    # Generate representative instances
    n_instances = 6
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

    best_params = (1.5, 1.5, 0.5, 0.5, 1.0, 0.5)
    best_score = -1.0

    # Parameter grid: (a, b, c, d, e, f)
    param_grid = []
    for a in [1.0, 1.5, 2.0]:
        for b in [1.0, 1.5, 2.0]:
            for c in [0.5, 1.0]:
                for d in [0.5, 1.0]:
                    for e in [0.5, 1.0, 1.5]:
                        for f in [0.0, 0.5, 1.0]:
                            param_grid.append((a, b, c, d, e, f))
    
    # Shuffle
    rng_grid = np.random.RandomState(42)
    rng_grid.shuffle(param_grid)
    
    eval_rng = np.random.RandomState(42)
    
    for params in param_grid:
        if time.time() > _DEADLINE - 20:
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
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Cache key based on instance identity
    cache_key = (n, tuple(prizes.tobytes()), tuple(distance_matrix.tobytes()), max_len)
    
    if _params_cache is None:
        params = _optimize_parameters()
    else:
        params = _params_cache
    
    heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
