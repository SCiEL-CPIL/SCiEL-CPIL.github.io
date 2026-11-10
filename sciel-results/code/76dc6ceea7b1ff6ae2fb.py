import numpy as np
import time
from scipy.optimize import minimize

_params_cache = None


def _knapsack_max_prize(prizes, dist_from_i, dist_to_home, remaining_budget, n):
    """
    Fast greedy knapsack approximation: estimate max prize collectible
    starting from node i with remaining budget, then returning to home.
    
    We greedily pick nodes with best prize/(travel_cost) ratio that fit.
    This is an upper-bound-ish estimate used for heuristic shaping.
    """
    # For each candidate node j, the cost to go i->j->home is dist_from_i[j] + dist_to_home[j]
    # But in a path, we visit multiple nodes. Approximate: each node j has an effective
    # "incremental cost" and we pick greedily by prize/cost.
    
    # Simplified: consider each node j as an item with:
    #   value = prizes[j]
    #   cost  = dist_from_i[j] + dist_to_home[j]  (if it were the only node)
    # But that overcounts. Better approximation:
    #   For a sequence, total cost ≈ sum of edge costs + dist_to_home[last]
    # Greedy: sort by prizes[j] / (dist_from_i[j] + dist_to_home[j]) descending,
    # pick while cumulative "budget" allows.
    
    # Even faster: just compute sum of top-k prizes where k is max nodes fit.
    # Use a simple greedy: repeatedly pick the node with highest prize among those
    # that are "reachable" (dist_from_i[j] + dist_to_home[j] <= remaining_budget)
    # and not yet picked, subtracting a rough cost.
    
    # For speed in heuristic computation, use a vectorized greedy approximation:
    # 1. Filter nodes where dist_from_i[j] + dist_to_home[j] <= remaining_budget
    # 2. Among those, greedily pick by prize/(dist_from_i[j] + dist_to_home[j])
    #    but approximate with just top prizes among feasible, capped by count.
    
    # Fastest approximation: 
    #   feasible_mask = (dist_from_i + dist_to_home) <= remaining_budget
    #   Among feasible, the max prize is sum of top min(k, num_feasible) prizes
    #   where k ≈ remaining_budget / avg_step (rough)
    # Even simpler: just sum all feasible prizes as an upper bound, or top few.
    
    # Let's do: sum of top 3 prizes among feasible nodes (since max_len=3, few nodes fit)
    feasible = (dist_from_i + dist_to_home) <= remaining_budget
    if not np.any(feasible):
        return 0.0
    feasible_prizes = prizes[feasible]
    if len(feasible_prizes) <= 3:
        return feasible_prizes.sum()
    # Top 3
    top3 = np.sort(feasible_prizes)[-3:]
    return top3.sum()


def _simulate_aco(prizes, dist, max_len, heuristic, n, rng):
    """Simulate ACO with precomputed structures for speed."""
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
                # Feasible nodes: not visited, and path_len + d[i,j] + d[j,0] <= max_len
                # d[i,j] + d[j,0] <= remaining
                feasible_mask = (~visited) & ((dist[current, :] + dist_to_home) <= remaining)
                feasible_mask[0] = False

                if not np.any(feasible_mask):
                    break

                candidates = np.where(feasible_mask)[0]
                # Weights: pheromone * heuristic
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
    Build heuristic matrix from structural task features.
    params: (form, p1, p2, p3, p4, p5)
    """
    n = len(prizes)
    form = int(params[0])
    p1, p2, p3, p4, p5 = params[1], params[2], params[3], params[4], params[5]

    # Ensure parameters are positive where required to avoid division by zero or invalid powers
    p1 = max(p1, 0.1)
    p2 = max(p2, 0.1)
    p3 = max(p3, 0.0)
    p4 = max(p4, 0.001)
    p5 = max(p5, 0.1)

    prize = prizes[None, :]  # (1, n)
    d_ij = dist + 1e-10       # (n, n)
    d_j0 = dist[:, 0] + 1e-10 # (n,)
    d_j0_row = d_j0[None, :]  # (1, n)

    try:
        if form == 0:
            # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)^p5
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** p1) / (cost ** p5)
        elif form == 1:
            # prize^p1 / (d_ij^p2 * (1 + d_j0^p3) + p4)^p5
            cost = d_ij ** p2 * (1.0 + d_j0_row ** p3) + p4
            h = (prize ** p1) / (cost ** p5)
        elif form == 2:
            # prize^p1 * (1 / (d_ij + p4))^p2 * (1 / (d_j0 + p4))^p3
            h = (prize ** p1) * ((1.0 / (d_ij + p4)) ** p2) * ((1.0 / (d_j0_row + p4)) ** p3)
        elif form == 3:
            # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** p1) / cost
        elif form == 4:
            # prize^p1 * exp(-p2 * d_ij) * exp(-p3 * d_j0)
            h = (prize ** p1) * np.exp(-p2 * d_ij) * np.exp(-p3 * d_j0_row)
        elif form == 5:
            # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)^p5 * (1 + p3 * d_j0)
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** p1) / (cost ** p5) * (1.0 + p3 * d_j0_row)
        elif form == 6:
            # prize^p1 / ((d_ij + d_j0)^p2 + p4)^p3
            total_cost = d_ij + d_j0_row
            h = (prize ** p1) / ((total_cost ** p2 + p4) ** p3)
        elif form == 7:
            # prize^p1 * (1 + p2) / (d_ij^p2 + d_j0^p3 + p4)^p3
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** p1) * (1.0 + p2) / (cost ** p3)
        elif form == 8:
            # prize^p1 / (d_ij + d_j0 + p4)^p2
            total_cost = d_ij + d_j0_row + p4
            h = (prize ** p1) / (total_cost ** p2)
        elif form == 9:
            # prize / (d_ij * d_j0 + p4)
            h = prize / (d_ij * d_j0_row + p4)
        elif form == 10:
            # (prize * (1 + p4)) / (d_ij^p2 + d_j0^p3 + p5)
            cost = d_ij ** p2 + d_j0_row ** p3 + p5
            h = (prize * (1.0 + p4)) / cost
        elif form == 11:
            # Budgeted prize density: for each (i,j), estimate max prize collectible
            # from j given the remaining budget after moving to j from i.
            # H[i,j] = prize[j] * (1 + p2 * knapsack_estimate_j) / (d_ij^p1 + p3)
            # where knapsack_estimate_j is the additional prize available from j
            # within the remaining budget (max_len - d[i,j] - d[j,0]).
            # This makes the heuristic constraint-aware: nodes that open up
            # high-prize neighborhoods get boosted.
            h = np.ones((n, n))
            for j in range(1, n):
                # For each i, the remaining budget after going i->j->home
                rem_budget = max_len - d_ij[:, j] - d_j0[j]
                # Knapsack estimate from node j with rem_budget
                # dist from j to all nodes: dist[j, :]
                # dist to home from all nodes: d_j0
                feasible_from_j = (dist[j, :] + d_j0) <= rem_budget
                feasible_from_j[j] = False  # can't revisit j
                if not np.any(feasible_from_j):
                    knap_est = 0.0
                else:
                    fp = prizes[feasible_from_j]
                    if len(fp) <= 3:
                        knap_est = fp.sum()
                    else:
                        knap_est = np.sort(fp)[-3:].sum()
                # H[i,j] = prize[j] * (1 + p2 * knap_est) / (d_ij[i,j]^p1 + p3)
                h[:, j] = (prizes[j] * (1.0 + p2 * knap_est)) / (d_ij[:, j] ** p1 + p3)
            h[0, :] = 1e-9  # row 0 less important but keep valid
        else:
            # Fallback to Form 0
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** p1) / (cost ** p5)
    except Exception:
        h = np.ones((n, n)) * 1e-9

    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _optimize_parameters():
    """Optimize heuristic parameters on representative instances."""
    global _params_cache
    if _params_cache is not None:
        return _params_cache

    start_time = time.time()
    deadline = start_time + 290.0  # Leave 40s margin for safety

    # Generate representative instances
    n_instances = 10
    instances = []
    for seed in range(n_instances):
        rng = np.random.RandomState(seed)
        n = 50
        points = rng.uniform(0, 1, size=(n, 2))
        prizes = rng.uniform(0, 1, size=n)
        prizes[0] = 0.0
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        instances.append((prizes, dist, 3.0))

    best_params = (0, 1.0, 1.0, 0.5, 1.0, 0.01)
    best_score = -1.0

    eval_rng = np.random.RandomState(42)

    # Define parameter grids per form
    # Form 0: p1, p2, p3, p4, p5
    # Form 1: p1, p2, p3, p4, p5
    # Form 2: p1, p2, p3, p4, _
    # Form 3: p1, p2, p3, p4, _
    # Form 4: p1, p2, p3, _, _
    # Form 5: p1, p2, p3, p4, p5
    # Form 6: p1, p2, p3, p4, _
    # Form 7: p1, p2, p3, p4, _
    # Form 8: p1, p2, _, p4, _
    # Form 9: _, _, _, p4, _
    # Form 10: _, p2, p3, p4, p5

    coarse_grids = {
        0: [
            (1.0, 2.0, 0.5, 1.0, 0.01),
            (1.5, 2.0, 1.0, 1.0, 0.01),
            (2.0, 3.0, 1.0, 1.0, 0.01),
            (1.0, 2.5, 1.5, 1.0, 0.01),
            (1.5, 3.0, 1.0, 1.0, 0.05),
        ],
        11: [
            (1.0, 0.5, 0.01, 0.0, 0.0),
            (1.5, 1.0, 0.01, 0.0, 0.0),
            (2.0, 1.0, 0.05, 0.0, 0.0),
            (1.0, 2.0, 0.01, 0.0, 0.0),
            (1.5, 1.5, 0.05, 0.0, 0.0),
            (2.0, 2.0, 0.01, 0.0, 0.0),
        ],
        1: [
            (1.0, 2.0, 0.5, 0.01, 1.0),
            (1.5, 2.0, 1.0, 0.01, 1.0),
            (2.0, 2.5, 1.0, 0.01, 1.0),
            (1.0, 2.0, 1.0, 0.05, 1.5),
        ],
        2: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.5, 0.01, 0.0),
        ],
        3: [
            (1.0, 2.0, 0.5, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.0, 0.01, 0.0),
        ],
        4: [
            (1.0, 2.0, 1.0, 0.0, 0.0),
            (1.5, 2.0, 1.0, 0.0, 0.0),
            (2.0, 2.5, 1.0, 0.0, 0.0),
            (1.0, 3.0, 1.5, 0.0, 0.0),
        ],
        5: [
            (1.0, 2.0, 0.5, 1.0, 0.01),
            (1.5, 2.0, 1.0, 1.0, 0.01),
            (2.0, 2.5, 1.0, 1.0, 0.01),
            (1.0, 3.0, 1.0, 1.0, 0.01),
        ],
        6: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.0, 0.01, 0.0),
        ],
        7: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.0, 0.01, 0.0),
        ],
        8: [
            (1.0, 2.0, 0.0, 0.01, 0.0),
            (1.5, 2.5, 0.0, 0.01, 0.0),
            (2.0, 3.0, 0.0, 0.01, 0.0),
            (1.0, 1.5, 0.0, 0.01, 0.0),
        ],
        9: [
            (1.0, 1.0, 1.0, 0.01, 0.0),
            (1.0, 1.0, 1.0, 0.05, 0.0),
            (1.0, 1.0, 1.0, 0.1, 0.0),
        ],
        10: [
            (1.0, 2.0, 1.0, 0.01, 0.01),
            (1.0, 2.5, 1.0, 0.01, 0.01),
            (1.0, 2.0, 1.5, 0.01, 0.01),
        ],
    }

    # Phase 1: Coarse search
    for form in range(12):
        if time.time() > deadline - 30.0:
            break
        for ps in coarse_grids[form]:
            if time.time() > deadline - 30.0:
                break
            params = (form,) + ps
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

    # Phase 2: Local Search (Nelder-Mead)
    # We optimize the continuous parameters p1..p5 for the best form found.
    # Note: Some forms don't use all params, but we can still pass them.
    
    if time.time() < deadline - 10.0:
        best_form = best_params[0]
        init_p = best_params[1:]
        
        def objective(p_cont):
            # Clip parameters to valid ranges
            p1 = max(0.1, min(5.0, p_cont[0]))
            p2 = max(0.1, min(5.0, p_cont[1]))
            p3 = max(0.0, min(5.0, p_cont[2]))
            p4 = max(0.001, min(1.0, p_cont[3]))
            p5 = max(0.1, min(5.0, p_cont[4]))
            
            params = (best_form, p1, p2, p3, p4, p5)
            total_score = 0.0
            for prizes, dist, max_len in instances:
                n = len(prizes)
                heuristic = _make_heuristic(prizes, dist, max_len, params)
                score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
                total_score += score
            return -total_score / len(instances) # Minimize negative score

        try:
            res = minimize(objective, init_p, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3})
            if res.success or res.fun < -best_score:
                new_p = res.x
                new_params = (best_form, 
                              max(0.1, new_p[0]), max(0.1, new_p[1]), 
                              max(0.0, new_p[2]), max(0.001, new_p[3]), 
                              max(0.1, new_p[4]))
                # Verify improvement
                total_score = 0.0
                for prizes, dist, max_len in instances:
                    n = len(prizes)
                    heuristic = _make_heuristic(prizes, dist, max_len, new_params)
                    score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
                    total_score += score
                mean_score = total_score / len(instances)
                if mean_score > best_score:
                    best_params = new_params
                    best_score = mean_score
        except Exception:
            pass

    _params_cache = best_params
    return _params_cache


def heuristics(prizes, distance_matrix, max_len):
    """Compute the heuristic matrix for ant colony optimization."""
    params = _optimize_parameters()
    heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)
    return heuristic
