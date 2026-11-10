import numpy as np
import time

_params_cache = None


def _simulate_aco_fast(prizes, dist, max_len, heuristic, n, rng):
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
            # Use a deterministic sub-seed per (it, ant) for reproducibility
            sub_rng = np.random.RandomState(it * 1000 + ant)
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
                idx = sub_rng.choice(len(candidates), p=probs)
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
    Supports multiple formula forms for richer search.
    """
    n = len(prizes)
    form = params[0]
    p1, p2, p3, p4, p5 = params[1], params[2], params[3], params[4], params[5]

    prize = prizes[None, :]  # (1, n)
    d_ij = dist + 1e-10       # (n, n)
    d_j0 = dist[:, 0] + 1e-10 # (n,)
    d_j0_row = d_j0[None, :]  # (1, n)

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
        # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)  (no outer exponent)
        cost = d_ij ** p2 + d_j0_row ** p3 + p4
        h = (prize ** p1) / cost
    elif form == 4:
        # prize^p1 * exp(-p2 * d_ij) * exp(-p3 * d_j0)
        h = (prize ** p1) * np.exp(-p2 * d_ij) * np.exp(-p3 * d_j0_row)
    elif form == 5:
        # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)^p5 * (1 + p3 * d_j0)  -- bonus for near-origin
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
        # prize^p1 * exp(-p2 * d_ij) / (d_j0^p3 + p4)^p5
        # Separates distance-to-next from distance-to-home with different treatments
        h = (prize ** p1) * np.exp(-p2 * d_ij) / ((d_j0_row ** p3 + p4) ** p5)
    elif form == 9:
        # prize^p1 / ((d_ij + d_j0)^p2 + p4) * (1 / (d_j0 + p4))^p3
        # Total path cost with extra penalty for far-from-home nodes
        total_cost = d_ij + d_j0_row
        h = (prize ** p1) / ((total_cost ** p2 + p4)) * ((1.0 / (d_j0_row + p4)) ** p3)
    elif form == 10:
        # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)^p5 * (1 + p5 * prize[j])
        # Bonus for high-prize destinations
        cost = d_ij ** p2 + d_j0_row ** p3 + p4
        h = (prize ** p1) / (cost ** p5) * (1.0 + p5 * prize)
    elif form == 11:
        # prize^p1 * (1 / (d_ij + p4))^p2 * (1 / (d_j0 + p4))^p3 * (1 + p5 * (1 - d_j0))
        # Near-origin bonus
        h = (prize ** p1) * ((1.0 / (d_ij + p4)) ** p2) * ((1.0 / (d_j0_row + p4)) ** p3) * (1.0 + p5 * (1.0 - d_j0_row))
    elif form == 12:
        # prize^p1 / (d_ij^p2 * d_j0^p3 + p4)^p5
        # Product of distances
        cost = (d_ij ** p2) * (d_j0_row ** p3) + p4
        h = (prize ** p1) / (cost ** p5)
    elif form == 13:
        # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)^p5 * exp(-p5 * d_ij)
        # Extra exponential decay on distance-to-next
        cost = d_ij ** p2 + d_j0_row ** p3 + p4
        h = (prize ** p1) / (cost ** p5) * np.exp(-p5 * d_ij)
    else:
        # Fallback: form 0
        cost = d_ij ** p2 + d_j0_row ** p3 + p4
        h = (prize ** p1) / (cost ** p5)

    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _optimize_parameters():
    """Optimize heuristic parameters on representative instances with multiple forms."""
    global _params_cache
    if _params_cache is not None:
        return _params_cache

    start_time = time.time()
    deadline = start_time + 295.0  # Leave 35s margin for safety

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
    coarse_grids = {
        0: [
            (1.0, 2.0, 0.5, 1.0, 0.01),
            (1.0, 3.0, 1.0, 1.0, 0.01),
            (1.5, 2.0, 1.0, 1.0, 0.01),
            (2.0, 2.0, 1.0, 1.0, 0.01),
            (1.0, 2.5, 1.5, 1.0, 0.01),
            (1.5, 3.0, 1.0, 1.0, 0.05),
            (2.0, 3.0, 1.5, 1.0, 0.01),
            (1.0, 2.0, 0.5, 1.5, 0.01),
            (1.5, 2.5, 1.0, 1.5, 0.01),
            (1.2, 2.2, 0.8, 1.2, 0.02),
        ],
        1: [
            (1.0, 2.0, 0.5, 0.01, 1.0),
            (1.0, 3.0, 1.0, 0.01, 1.0),
            (1.5, 2.0, 1.0, 0.01, 1.0),
            (2.0, 2.5, 1.0, 0.01, 1.0),
            (1.0, 2.0, 1.0, 0.05, 1.5),
            (1.2, 2.5, 0.8, 0.02, 1.2),
        ],
        2: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.5, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 2.0, 0.5, 0.05, 0.0),
            (1.2, 2.2, 1.2, 0.02, 0.0),
        ],
        3: [
            (1.0, 2.0, 0.5, 0.01, 0.0),
            (1.0, 3.0, 1.0, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 2.0, 1.5, 0.05, 0.0),
            (1.2, 2.2, 0.8, 0.02, 0.0),
        ],
        4: [
            (1.0, 2.0, 1.0, 0.0, 0.0),
            (1.0, 3.0, 1.5, 0.0, 0.0),
            (1.5, 2.0, 1.0, 0.0, 0.0),
            (2.0, 2.5, 1.0, 0.0, 0.0),
            (1.0, 2.5, 1.0, 0.0, 0.0),
            (1.2, 2.2, 1.2, 0.0, 0.0),
        ],
        5: [
            (1.0, 2.0, 0.5, 1.0, 0.01),
            (1.0, 3.0, 1.0, 1.0, 0.01),
            (1.5, 2.0, 1.0, 1.0, 0.01),
            (2.0, 2.5, 1.0, 1.0, 0.01),
            (1.0, 2.0, 1.5, 1.5, 0.01),
            (1.2, 2.2, 0.8, 1.2, 0.02),
        ],
        6: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.0, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 2.0, 1.5, 0.05, 0.0),
            (1.2, 2.2, 1.2, 0.02, 0.0),
        ],
        7: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.0, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 2.0, 1.5, 0.05, 0.0),
            (1.2, 2.2, 1.2, 0.02, 0.0),
        ],
        8: [
            (1.0, 2.0, 1.0, 0.01, 1.0),
            (1.0, 3.0, 1.0, 0.01, 1.0),
            (1.5, 2.0, 1.0, 0.01, 1.0),
            (2.0, 2.5, 1.0, 0.01, 1.0),
            (1.0, 2.0, 1.5, 0.05, 1.0),
            (1.0, 2.5, 1.0, 0.01, 1.5),
            (1.5, 3.0, 1.0, 0.01, 1.0),
            (2.0, 2.0, 1.5, 0.01, 1.0),
            (1.2, 2.2, 1.2, 0.02, 1.2),
        ],
        9: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.5, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 2.0, 2.0, 0.05, 0.0),
            (1.5, 3.0, 1.5, 0.01, 0.0),
            (2.0, 2.0, 1.0, 0.01, 0.0),
            (1.0, 2.5, 1.0, 0.01, 0.0),
            (1.2, 2.2, 1.2, 0.02, 0.0),
        ],
        10: [
            (1.0, 2.0, 0.5, 1.0, 0.01),
            (1.5, 2.0, 1.0, 1.0, 0.01),
            (2.0, 2.5, 1.0, 1.0, 0.01),
            (1.0, 2.0, 1.0, 1.0, 0.05),
            (1.2, 2.2, 0.8, 1.2, 0.02),
        ],
        11: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 2.0, 0.5, 0.05, 0.0),
            (1.2, 2.2, 1.2, 0.02, 0.0),
        ],
        12: [
            (1.0, 1.5, 1.0, 0.01, 1.0),
            (1.0, 2.0, 1.0, 0.01, 1.0),
            (1.5, 1.5, 1.0, 0.01, 1.0),
            (2.0, 2.0, 1.0, 0.01, 1.0),
            (1.2, 1.8, 1.2, 0.02, 1.0),
        ],
        13: [
            (1.0, 2.0, 0.5, 1.0, 0.01),
            (1.5, 2.0, 1.0, 1.0, 0.01),
            (2.0, 2.5, 1.0, 1.0, 0.01),
            (1.0, 2.0, 1.0, 1.0, 0.05),
            (1.2, 2.2, 0.8, 1.2, 0.02),
        ],
    }

    # Phase 1: Coarse search across all forms
    for form in range(14):
        if time.time() > deadline - 25.0:
            break
        for ps in coarse_grids[form]:
            if time.time() > deadline - 25.0:
                break
            params = (form,) + ps
            total_score = 0.0
            for prizes, dist, max_len in instances:
                n = len(prizes)
                heuristic = _make_heuristic(prizes, dist, max_len, params)
                score = _simulate_aco_fast(prizes, dist, max_len, heuristic, n, eval_rng)
                total_score += score
            mean_score = total_score / len(instances)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

    # Phase 2: Fine-tune around best
    if time.time() < deadline - 30.0:
        form0 = best_params[0]
        p1_0, p2_0, p3_0, p4_0, p5_0 = best_params[1], best_params[2], best_params[3], best_params[4], best_params[5]

        # Generate fine-tuned candidates around the best
        fine_candidates = []
        steps = [0.1, 0.2, 0.4]
        for s in steps:
            for da in [-s, -s*0.5, 0, s*0.5, s]:
                for db in [-s, -s*0.5, 0, s*0.5, s]:
                    for dc in [-s, -s*0.5, 0, s*0.5, s]:
                        for dd in [-s*0.5, 0, s*0.5]:
                            for de in [-s*0.5, 0, s*0.5]:
                                p1 = max(0.1, p1_0 + da)
                                p2 = max(0.1, p2_0 + db)
                                p3 = max(0.0, p3_0 + dc)
                                p4 = max(0.001, p4_0 + dd)
                                p5 = max(0.0, p5_0 + de)
                                fine_candidates.append((form0, p1, p2, p3, p4, p5))

        # Also try neighboring forms
        for nf in range(14):
            if nf == form0:
                continue
            for ps in coarse_grids[nf][:5]:  # top 5 of each form
                fine_candidates.append((nf,) + ps)

        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in fine_candidates:
            key = (c[0], round(c[1],3), round(c[2],3), round(c[3],3), round(c[4],3), round(c[5],3))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        # Evaluate all candidates, tracking best
        for params in unique_candidates:
            if time.time() > deadline - 25.0:
                break
            total_score = 0.0
            for prizes, dist, max_len in instances:
                n = len(prizes)
                heuristic = _make_heuristic(prizes, dist, max_len, params)
                score = _simulate_aco_fast(prizes, dist, max_len, heuristic, n, eval_rng)
                total_score += score
            mean_score = total_score / len(instances)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

    # Phase 3: High-precision polish around the best found
    if time.time() < deadline - 15.0:
        form0 = best_params[0]
        p1_0, p2_0, p3_0, p4_0, p5_0 = best_params[1], best_params[2], best_params[3], best_params[4], best_params[5]

        polish_candidates = []
        fine_steps = [0.03, 0.06, 0.1]
        for s in fine_steps:
            for da in [-s, 0, s]:
                for db in [-s, 0, s]:
                    for dc in [-s, 0, s]:
                        for dd in [-s*0.5, 0, s*0.5]:
                            for de in [-s*0.5, 0, s*0.5]:
                                p1 = max(0.1, p1_0 + da)
                                p2 = max(0.1, p2_0 + db)
                                p3 = max(0.0, p3_0 + dc)
                                p4 = max(0.001, p4_0 + dd)
                                p5 = max(0.0, p5_0 + de)
                                polish_candidates.append((form0, p1, p2, p3, p4, p5))

        seen = set()
        unique_polish = []
        for c in polish_candidates:
            key = (c[0], round(c[1],4), round(c[2],4), round(c[3],4), round(c[4],4), round(c[5],4))
            if key not in seen:
                seen.add(key)
                unique_polish.append(c)

        for params in unique_polish:
            if time.time() > deadline - 10.0:
                break
            total_score = 0.0
            for prizes, dist, max_len in instances:
                n = len(prizes)
                heuristic = _make_heuristic(prizes, dist, max_len, params)
                score = _simulate_aco_fast(prizes, dist, max_len, heuristic, n, eval_rng)
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


# Keep the old function name for compatibility if referenced elsewhere
_simulate_aco = _simulate_aco_fast
