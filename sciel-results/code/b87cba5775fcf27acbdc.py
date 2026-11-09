"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_aco(prizes, weights, heuristic, n_ants=20, n_iters=30, seed=0, capacity=25.0):
    """Run the fixed ACO and return the best total prize found."""
    n_items = prizes.shape[1]
    rng = np.random.default_rng(seed)
    pheromone = np.ones(n_items)
    best_total = 0.0

    for _ in range(n_iters):
        iter_best = 0.0
        for _ in range(n_ants):
            remaining = np.full(prizes.shape[0], capacity)
            used = np.zeros(n_items, dtype=bool)
            total = 0.0
            while True:
                # find items that still fit
                fits = np.ones(n_items, dtype=bool)
                for c in range(prizes.shape[0]):
                    fits &= (weights[c] <= remaining[c] + 1e-12)
                fits &= ~used
                if not fits.any():
                    break
                cand = np.where(fits)[0]
                scores = pheromone[cand] * heuristic[cand]
                scores = np.maximum(scores, 1e-300)
                total_p = scores.sum()
                if total_p <= 0:
                    break
                probs = scores / total_p
                idx = rng.choice(cand, p=probs)
                for c in range(prizes.shape[0]):
                    remaining[c] -= weights[c, idx]
                used[idx] = True
                total += prizes.sum(axis=0)[idx]
            if total > iter_best:
                iter_best = total
        if iter_best > best_total:
            best_total = iter_best
        # pheromone update
        pheromone *= 0.9
        pheromone += iter_best / 10.0

    return best_total


def _candidate_heuristics(prizes, weights):
    """Generate a diverse set of candidate heuristic vectors."""
    n_items = weights.shape[1]
    n_cons = weights.shape[0]
    w_sum = weights.sum(axis=0)
    w_max = weights.max(axis=0)
    p_sum = prizes.sum(axis=0)
    p_max = prizes.max(axis=0)

    cands = []

    # 1. prize / weight_sum
    cands.append(p_sum / (w_sum + 1e-9))

    # 2. prize / weight_max
    cands.append(p_sum / (w_max + 1e-9))

    # 3. prize only
    cands.append(p_sum)

    # 4. prize / (weighted avg of constraints, emphasizing tight ones)
    # tightness: how close the average weight per constraint is to capacity
    avg_w = weights.mean(axis=1)
    tightness = 1.0 / (avg_w + 1e-9)
    # weight each constraint by its tightness
    w_weighted = (weights * tightness[:, None]).sum(axis=0)
    cands.append(p_sum / (w_weighted + 1e-9))

    # 5. geometric mean of per-constraint ratios
    # ratio_c = prize_c / weight_c, then geo mean
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(weights > 1e-9, prizes / np.maximum(weights, 1e-9), 0.0)
    geo = np.exp(np.mean(np.log(np.maximum(ratios, 1e-9)), axis=0))
    cands.append(geo)

    # 6. prize / (sum of weights) ** 0.5
    cands.append(p_sum / (np.sqrt(w_sum) + 1e-9))

    # 7. prize / (sum of squares of weights) ** 0.5
    w_sq = np.sqrt((weights ** 2).sum(axis=0))
    cands.append(p_sum / (w_sq + 1e-9))

    # 8. prize * (1 / (1 + w_sum))
    cands.append(p_sum / (1.0 + w_sum))

    # 9. prize / (max weight across constraints)
    cands.append(p_sum / (w_max + 1e-9))

    # 10. prize / (mean weight)
    w_mean = weights.mean(axis=0)
    cands.append(p_sum / (w_mean + 1e-9))

    # 11. prize / (std of weights + mean)
    w_std = weights.std(axis=0)
    cands.append(p_sum / (w_std + w_mean + 1e-9))

    # 12. prize / (range of weights + mean)
    w_range = weights.max(axis=0) - weights.min(axis=0)
    cands.append(p_sum / (w_range + w_mean + 1e-9))

    # 13. prize / (sum of weights) * (1 + prize / (sum of weights))
    ratio = p_sum / (w_sum + 1e-9)
    cands.append(ratio * (1.0 + ratio))

    # 14. prize / (sum of weights) ** 1.5
    cands.append(p_sum / (w_sum ** 1.5 + 1e-9))

    # 15. prize / (sum of weights) ** 0.75
    cands.append(p_sum / (w_sum ** 0.75 + 1e-9))

    # 16. prize / (sum of weights) * (1 / (1 + w_max))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_max))

    # 17. prize / (sum of weights) * (1 / (1 + w_mean))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_mean))

    # 18. prize / (sum of weights) * (1 / (1 + w_std))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_std))

    # 19. prize / (sum of weights) * (1 / (1 + w_range))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_range))

    # 20. prize / (sum of weights) * (1 / (1 + w_sq))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_sq))

    # 21. prize / (sum of weights) * (1 / (1 + geo))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + geo))

    # 22. prize / (sum of weights) * (1 / (1 + ratio))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + ratio))

    # 23. prize / (sum of weights) * (1 / (1 + p_sum))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + p_sum))

    # 24. prize / (sum of weights) * (1 / (1 + p_max))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + p_max))

    # 25. prize / (sum of weights) * (1 / (1 + w_weighted))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_weighted))

    # 26. prize / (sum of weights) * (1 / (1 + w_max + w_mean))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_max + w_mean))

    # 27. prize / (sum of weights) * (1 / (1 + w_max + w_std))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_max + w_std))

    # 28. prize / (sum of weights) * (1 / (1 + w_max + w_range))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_max + w_range))

    # 29. prize / (sum of weights) * (1 / (1 + w_max + w_sq))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_max + w_sq))

    # 30. prize / (sum of weights) * (1 / (1 + w_max + geo))
    cands.append(p_sum / (w_sum + 1e-9) / (1.0 + w_max + geo))

    return cands


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s kill

    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if prizes.ndim != 2 or weights.ndim != 2:
        return np.ones(prizes.shape[1] if prizes.ndim == 2 else 100)
    if prizes.shape != weights.shape:
        return np.ones(prizes.shape[1])
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        return np.ones(prizes.shape[1])

    n_items = prizes.shape[1]

    # Generate candidate heuristics
    candidates = _candidate_heuristics(prizes, weights)

    # Ensure all candidates are positive and finite
    valid_cands = []
    for h in candidates:
        h = np.asarray(h, dtype=float)
        if h.shape != (n_items,):
            continue
        if not np.all(np.isfinite(h)):
            continue
        h = np.maximum(h, 1e-9)
        valid_cands.append(h)

    if not valid_cands:
        return np.ones(n_items)

    # Simulate ACO for each candidate and pick the best
    best_score = -1.0
    best_heuristic = valid_cands[0]

    for i, h in enumerate(valid_cands):
        if time.time() > deadline - 30.0:
            break
        score = _simulate_aco(prizes, weights, h, seed=i)
        if score > best_score:
            best_score = score
            best_heuristic = h

    # Ensure output is positive and finite
    best_heuristic = np.maximum(best_heuristic, 1e-9)
    best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)

    return best_heuristic
# EVOLVE-BLOCK-END
