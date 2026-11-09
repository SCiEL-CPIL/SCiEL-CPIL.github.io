"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def _simulate_aco(prizes, weights, heuristic, n_ants=10, n_iters=10, capacity=25):
    """Run a local ACO simulation to evaluate a heuristic. Returns best total prize."""
    n_items = prizes.shape[1]
    pheromone = np.ones(n_items)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    rng = np.random.RandomState(12345)
    
    best_total = 0.0
    
    for _ in range(n_iters):
        iter_best = 0.0
        iter_best_items = []
        
        for _ant in range(n_ants):
            remaining = np.ones(n_items, dtype=bool)
            used = np.zeros(prizes.shape[0])
            total_prize = 0.0
            items = []
            
            while True:
                # Check which items still fit
                fits = remaining.copy()
                for c in range(prizes.shape[0]):
                    fits &= (used[c] + weights[c] <= capacity)
                    # Actually need per-item check
                # More efficient: for each remaining item, check all constraints
                candidates = []
                for i in range(n_items):
                    if not remaining[i]:
                        continue
                    ok = True
                    for c in range(prizes.shape[0]):
                        if used[c] + weights[c, i] > capacity:
                            ok = False
                            break
                    if ok:
                        candidates.append(i)
                
                if not candidates:
                    break
                
                cand_arr = np.array(candidates)
                probs = pheromone[cand_arr] * heuristic[cand_arr]
                total = probs.sum()
                if total <= 0:
                    break
                probs = probs / total
                
                idx = rng.choice(len(cand_arr), p=probs)
                chosen = cand_arr[idx]
                
                remaining[chosen] = False
                for c in range(prizes.shape[0]):
                    used[c] += weights[c, chosen]
                total_prize += prizes[:, chosen].sum()
                items.append(chosen)
            
            if total_prize > iter_best:
                iter_best = total_prize
                iter_best_items = items
        
        if iter_best > best_total:
            best_total = iter_best
        
        # Update pheromone
        pheromone *= 0.9
        pheromone += iter_best / 10.0
    
    return best_total


def _pareto_dominance_score(prizes, weights):
    """Score items based on whether they are Pareto-optimal in prize-weight space."""
    n_items = prizes.shape[1]
    n_constraints = prizes.shape[0]
    
    # For each item, compute:
    # - total prize (sum across constraints)
    # - total weight (sum across constraints)
    prize_sum = prizes.sum(axis=0)
    weight_sum = weights.sum(axis=0)
    
    # An item i is dominated by item j if:
    #   prize_j >= prize_i (all constraints) AND weight_j <= weight_i (all constraints)
    #   with at least one strict inequality
    # Simplified: use sum of prizes and sum of weights for dominance check
    # More robust: check if there exists j with prize_j[c] >= prize_i[c] for all c
    # and weight_j[c] <= weight_i[c] for all c, with strict in at least one
    
    dominated = np.zeros(n_items, dtype=bool)
    
    # For efficiency with 100 items, O(n^2) is fine
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            # Check if j dominates i
            prize_ge = np.all(prizes[:, j] >= prizes[:, i])
            weight_le = np.all(weights[:, j] <= weights[:, i])
            # Strict in at least one dimension
            strict_prize = np.any(prizes[:, j] > prizes[:, i])
            strict_weight = np.any(weights[:, j] < weights[:, i])
            if prize_ge and weight_le and (strict_prize or strict_weight):
                dominated[i] = True
                break
    
    # Non-dominated items get higher scores
    # Use prize/weight ratio as base, boosted for non-dominated
    ratio = prize_sum / (weight_sum + 1e-9)
    
    # Normalize
    score = np.ones(n_items)
    non_dom = ~dominated
    if non_dom.sum() > 0:
        score[non_dom] = ratio[non_dom] / ratio[non_dom].max() + 1.0
        score[dominated] = 0.1
    
    return score


def _multi_constraint_ratio(prizes, weights):
    """Score based on prize-to-weight ratio considering each constraint separately."""
    n_items = prizes.shape[1]
    n_constraints = prizes.shape[0]
    
    # For each constraint, compute ratio
    ratios = []
    for c in range(n_constraints):
        r = prizes[c] / (weights[c] + 1e-9)
        ratios.append(r)
    
    ratios = np.array(ratios)  # shape (n_constraints, n_items)
    
    # Item is attractive if it has good ratio across all constraints
    # Use minimum ratio (weakest constraint) as a conservative score
    min_ratio = ratios.min(axis=0)
    mean_ratio = ratios.mean(axis=0)
    
    # Combine: emphasize items that are good across all constraints
    score = min_ratio * (1 + 0.5 * mean_ratio / (mean_ratio.max() + 1e-9))
    
    return score


def _bottleneck_constraint_score(prizes, weights):
    """Focus on the most constraining resource for each item."""
    n_items = prizes.shape[1]
    n_constraints = prizes.shape[0]
    capacity = 25
    
    # For each item, find which constraint is most binding
    # i.e., which constraint has the highest weight fraction
    # Higher weight means more constraining
    weight_fractions = weights / capacity  # shape (n_constraints, n_items)
    
    # For each item, the bottleneck is the constraint with max weight fraction
    bottleneck_idx = weight_fractions.argmax(axis=0)  # shape (n_items,)
    
    # Score: prize in the bottleneck constraint / weight in that constraint
    score = np.zeros(n_items)
    for i in range(n_items):
        c = bottleneck_idx[i]
        score[i] = prizes[c, i] / (weights[c, i] + 1e-9)
    
    # Also consider total prize
    total_prize = prizes.sum(axis=0)
    total_weight = weights.sum(axis=0)
    overall_ratio = total_prize / (total_weight + 1e-9)
    
    # Blend
    score = 0.6 * score / (score.max() + 1e-9) + 0.4 * overall_ratio / (overall_ratio.max() + 1e-9)
    
    return score


def _composite_score(prizes, weights):
    """Combine multiple scoring strategies."""
    s1 = _multi_constraint_ratio(prizes, weights)
    s2 = _bottleneck_constraint_score(prizes, weights)
    s3 = _pareto_dominance_score(prizes, weights)
    
    # Normalize each
    def normalize(x):
        x = np.asarray(x, dtype=float)
        mn = x.min()
        mx = x.max()
        if mx - mn < 1e-12:
            return np.ones_like(x)
        return (x - mn) / (mx - mn)
    
    n1 = normalize(s1)
    n2 = normalize(s2)
    n3 = normalize(s3)
    
    # Weighted combination
    score = 0.4 * n1 + 0.3 * n2 + 0.3 * n3
    
    return score


def _generate_candidate_heuristics(prizes, weights):
    """Generate multiple candidate heuristics to evaluate."""
    candidates = []
    
    # 1. Simple ratio
    prize_sum = prizes.sum(axis=0)
    weight_sum = weights.sum(axis=0)
    candidates.append(prize_sum / (weight_sum + 1e-9))
    
    # 2. Multi-constraint ratio
    candidates.append(_multi_constraint_ratio(prizes, weights))
    
    # 3. Bottleneck constraint
    candidates.append(_bottleneck_constraint_score(prizes, weights))
    
    # 4. Pareto dominance
    candidates.append(_pareto_dominance_score(prizes, weights))
    
    # 5. Composite
    candidates.append(_composite_score(prizes, weights))
    
    # 6. Prize-weighted: emphasize high-prize items
    prize_max = prizes.max(axis=0)
    candidates.append(prize_max / (weight_sum + 1e-9))
    
    # 7. Geometric mean of ratios across constraints
    ratios = prizes / (weights + 1e-9)
    geo_mean = np.exp(np.log(ratios).mean(axis=0))
    candidates.append(geo_mean)
    
    return candidates


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 270  # 60s safety margin before 330s limit
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    n_items = prizes.shape[1]
    n_constraints = prizes.shape[0]
    
    # Generate candidate heuristics
    candidates = _generate_candidate_heuristics(prizes, weights)
    
    # Ensure all candidates have correct shape and are finite
    valid_candidates = []
    for c in candidates:
        c = np.asarray(c, dtype=float).reshape(-1)
        if c.shape == (n_items,) and np.all(np.isfinite(c)):
            valid_candidates.append(c)
    
    if not valid_candidates:
        # Fallback
        return prizes.sum(axis=0) / (weights.sum(axis=0) + 1e-9)
    
    # Evaluate each candidate with local ACO simulation
    best_heuristic = valid_candidates[0]
    best_score = -1.0
    
    for idx, cand in enumerate(valid_candidates):
        elapsed = time.time() - start_time
        if elapsed > deadline - 40:  # Need buffer for remaining work
            break
        
        # Use fewer ants/iterations for speed, but enough to differentiate
        n_ants = 8
        n_iters = 6
        
        score = _simulate_aco(prizes, weights, cand, n_ants=n_ants, n_iters=n_iters, capacity=25)
        
        if score > best_score:
            best_score = score
            best_heuristic = cand
    
    # Final validation
    result = np.asarray(best_heuristic, dtype=float)
    result = np.where(np.isfinite(result), result, 1e-9)
    result = np.where(result > 0, result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
