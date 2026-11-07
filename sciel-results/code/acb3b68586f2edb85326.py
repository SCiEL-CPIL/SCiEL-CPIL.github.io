"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using a marginal utility / opportunity cost heuristic.
    
    For each constraint (resource), we estimate scarcity by looking at the efficiency
    (prize/weight) of the top items. Items that consume scarce resources at low efficiency
    get penalized, while items that offer high prize relative to the opportunity cost of
    their resource consumption are rewarded.
    
    Strategy:
    1. For each constraint c, compute the efficiency of each item: prize[c,i] / weight[c,i]
    2. Sort items by efficiency for each constraint; the top-k items define the "scarcity"
       level. We use a weighted sum of the top efficiencies to estimate the opportunity
       cost per unit of resource c.
    3. For each item i, compute the opportunity cost: sum over constraints c of
       weight[c,i] * opportunity_cost_per_unit[c]
    4. Score = total_prize[i] / (1 + opportunity_cost[i])
    
    This rewards items that give high total prize while consuming resources that are
    "cheap" in terms of opportunity cost (i.e., resources where there are many efficient
    alternatives), and penalizes items that consume scarce resources at low efficiency.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    
    # Total prize for each item
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # For each constraint, compute efficiency of each item
    # efficiency[c, i] = prizes[c, i] / max(weights[c, i], eps)
    eps = 1e-9
    safe_weights = np.maximum(weights, eps)
    efficiency = prizes / safe_weights  # shape (5, 100)
    
    # For each constraint, estimate the opportunity cost per unit of resource
    # by looking at the top-k efficiencies. We use k = 20 (about 20% of items).
    k = max(1, n_items // 5)  # top 20 items per constraint
    
    # Sort efficiencies in descending order for each constraint
    # top_eff[c] will be the k-th highest efficiency for constraint c
    # We use the mean of the top-k as the opportunity cost per unit
    sorted_eff = np.sort(efficiency, axis=1)[:, ::-1]  # descending, shape (5, 100)
    top_k_eff = sorted_eff[:, :k]  # shape (5, k)
    
    # Opportunity cost per unit of resource c: mean of top-k efficiencies
    opp_cost_per_unit = top_k_eff.mean(axis=1)  # shape (5,)
    
    # For each item, compute total opportunity cost:
    # opp_cost[i] = sum_c weight[c,i] * opp_cost_per_unit[c]
    opp_cost = weights @ opp_cost_per_unit  # shape (100,)
    
    # Score = total_prize / (1 + opp_cost)
    # This gives higher scores to items with high prize relative to the opportunity cost
    score = total_prize / (1.0 + opp_cost)
    
    # Ensure all values are positive and finite
    score = np.where(np.isfinite(score) & (score > 0), score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
