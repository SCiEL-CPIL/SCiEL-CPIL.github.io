"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr
import time

_DEADLINE = 325.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Cost-Adjusted Probability of Improvement (POI)
    combined with a cost-normalized Upper Confidence Bound (UCB) for exploration.
    
    The POI threshold is dynamically adjusted based on remaining budget to 
    force larger relative improvements as the run progresses.
    
    Returns 2000 finite scores (higher = more promising).
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    if n != 2000:
        return np.ones(n)

    # Safety for division and log
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1.0)  # Cost is at least 1

    # --- 1. Calculate Progress and Remaining Budget ---
    # Hartmann 6d global min is approx -3.32237
    f_min = -3.32237
    f_start = -1.0  # Typical initial observation range
    
    # Progress: 0.0 at start, 1.0 at end
    progress = np.clip((f_start - f_best) / (f_start - f_min), 0.0, 1.0)
    
    # Remaining budget proxy:
    # Total budget is 30. We don't know exact total_cost, but we can estimate 
    # remaining potential evaluations or just use 'progress' as a proxy for 
    # resource depletion. 
    # The directive suggests: target = f_best - gamma * (30 - total_cost) / n_evals_remaining
    # Since we don't have total_cost explicitly passed (only the cost vector of 
    # *candidates* being scored, or is it the history? The task says 
    # "cost have shape (2000,)" and "cost is ... for the candidates being scored" 
    # wait, re-reading: "mu, sigma and cost have shape (2000,). Return 2000 finite scores".
    # "Evaluating x costs ... Runs start with 5 uniform random points (cost counted). 
    # While total cost is below 30... evaluator scores 2000 new uniform candidates...
    # and evaluates the best one."
    # The `cost` parameter in the function signature usually refers to the cost 
    # associated with evaluating *these specific candidates* if they were chosen, 
    # OR it's the cost of the *history*? 
    # Looking at the signature `acquisition(mu, sigma, f_best, cost)`:
    # In standard BO loops with cost, `cost` is often the cost vector of the 
    # *candidates* being considered, or the cost of the *observations*?
    # Given `mu` and `sigma` are for the 2000 candidates, `cost` is almost certainly 
    # the evaluation cost *for these 2000 candidates*.
    # However, the "remaining budget" depends on the *history* of evaluations.
    # We don't have the total spent cost passed directly. 
    # BUT, `f_best` changes over time. 
    # Let's approximate remaining budget fraction as (1 - progress).
    # If progress is 0, we have full budget. If 1, we have none.
    
    # The directive formula: `f_best - gamma * (30 - total_cost) / n_evals_remaining`
    # We don't have `total_cost` or `n_evals_remaining` directly.
    # We can estimate `n_evals_remaining` based on average cost. 
    # Avg cost is roughly 1.5 (mean of 1 + mean(x) where mean(x)=0.5 -> 1.5).
    # Total budget 30 -> ~20 evals max.
    # Let's use `progress` to modulate the threshold tightness.
    
    # As progress increases, we should demand *higher* improvement (larger threshold).
    # Standard POI: P(f(x) < f_best - xi)
    # If we want larger improvements, we increase xi.
    # Let xi = base_xi * (1 + progress * sensitivity)
    
    base_xi = 0.0  # Standard POI starts with xi=0
    # Scale xi up as we approach the end to prevent marginal gains
    xi = 0.05 * (1.0 + 2.0 * progress) 
    # At start (prog=0): xi=0.05. At end (prog=1): xi=0.15.
    # This forces the model to only pick points with >0.05 to >0.15 expected improvement 
    # in terms of probability mass.

    # --- 2. Compute POI ---
    # P(f(x) < f_best - xi) = Phi( (f_best - xi - mu) / sigma )
    z_poi = (f_best - xi - mu) / sigma_safe
    z_poi = np.clip(z_poi, -10, 10) # numerical stability
    poi = ndtr(z_poi)
    
    # --- 3. Compute Cost-Aware UCB for Exploration ---
    # We want to balance POI (exploitation) with Uncertainty (exploration).
    # UCB = mu + kappa * sigma
    # But we must penalize cost.
    # Score = POI / Cost^exponent + ExplorationBonus
    
    # Exploration bonus: How much does this point reduce our uncertainty?
    # A point with high sigma is good for exploration.
    # Let's use a "Value of Information" style term or just a scaled UCB.
    # To make it distinct from the previous EI+LCB:
    # We will use a multiplicative cost penalty on the POI term, 
    # and an additive cost-penalized uncertainty term.
    
    # Uncertainty term: sigma / cost
    # This rewards high uncertainty but penalizes high cost.
    # Let's scale it appropriately.
    # Typical sigma is around 0.1-1.0. Typical cost is 1.0-2.0.
    
    # Adaptive kappa: Start high, end low.
    kappa = 1.5 * (1.0 - 0.5 * progress) # 1.5 early, 0.75 late
    
    ucb_term = mu + kappa * sigma_safe
    # Convert UCB to a "gain" relative to f_best?
    # No, UCB is a value. Lower is better for minimization? 
    # Wait, UCB for minimization is usually LCB (Lower Confidence Bound) = mu - kappa*sigma.
    # Let's stick to LCB for minimization.
    lcb = mu - kappa * sigma_safe
    # LCB Gain: f_best - lcb (how much better the lower bound is)
    lcb_gain = np.maximum(f_best - lcb, 0.0)
    
    # Normalize LCB gain to be comparable to POI (which is 0-1)?
    # LCB gain can be large. Let's divide by a scale factor.
    # Or, simpler: Combine POI and a normalized LCB.
    
    # Let's try a different combination structure:
    # Score = (POI * Weight_Exploit + LCB_Gain * Weight_Explore) / Cost^p
    
    # But the directive specifically asked for "Cost-Adjusted POI".
    # Let's make POI the dominant term, adjusted by cost.
    
    # Cost Penalty: exp(-lambda * cost) or 1/cost^p?
    # 1/cost^1.5 was used before. Let's use exponential for a sharper cutoff.
    # lambda = 0.5
    cost_penalty = np.exp(-0.5 * (cost_safe - 1.0)) # Max penalty at high cost
    # At cost=1, penalty=1. At cost=2, penalty=exp(-0.5)=0.6.
    
    # Primary Score: POI * Cost_Penalty
    primary_score = poi * cost_penalty
    
    # Secondary Score: Exploration
    # We add a term that encourages high sigma (uncertainty) but penalizes cost.
    # exploration_term = sigma_safe / cost_safe
    # Scale it to be in the same range as primary_score (0-1).
    # Max sigma is ~1. Min cost 1. So max ~1.
    exploration_term = (sigma_safe / cost_safe) * 0.2 * (1.0 - 0.5 * progress)
    
    # Final Score
    scores = primary_score + exploration_term
    
    # Ensure finite and positive
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 1e-10)
    
    return scores
# EVOLVE-BLOCK-END
