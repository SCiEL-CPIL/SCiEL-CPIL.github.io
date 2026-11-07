"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Hypervolume-based cost-aware acquisition function.
    
    Instead of simply dividing EI or UCB by cost, we estimate the expected
    contribution of each candidate to the hypervolume of the non-dominated
    front in the (value, cost) space. This naturally balances low value
    (good objective) and low cost (cheap to evaluate) without arbitrary
    scaling parameters like beta or temperature.
    
    We also incorporate a 'surprise' term based on the GP uncertainty to
    encourage exploration in regions where the model is poor, scaled by
    the potential hypervolume gain.
    """
    # Check deadline
    if time.time() - _START_TIME > _TIMEOUT:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure positive values for log-space operations if needed, though we use linear HV here
    # But we need to be careful with the 'reference point' for hypervolume.
    
    # Reference point for hypervolume calculation.
    # We want to minimize value and minimize cost.
    # A standard approach is to use the current worst observed value and max possible cost.
    # However, since we don't have the full history, we approximate the reference point.
    # f_best is the best value seen so far.
    # cost is in [1, 2] (since x scaled to [0,1] is in [0,1], mean is in [0,1], so cost in [1,2]).
    # Let's define a reference point that is 'worse' than any potential candidate.
    # ref_value = f_best (since any improvement must be < f_best)
    # ref_cost = 2.0 (max possible cost)
    
    # Actually, for a single-point acquisition, we can compute the *incremental* hypervolume
    # that a new point (v, c) would add to a front that currently only contains the reference point
    # and potentially the best known point (f_best, c_best).
    # Since we don't have c_best, let's assume the 'current front' is effectively anchored at
    # (f_best, max_cost) and we are looking for points that dominate the 'origin' of the improvement space.
    
    # A simpler and more robust structural change: 
    # Use the **Log-EI** or **Hypervolume EI** approximation.
    # For a single objective with cost, the optimal policy is to maximize the expected
    # reduction in the 'regret per cost' or similar.
    
    # Let's try a different structural family: **Knowledge Gradient (KG)** approximation.
    # KG estimates the expected improvement in the final best value.
    # For cost-aware, we consider the probability that this point becomes the new best
    # given its cost.
    
    # Approximation:
    # P(new best) = P(f(x) < f_best | mu, sigma) = Phi((f_best - mu)/sigma)
    # Expected improvement in best value = E[max(0, f_best - f(x))] = EI
    # But we care about the *rate* or *efficiency*.
    
    # Let's use a **Cost-Weighted Expected Improvement with a Gaussian Kernel smoothing**.
    # The current program uses a hard division by cost. This creates a discontinuity in the
    # landscape if cost varies smoothly but the division is sharp.
    # Instead, let's use a **soft cost penalty** in the exponent of a Gaussian-like score.
    
    # New Strategy: **Exponential Utility Acquisition**
    # Score = exp( -alpha * (mu - f_best)^2 / sigma^2 - beta * cost ) * sigma
    # This is essentially a Gaussian-like exploration term modulated by cost.
    
    # Even better: **Hypervolume Indicator Approximation**
    # Let's compute the expected hypervolume contribution.
    # For a point (mu, cost), the expected value is mu.
    # The contribution to the front is roughly the area between the current front and this point.
    # If we assume the current front is at (f_best, C_max), the area added by a point (v, c)
    # that is better than f_best is (f_best - v) * (C_max - c).
    # E[Area] = E[(f_best - V) * (C_max - C)]
    # Since C is deterministic for a candidate, and V ~ N(mu, sigma^2):
    # E[(f_best - V)] = f_best - mu
    # But we need max(0, f_best - V).
    # E[max(0, f_best - V) * (C_max - C)] = (C_max - C) * E[max(0, f_best - V)]
    # E[max(0, f_best - V)] = EI (standard expected improvement)
    
    # So, Score = (C_max - cost) * EI
    # This is a structural change from EI/Cost to (C_max - Cost) * EI.
    # It favors points with low cost (high C_max - cost) AND high EI.
    # It avoids the division by small costs which can blow up scores for expensive points
    # if EI is slightly larger. It also naturally zeros out the score if cost is high.
    
    C_max = 2.0  # Max possible cost is 1 + 1.0
    
    # Calculate EI
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    ei_term = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei_term = np.maximum(ei_term, 0.0)
    
    # Cost factor: (C_max - cost)
    # cost is in [1, 2]. So (C_max - cost) is in [0, 1].
    # This acts as a weight.
    cost_weight = np.maximum(C_max - cost, 0.0)
    
    # Combine
    score = cost_weight * ei_term
    
    # To further differentiate, we can add a small exploration bonus based on sigma
    # weighted by the cost factor, to ensure we don't just exploit.
    # Exploration term: sigma * cost_weight
    # This is similar to UCB but scaled by cost availability.
    
    # Let's blend EI and UCB-like term using the cost weight.
    # UCB score: f_best - mu + beta * sigma
    # Let's use beta = 1.0
    beta = 1.0
    ucb_term = (f_best - mu + beta * sigma_safe)
    ucb_term = np.maximum(ucb_term, 0.0) # Ensure non-negative contribution
    
    # Soft-max blend of EI and UCB, both weighted by cost availability
    # This is structurally different because the cost weight is multiplicative
    # on the *value* of the acquisition, not a divisor.
    
    # We can also normalize the scores to be in a comparable range.
    # But the evaluator just picks the max.
    
    # Let's try a simple weighted sum first, as it's more stable.
    # w_ei = 0.7, w_ucb = 0.3
    # score = cost_weight * (0.7 * ei_term + 0.3 * ucb_term)
    
    # However, the "inspiration" mentioned adaptive exploration.
    # Let's make the UCB weight adaptive based on the magnitude of EI.
    # If EI is high, exploit more. If EI is low, explore more.
    # But we only have mu, sigma, f_best, cost. We don't have a global count.
    # We can estimate "convergence" by looking at the ratio of sigma to (f_best - min(mu)).
    # If sigma is large relative to the range of mu, we are early.
    
    # Let's stick to the structural change: **Multiplicative Cost Weighting** instead of Division.
    # This is a different optimization landscape.
    
    score = cost_weight * ei_term + 0.5 * cost_weight * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Floor at 0
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
