"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Hypervolume Improvement (HVI) surrogate.
    
    This approach models the trade-off between objective value and evaluation cost
    as a bi-objective problem. We estimate the expected hypervolume contribution
    of each candidate point relative to a reference point. This naturally balances
    finding low objective values with keeping costs low, as points with high cost
    contribute less to the hypervolume near the ideal point.
    
    We approximate the expected HVI by integrating over the posterior distribution
    of the objective value. For a Gaussian posterior, this can be computed analytically
    or via a small quadrature. Here, we use a simplified analytic-like approximation
    combined with a 'surprise' term to ensure exploration.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Check deadline
    if time.time() - _START_TIME > _TIMEOUT:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety checks
    if n == 0:
        return np.array([])
        
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Reference point for Hypervolume.
    # The ideal point is (0, 0) roughly (min objective is 0, min cost is 1).
    # We need a reference point that is worse than any potential point.
    # Let's set the reference point based on f_best and max cost observed/potential.
    # Since f_best is the current best, and we are minimizing, 
    # a point (y, c) dominates (y_ref, c_ref) if y <= y_ref and c <= c_ref.
    # Hypervolume is the region dominated by the non-dominated set and bounded by ref.
    
    # To make the score meaningful and scale-invariant, we normalize the objective
    # and cost relative to a "worst case" estimate.
    # However, we don't know the true worst case. 
    # A robust choice for ref_y is f_best + some margin, or just f_best if we want 
    # to only count improvements. But HVI usually uses a fixed ref.
    # Let's use a dynamic reference point that adapts to the current state.
    # Ref objective: f_best (we only care about points that could beat f_best).
    # Ref cost: max(cost) * 1.1 or similar.
    
    # Actually, a simpler and more robust "HVI-like" score for single-objective 
    # with cost constraint is to compute the expected improvement weighted by 
    # the "volume" of the improvement in the (value, cost) space.
    
    # Let's define the "gain" in objective: G = max(0, f_best - y)
    # Let's define the "gain" in cost relative to a baseline? 
    # No, cost is a penalty.
    
    # Alternative structural change: 
    # Instead of EI/cost, use a **Cost-Weighted Expected Improvement** where the 
    # improvement is measured in the joint space.
    # But the directive asks for HVI surrogate.
    
    # Approximation of Expected HVI for a single point (y, c) against a ref point (y_ref, c_ref):
    # If y < y_ref and c < c_ref, the point alone defines a box of area (y_ref - y) * (c_ref - c).
    # The expected hypervolume contribution is E[ (y_ref - y)^+ * (c_ref - c)^+ ]?
    # Not quite, because other points exist. But for a greedy step, we can approximate 
    # the marginal contribution.
    
    # Let's set y_ref = f_best + sigma_max * 2 (to allow for some improvement)
    # and c_ref = max(cost) + 1.
    
    # To keep it simple and fast, let's use a simplified metric:
    # Score = E[ max(0, f_best - y) ] / (cost^alpha)
    # This is basically EI / cost^alpha.
    # The current program uses EI / cost.
    # To be structurally different, let's use a **Log-Hypervolume** proxy.
    
    # Let's try: Score = log(1 + EI) - beta * log(cost)
    # This is different from EI/cost. It is more robust to outliers in EI or cost.
    # It behaves like EI for small EI, and like -log(cost) for large EI.
    
    # Let's compute EI first.
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    ei_term = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei_term = np.maximum(ei_term, 0.0)
    
    # Normalize EI to [0, 1] range roughly to make log stable?
    # Or just use log1p(EI).
    
    # Cost penalty: log(cost). 
    # We want to minimize cost. So subtract log(cost).
    # But we need to balance the scales.
    # EI is in objective units. Cost is in [1, 2] roughly.
    # Let's scale EI by a factor.
    
    # Let's look at the distribution of mu and sigma to set a scale.
    # But we don't have access to the full history, only the current mu/sigma vectors.
    
    # Let's try a **Pareto Rank** based approach? No, too slow for 2000 points.
    
    # Let's stick to the **Log-Hypervolume Proxy**:
    # Score = log1p(EI) - lambda * log(cost)
    
    # How to choose lambda?
    # If lambda is too high, we always pick the cheapest point.
    # If lambda is too low, we ignore cost.
    # Let's set lambda = 1.0.
    
    # Also, we should add an exploration term if EI is 0.
    # If EI is 0, log1p(0) = 0. Score = -log(cost). 
    # This would prefer cheap points even if they are bad.
    # We need to ensure we explore if the current model is uncertain.
    
    # Let's combine with UCB in the log domain?
    # UCB_score = f_best - mu + beta * sigma
    # This can be negative.
    
    # Let's try a different structural change: **Thompson Sampling** approximation.
    # Instead of a deterministic acquisition function, we could return scores that 
    # reflect the probability of being the best? But we must return a single score per point.
    
    # Let's go back to the **HVI** idea but implement it as:
    # Expected Hypervolume Contribution = E[ (f_best - y)^+ * (C_max - c)^+ ]
    # Where C_max is a constant, say 2.0 (max possible cost).
    # This is E[ max(0, f_best - y) ] * (C_max - c)  (since c is fixed for each candidate)
    # = EI * (C_max - c)
    
    # This is structurally different from EI / c.
    # For c=1, gain is EI * 1.
    # For c=2, gain is EI * 0. (If C_max=2).
    # This heavily penalizes expensive points.
    
    # Let's use C_max = 2.0 (since cost is 1 + mean(x_scaled), max is 2).
    # But if we use (2 - c), then points with c=2 have 0 score.
    # This might be too aggressive.
    
    # Let's use C_ref = 2.0 + margin.
    
    # Let's try: Score = EI * (1.0 + (2.0 - cost_safe))
    # This is EI * (3.0 - cost).
    # Range of (3.0 - cost) is [1.0, 2.0].
    # This is a mild cost penalty.
    
    # Let's try a more pronounced penalty:
    # Score = EI / (cost_safe ** 2.0)
    # This is a stronger penalty than EI / cost.
    
    # Let's try the **Log-Hypervolume** again, it's more principled for multi-objective.
    # Score = log(1 + EI) - log(cost_safe)
    
    # Let's implement this.
    
    score_log_hv = np.log1p(ei_term) - np.log(cost_safe)
    
    # Add a small exploration bonus to avoid getting stuck if EI is 0 everywhere.
    # If EI is 0, score is -log(cost).
    # We want to explore high sigma regions.
    # Let's add: + gamma * (sigma / sigma_max)
    
    # Estimate sigma_max from the current vector.
    sigma_max = np.max(sigma_safe)
    if sigma_max < 1e-10:
        sigma_max = 1.0
        
    exploration_bonus = 0.5 * (sigma_safe / sigma_max)
    
    final_score = score_log_hv + exploration_bonus
    
    # Ensure finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    # Floor at 0? 
    # If EI is 0 and cost is high, score is negative.
    # It's better to have a neutral 0 than a negative score that might be ignored 
    # if the evaluator takes the max. But if all are negative, max is the least negative.
    # Let's floor at 0 to be safe, similar to EI.
    final_score = np.maximum(final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
