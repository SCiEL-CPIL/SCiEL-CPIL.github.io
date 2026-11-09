# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Stochastic Weighted Average (SWA) of Predictive Distributions.
    
    Instead of a single point estimate for the acquisition parameter (beta),
    we sample multiple beta values from a distribution that reflects the
    current uncertainty in the GP. This creates a portfolio of acquisition
    functions. We then select the candidate that maximizes the expected
    minimum regret across this ensemble, effectively hedging against
    model uncertainty in the acquisition parameter itself.
    
    The distribution of beta is centered around a value that adapts to
    the current mean sigma, with variance that decreases as the GP becomes
    more certain (lower sigma).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Check for time budget
    start_time = time.time()
    
    # Determine the number of samples for the SWA ensemble
    # We want to be fast, so we limit this.
    # 1000 candidates, 20 samples -> 20000 operations, very fast.
    n_samples = 20
    
    # Adaptive Beta Distribution
    # Mean beta should be higher when sigma is high (exploration)
    # Variance of beta should be higher when sigma is high (uncertainty about beta)
    mean_sigma = np.mean(sigma)
    
    # Base beta logic from previous version, but now treated as a random variable
    base_beta = 1.0 + 2.0 * mean_sigma
    
    # Variance of beta: 
    # If mean_sigma is high, we are less sure about the optimal beta, so wider spread.
    # If mean_sigma is low, we are more sure, so tighter spread.
    # Let's say variance is proportional to mean_sigma^2, but capped.
    beta_std = 0.5 * mean_sigma + 0.1
    
    # Sample betas from a Normal distribution, clipped to positive values
    # Using a fixed seed? No, stochastic is fine, or we can use a deterministic set
    # for reproducibility if needed, but the task allows random.
    # However, for stability in scoring, maybe a fixed grid of betas is better than random sampling
    # each time? The directive says "sampling multiple GP weight vectors".
    # Let's use a deterministic set of betas centered on base_beta to avoid
    # high variance in the score itself due to random sampling of betas.
    # Actually, the "Stochastic" part usually implies randomness.
    # But if the score fluctuates too much, it might hurt.
    # Let's use a small number of random samples.
    
    rng = np.random.default_rng()
    betas = rng.normal(base_beta, beta_std, size=n_samples)
    betas = np.clip(betas, 0.1, 10.0) # Ensure positive and reasonable range
    
    # Compute LCB scores for each beta sample
    # Shape: (n_samples, 2000)
    lcb_scores = -mu[None, :] + betas[:, None] * sigma[None, :]
    
    # Cost-aware normalization
    # scores = lcb / cost
    # We want to maximize the "expected minimum regret" or similar robust metric.
    # A common robust strategy is to maximize the mean of the sorted scores,
    # or the quantile.
    # Let's try maximizing the mean of the top 50% of scores across the ensemble?
    # Or simply the mean of the scores?
    # The "portfolio optimization" idea suggests we want candidates that perform well
    # on average across the different beta scenarios.
    # Let's compute the mean score for each candidate across the beta samples.
    mean_scores = np.mean(lcb_scores, axis=0)
    
    # To add a "robustness" or "hedging" aspect, we can penalize high variance
    # in the scores across the beta samples.
    # Variance of scores for each candidate:
    var_scores = np.var(lcb_scores, axis=0)
    
    # Robust Score = Mean Score - lambda * Std Score
    # This encourages candidates that are good on average AND consistent across
    # different exploration/exploitation settings.
    # If a candidate is great for one beta but bad for others, it has high variance.
    std_scores = np.sqrt(var_scores)
    
    # lambda: how much to penalize variance.
    # If lambda is 0, it's just mean LCB.
    # If lambda is large, it prefers consistent performers.
    # Let's set lambda to a value that scales with the magnitude of the scores.
    # Or a fixed value.
    # Given scores are in objective units / cost, and mu/sigma are standardized-ish,
    # let's try lambda = 0.5.
    lambda_penalty = 0.5
    
    scores = mean_scores - lambda_penalty * std_scores
    
    # Cost normalization was applied inside lcb_scores? No, I did lcb_scores = -mu + beta*sigma.
    # I need to divide by cost.
    # Let's re-do that properly.
    
    # lcb_scores is currently (n_samples, 2000)
    # Divide by cost (broadcasting)
    lcb_scores_cost = lcb_scores / cost[None, :]
    
    # Recalculate mean and std with cost normalized scores
    mean_scores = np.mean(lcb_scores_cost, axis=0)
    var_scores = np.var(lcb_scores_cost, axis=0)
    std_scores = np.sqrt(var_scores)
    
    scores = mean_scores - lambda_penalty * std_scores
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
