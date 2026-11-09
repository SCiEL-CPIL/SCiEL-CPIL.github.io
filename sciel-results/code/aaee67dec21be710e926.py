"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _sanitize(scores):
    """Ensure scores are finite and of the correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, leave 10s safety margin
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Strategy: Use a "Thompson Sampling-like" score combined with EI
    # The idea is to sample from the posterior and rank by the sampled value,
    # but since we can't easily do full TS here without coordinates, we use a
    # proxy: the z-score (standardized improvement) which relates to the
    # probability of improvement, plus a noise term to encourage exploration.
    # 
    # Actually, a better structural change: 
    # Use a "Multi-armed Bandit" style approach where we balance exploitation
    # (EI) and exploration (UCB) but with a dynamic weight based on the 
    # current uncertainty relative to the best value.
    #
    # Let's try a different family: "Probability of Improvement" (PI) combined with
    # a "Noise" term to break ties and encourage exploration in flat regions.
    # PI(x) = Phi((f_best - mu(x)) / sigma(x))
    # This is bounded [0, 1] and focuses on the chance of beating f_best.
    # We combine it with a normalized sigma to add exploration.
    
    z_pi = (f_best_safe - mu_safe) / sigma_safe
    pi_scores = norm.cdf(z_pi)
    
    # Normalize sigma to [0, 1] range for this batch to create a consistent exploration signal
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # The current program uses a fixed linear combination.
    # Let's try a "Multiplicative" or "Ratio" based approach?
    # Or perhaps a "Softmax" style ranking?
    #
    # A distinct structural change: Instead of a linear combination, use a 
    # "Tournament" or "Rank" based acquisition. 
    # However, we need to return scores.
    #
    # Let's try: Score = EI + beta * (sigma / (f_best - mu + epsilon))
    # This amplifies exploration when potential improvement is small relative to noise.
    
    # Actually, let's look at the inspirations.
    # Inspiration 1 (1.81) used a portfolio.
    # Inspiration 3 (1.10) used z-score.
    #
    # Let's try a "Bayesian Optimization with Uncertainty" approach that uses
    # the "Lower Confidence Bound" but with an adaptive kappa that increases
    # as the number of evaluations grows (we don't have the count, but we can
    # infer from the spread of mu/sigma?).
    #
    # Better idea: Use "Expected Improvement" but with a "Lookahead" or "Hypervolume"
    # proxy? No, too complex.
    #
    # Let's try a "Gaussian Process Upper Confidence Bound" (GP-UCB) but optimized
    # for the specific characteristics of Hartmann 3D. Hartmann 3D is multimodal.
    # Standard EI can get stuck. 
    #
    # New Strategy: "Optimistic" acquisition using the "Mean + beta * Std"
    # but with beta scaled by the inverse of the current best value's uncertainty?
    #
    # Let's try a completely different mathematical form:
    # Score = -mu + kappa * sigma * log(1 + sigma / (f_best - mu + 1e-6))
    # This is a log-UCB variant that dampens exploration when improvement is large.
    
    improvement = f_best_safe - mu_safe
    # Avoid division by zero or negative log
    log_term = np.log1p(np.maximum(sigma_safe / np.maximum(improvement, 1e-6), 0))
    
    # Adaptive kappa: If sigma is high relative to improvement, explore more.
    # kappa = 1.5 + 0.5 * (sigma_safe / (np.std(mu_safe) + 1e-6))
    # This is getting complicated.
    
    # Let's go back to a robust, simple but distinct family:
    # "Probability of Improvement" (PI) is often better for multimodal functions
    # when combined with a strong exploration term.
    #
    # Let's try: Score = PI * (1 + sigma_norm)
    # This rewards both high probability of improvement AND high uncertainty.
    
    pi_scores = norm.cdf(z_pi)
    
    # Add a small exploration bonus based on normalized sigma
    exploration_bonus = 0.5 * sigma_norm
    
    final_scores = pi_scores + exploration_bonus
    
    # Check time
    if time.time() > deadline:
        return _sanitize(pi_scores)
    
    final_scores = _sanitize(final_scores)
    
    return final_scores
# EVOLVE-BLOCK-END
