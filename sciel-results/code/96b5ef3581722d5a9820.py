"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Multi-sample Thompson Sampling acquisition for Bayesian optimization.
    
    We sample multiple latent function values from the GP posterior at each
    candidate point and score by the negative of the minimum sampled value.
    This directly optimizes for the probability that a candidate is the new minimum.
    
    For efficiency with 2000 candidates, we use a reduced set of samples 
    and exploit the fact that for 1D GP with Matern-5/2, the posterior at 
    candidate points can be computed efficiently.
    
    However, doing full Cholesky on 2000x2000 for each call may be expensive.
    Instead, we use a simpler approach: sample a few times from the posterior
    using the mean and std, which for a GP posterior is Gaussian.
    
    Actually, the posterior at each candidate point is independent in the sense
    that we can sample f(x_i) ~ N(mu_i, sigma_i^2) independently for each i.
    This is NOT the same as joint sampling from the GP posterior (which would
    account for correlations), but it's a common approximation and much faster.
    
    For a more principled approach with limited budget, let's use a hybrid:
    - Compute the standard EI as a baseline
    - Add an exploration bonus based on uncertainty
    - Use a UCB-like acquisition with adaptive beta
    
    Actually, let me try a different structural approach: use a combination of
    Expected Improvement and Upper Confidence Bound, which balances exploitation
    and exploration differently than pure EI.
    
    Let's use: score = (f_best - mu) * Phi(z) + sigma * phi(z) + beta * sigma
    where beta is chosen to encourage exploration in early stages.
    
    But since we don't know the step number, let's use a fixed reasonable beta.
    
    Alternative: Use the Probability of Improvement with a small epsilon,
    combined with uncertainty.
    
    Let me try: score = -mu + beta * sigma
    This is a UCB (Upper Confidence Bound for minimization, so we want low mu
    and high sigma for exploration).
    
    Actually for minimization, UCB would be: mu - beta * sigma (we want low values)
    But we want higher score to mean "evaluate next", so:
    score = -(mu) + beta * sigma = -mu + beta * sigma
    
    Let's combine EI and UCB:
    score = EI + gamma * sigma
    
    Or better, let's use a well-tuned UCB:
    score = -mu + sqrt(2 * log(N) * t) * sigma  -- but we don't have t
    
    Let's just use: score = -mu + beta * sigma with beta = 2.0
    This encourages exploring regions with high uncertainty.
    """
    # Guard against zero sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB component for exploration
    # For minimization, we want to explore where sigma is high
    # Higher score = more likely to be evaluated next
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Combine: use EI as primary, add exploration bonus
    # Normalize EI to [0, 1] range approximately, then blend
    # Actually, let's just use a weighted sum
    # EI is in objective units, UCB is in objective units
    # Let's use: score = 0.7 * ei + 0.3 * ucb
    # But they have different scales. Let's think...
    
    # Better approach: Use EI but with a modified sigma that accounts for
    # the fact that we're minimizing. Actually EI already does this.
    
    # Let's try a pure UCB with a reasonable beta:
    # For 1D sphere function on [-5,5], the scale of the function is up to 25
    # sigma will be in that range too
    # beta = 2.0 seems reasonable
    
    # Actually, let me reconsider. The issue with pure EI is that it can be
    # too greedy. Let's use a hybrid that's known to work well:
    
    # score = EI + alpha * sigma, where alpha is small
    # This adds a small exploration bonus on top of EI
    
    alpha = 0.5
    score = ei + alpha * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
