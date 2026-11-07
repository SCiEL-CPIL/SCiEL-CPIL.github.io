"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Multi-sample Thompson Sampling acquisition function.
    
    mu: array of shape (2000,) - posterior mean at candidates
    sigma: array of shape (2000,) - posterior std at candidates
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    n_candidates = len(mu)
    n_samples = 100
    
    # Since we only have mu and sigma (marginal posterior at each point),
    # we cannot do full Cholesky-based joint sampling without the full covariance matrix.
    # However, the directive asks for a multi-sample Thompson Sampling estimate.
    # With only marginal information, the best we can do is sample from the marginal
    # Gaussian at each point independently. This is a simplified Thompson Sampling
    # where we sample from N(mu, sigma^2) at each candidate independently.
    
    # Generate random samples from the marginal posterior at each candidate
    # Shape: (n_samples, n_candidates)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    samples = mu[None, :] + sigma[None, :] * rng.standard_normal((n_samples, n_candidates))
    
    # For minimization, we want to find candidates that are likely to have low values.
    # Thompson Sampling: sample a function value at each candidate, pick the one with lowest value.
    # The score for each candidate should reflect how often it would be selected.
    # A common approach: use the negative of the minimum sampled value per candidate,
    # or use the quantile of the samples.
    
    # Following the directive: return the negative of the minimum sampled value per candidate
    # Wait, the directive says "return the negative of the minimum sampled value per candidate as the score"
    # But that doesn't quite make sense for Thompson Sampling. Let me re-read.
    
    # Actually, in Thompson Sampling for minimization:
    # - We sample a full function from the posterior
    # - We pick the point with the minimum sampled value
    # - The probability that point x is selected is the score
    
    # With independent marginal samples, we can approximate this by:
    # For each sample, find the argmin, and count how often each candidate is selected.
    # Or, we can use the expected minimum of the samples as a proxy.
    
    # Let's use the approach: for each candidate, compute the expected value of the minimum
    # across samples... no, that's not right either.
    
    # Simpler interpretation: The score for candidate i is how "good" it looks.
    # In Thompson Sampling, we'd pick argmin of the sampled values.
    # A reasonable score: the negative of the mean of the samples (which is just -mu),
    # or we can use a quantile-based approach.
    
    # Let's follow the directive more literally: "return the negative of the minimum sampled value per candidate"
    # This means for each candidate i, take min over samples of samples[:, i], then negate.
    
    min_per_candidate = np.min(samples, axis=0)  # Shape: (n_candidates,)
    scores = -min_per_candidate
    
    return scores
# EVOLVE-BLOCK-END
