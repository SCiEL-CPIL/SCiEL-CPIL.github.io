import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural Change: Thompson Sampling
    # Sample from the posterior distribution N(mu, sigma^2)
    # For minimization, we want to find the candidate with the lowest sample value.
    # Since the evaluator picks the candidate with the HIGHEST score,
    # we return the negative of the sample: score = -sample.
    
    # To improve stability and exploration, we can use multiple samples and take the average
    # or a quantile. However, standard Thompson Sampling is just one sample.
    # Let's try a "robust" Thompson Sampling where we take the mean of K samples.
    # This reduces the variance of the selection criterion, potentially leading to more consistent
    # exploration of regions with high posterior mean but also high uncertainty.
    
    # Alternatively, we can use a "multi-sample" approach where we sample K times
    # and score based on the best (lowest) sample. This is similar to taking a lower quantile.
    
    # Let's try K=10 samples and take the mean.
    # This approximates the expected value but with a stochastic twist.
    
    # Actually, standard Thompson Sampling is known to be effective.
    # Let's implement it with a fixed number of samples to ensure we don't time out.
    # If K is too large, it might be slow. K=1 is fastest.
    
    # Let's try a hybrid: 
    # 1. Compute EI (Expected Improvement)
    # 2. Compute UCB (Upper Confidence Bound for minimization, i.e., lower bound)
    # 3. Combine them?
    
    # The directive says "Make a STRUCTURAL change".
    # Thompson Sampling is a distinct structural change from deterministic EI/UCB.
    
    # Let's implement Thompson Sampling with K=5 samples.
    # For each candidate, draw 5 samples from N(mu, sigma^2).
    # The score is the negative of the minimum of these samples.
    # Why minimum? Because we want to bet on the best possible outcome from the posterior.
    # This is similar to a "best-of-K" strategy.
    
    K = 5
    
    # Vectorized sampling
    # shape: (K, 2000)
    samples = mu[None, :] + sigma_safe[None, :] * np.random.randn(K, len(mu))
    
    # Check time
    if time.time() > deadline - 10.0:
        # Fallback to simple UCB if running out of time
        kappa = 2.0
        scores = kappa * sigma_safe - mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores
    
    # For minimization, we want the lowest value.
    # The "best" sample for a candidate is the minimum of the K samples.
    # We want the candidate whose best sample is the lowest.
    # So score = -min(samples, axis=0)
    
    best_samples = np.min(samples, axis=0)
    scores = -best_samples
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
